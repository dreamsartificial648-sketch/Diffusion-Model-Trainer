# Adapted from Hugging Face diffusers examples/unconditional_image_generation/train_unconditional.py
# Optimized for Ampere GPUs with 12GB VRAM (e.g. RTX 3060), also runs fine on 6GB cards.

import argparse
import inspect
import json
import logging
import math
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import Dataset, load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

logger = get_logger(__name__, log_level="INFO")


def emit_progress(enabled, **fields):
    """Print a single JSON line the desktop app can parse for live progress/ETA.

    Prefixed with PROGRESS_JSON: so it's trivially greppable out of normal
    log/tqdm noise on stdout.
    """
    if not enabled:
        return
    payload = {"ts": time.time(), **fields}
    print(f"PROGRESS_JSON:{json.dumps(payload)}", flush=True)


# Formats PIL can actually open as static images. GIF is deliberately
# excluded: PIL loads only its first frame and the result is frequently
# palette-mode or otherwise malformed for training purposes, so we treat
# GIFs the same as any other unsupported file rather than half-supporting
# them.
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}


def scan_image_folder(data_dir, gui_progress=False):
    """Recursively collect usable image files from data_dir, regardless of
    subfolder layout (no train/test split structure required).

    Skips:
      - GIFs (animated or not — see SUPPORTED_IMAGE_FORMATS above)
      - files that don't open as an image at all (corrupt, wrong extension,
        non-image junk that happened to be in the folder)

    Returns a sorted list of absolute file paths. Raises ValueError if
    nothing usable is found, with a message that explains what was skipped
    rather than just "no examples".
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Dataset folder does not exist: {data_dir}")

    candidates = [p for p in data_dir.rglob("*") if p.is_file()]
    valid_paths = []
    skipped_gif = 0
    skipped_unreadable = 0

    for path in candidates:
        try:
            with Image.open(path) as img:
                img_format = img.format
        except Exception:
            skipped_unreadable += 1
            continue

        if img_format == "GIF":
            skipped_gif += 1
            continue
        if img_format not in SUPPORTED_IMAGE_FORMATS:
            skipped_unreadable += 1
            continue

        valid_paths.append(str(path))

    if skipped_gif:
        logger.info(f"Skipped {skipped_gif} GIF file(s) (not supported for training).")
    if skipped_unreadable:
        logger.info(f"Skipped {skipped_unreadable} file(s) that weren't readable images.")

    if not valid_paths:
        raise ValueError(
            f"No usable images found in {data_dir}. "
            f"Scanned {len(candidates)} file(s); skipped {skipped_gif} GIF(s) and "
            f"{skipped_unreadable} unreadable/unsupported file(s)."
        )

    valid_paths.sort()

    emit_progress(
        gui_progress,
        event="dataset_scan",
        total_files=len(candidates),
        usable_images=len(valid_paths),
        skipped_gif=skipped_gif,
        skipped_unreadable=skipped_unreadable,
        example_images=valid_paths[:8],
    )

    return valid_paths


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Extract values from a 1-D numpy array for a batch of indices."""
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _ensure_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure the tensor has exactly three channels (C, H, W)."""
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    channels = tensor.shape[0]
    if channels == 3:
        return tensor
    if channels == 1:
        return tensor.repeat(3, 1, 1)
    if channels == 2:
        return torch.cat([tensor, tensor[:1]], dim=0)
    if channels > 3:
        return tensor[:3]
    raise ValueError(f"Unsupported number of channels: {channels}")




def is_stop_requested(args):
    """Cooperative stop used by the desktop app.

    The GUI writes a tiny flag file instead of killing the Python process.
    That lets training finish the current step, save the current pipeline,
    and exit cleanly instead of leaving a half-written model behind.
    """
    return bool(args.stop_signal_file and os.path.exists(args.stop_signal_file))


def dir_size_bytes(path):
    """Best-effort recursive folder size for progress messages."""
    total = 0
    path = Path(path)
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def format_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024


def cleanup_training_artifacts(output_dir, gui_progress=False, keep_latest_checkpoint=False):
    """Delete heavy training-only artifacts while keeping the loadable pipeline.

    The final Diffusers pipeline only needs model_index.json, scheduler/, and unet/.
    Accelerate checkpoint-* folders can contain optimizer state, scheduler state,
    RNG state, and EMA training weights, so they often become far larger than
    the model itself. This cleanup is what keeps saved models in the MB/low-GB
    range instead of exploding into dozens or hundreds of GB.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 0, []

    removed_bytes = 0
    removed_items = []

    checkpoints = sorted(
        [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if keep_latest_checkpoint and checkpoints:
        checkpoints = checkpoints[:-1]

    trash_targets = list(checkpoints)

    # TensorBoard/W&B logs are useful while debugging, but not needed for generating.
    # Keep GUI previews because they are tiny and useful to inspect model history.
    for name in ["logs", "runs", "wandb", "gui_stop_training.flag"]:
        p = output_dir / name
        if p.exists():
            trash_targets.append(p)

    for target in trash_targets:
        try:
            size_before = dir_size_bytes(target) if target.is_dir() else target.stat().st_size
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            removed_bytes += size_before
            removed_items.append(target.name)
        except OSError as exc:
            logger.warning(f"Could not remove training artifact {target}: {exc}")

    if removed_items:
        emit_progress(
            gui_progress,
            event="storage_cleanup",
            output_dir=str(output_dir),
            removed_bytes=removed_bytes,
            removed_human=format_bytes(removed_bytes),
            removed_items=removed_items[:25],
            final_size_bytes=dir_size_bytes(output_dir),
            final_size_human=format_bytes(dir_size_bytes(output_dir)),
        )
    return removed_bytes, removed_items


def save_pipeline_snapshot(accelerator, model, noise_scheduler, args, ema_model=None, epoch=None, stopped=False):
    """Save a lightweight DDPMPipeline that can be loaded by the Generate tab."""
    if not accelerator.is_main_process:
        return None
    unet = accelerator.unwrap_model(model)
    if args.use_ema and ema_model is not None:
        ema_model.store(unet.parameters())
        ema_model.copy_to(unet.parameters())
    pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
    pipeline.save_pretrained(args.output_dir, safe_serialization=True)
    if args.use_ema and ema_model is not None:
        ema_model.restore(unet.parameters())

    if args.storage_saver:
        cleanup_training_artifacts(
            args.output_dir,
            gui_progress=args.gui_progress,
            keep_latest_checkpoint=args.keep_latest_resume_checkpoint,
        )

    final_size = dir_size_bytes(args.output_dir)
    emit_progress(
        args.gui_progress,
        event="model_saved",
        output_dir=args.output_dir,
        epoch=epoch,
        stopped=stopped,
        size_bytes=final_size,
        size_human=format_bytes(final_size),
        storage_saver=args.storage_saver,
    )
    return pipeline


def save_preview_grid(images, output_dir, epoch, gui_progress=False):
    """Save generated samples as a visible PNG grid for the desktop app."""
    try:
        from PIL import Image as PILImage
        images_processed = (images * 255).round().clip(0, 255).astype("uint8")
        pil_images = [PILImage.fromarray(img) for img in images_processed]
        if not pil_images:
            return None
        cols = min(4, len(pil_images))
        rows = math.ceil(len(pil_images) / cols)
        w, h = pil_images[0].size
        grid = PILImage.new("RGB", (cols * w, rows * h), (20, 22, 28))
        for i, img in enumerate(pil_images):
            grid.paste(img.convert("RGB"), ((i % cols) * w, (i // cols) * h))
        preview_dir = Path(output_dir) / "gui_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        path = preview_dir / f"preview_epoch_{epoch:04d}.png"
        grid.save(path)
        emit_progress(gui_progress, event="preview_image", epoch=epoch, path=str(path))
        return str(path)
    except Exception as exc:
        emit_progress(gui_progress, event="preview_error", message=str(exc)[:300])
        return None

def _identity(x):
    """No-op transform (used in place of a lambda when random_flip is off).

    Must be a real top-level function, not a lambda or closure: on Windows,
    DataLoader workers are spawned as fresh processes and the whole transform
    pipeline has to be pickled to hand off to them. Lambdas and local
    functions can't be pickled, which breaks multi-worker loading on Windows
    (it's silently fine on Linux/Mac, which use fork instead of spawn).
    """
    return x


class ImageTransform:
    """Picklable replacement for a closure-based transform function.

    Same reasoning as _identity above: this used to be a function defined
    inside main() that closed over `args`, `augmentations`, etc. That's
    invisible on Linux (fork-based workers) but crashes immediately on
    Windows with "Can't pickle local object" as soon as dataloader_num_workers
    is set above 0. Storing the same state as attributes on a top-level class
    instance keeps it picklable everywhere.
    """

    def __init__(self, augmentations, precision_augmentations, preserve_input_precision):
        self.augmentations = augmentations
        self.precision_augmentations = precision_augmentations
        self.preserve_input_precision = preserve_input_precision

    def __call__(self, examples):
        processed = []
        for image in examples["image"]:
            if not self.preserve_input_precision:
                processed.append(self.augmentations(image.convert("RGB")))
            else:
                precise_image = image
                if precise_image.mode == "P":
                    precise_image = precise_image.convert("RGB")
                processed.append(self.precision_augmentations(precise_image))
        return {"input": processed}


def parse_args():
    parser = argparse.ArgumentParser(description="Train unconditional DDPM on your images.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace dataset name or path to local dataset.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Dataset config name.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="UNet config path. Leave None for default DDPM config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="data",
        help="Folder containing training images. Use structure: data_dir/train/*.png (default: data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/model",
        help="Output directory for checkpoints and final model.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Optional full DDPMPipeline folder to load before continuing/fine-tuning training.",
    )
    parser.add_argument(
        "--stop_signal_file",
        type=str,
        default=None,
        help="If this file appears during GUI training, save the current model and stop cleanly.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Input resolution (64 for 6GB VRAM, 128+ comfortable on 12GB Ampere cards).",
    )
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true", help="Random horizontal flip augmentation.")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size per device. 1 is safe for 6GB VRAM; 4 is a good starting point on 12GB Ampere cards.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Background processes that load/augment images in parallel with GPU "
        "training. 0 forces the main process to do this serially between every "
        "step, which leaves the GPU idle waiting on the CPU. 2-4 is a good "
        "default on a modern desktop CPU; lower if you see system stutter.",
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients to simulate a larger batch. Raise this if you lower train_batch_size.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Use fp16 for 6GB VRAM.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument(
        "--ddpm_num_inference_steps",
        type=int,
        default=1000,
        help="Denoising steps used for the FINAL saved model's quality. Leave high (e.g. 1000).",
    )
    parser.add_argument(
        "--preview_num_inference_steps",
        type=int,
        default=100,
        help="Denoising steps used only for in-training preview grids. Lower than "
        "ddpm_num_inference_steps on purpose: previews just need to look "
        "recognizable, not perfect, so we don't want to pay full sampling cost "
        "every save_images_epochs during training. Final model is unaffected.",
    )
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=0,
        help="Save heavy Accelerate resume checkpoints every N optimizer steps. 0 disables them. "
        "The final Generate-ready model is still saved by --save_model_epochs.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help="Maximum number of heavy checkpoint-* resume folders to keep when checkpointing is enabled.",
    )
    parser.add_argument(
        "--storage_saver",
        type=lambda x: x.lower() != "false",
        default=True,
        help="Delete training-only artifacts after saving the loadable DDPM pipeline. Default: true.",
    )
    parser.add_argument(
        "--keep_latest_resume_checkpoint",
        action="store_true",
        help="With --storage_saver, keep the newest checkpoint-* folder for exact resume. Off by default for smallest models.",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--preserve_input_precision", action="store_true")
    parser.add_argument(
        "--tf32",
        type=lambda x: x.lower() != "false",
        default=True,
        help="Enable TF32 matmul/conv on Ampere+ GPUs (RTX 30-series and newer). Free speedup, default on.",
    )
    parser.add_argument(
        "--channels_last",
        action="store_true",
        default=True,
        help="Use channels-last memory format. Speeds up conv-heavy UNets on Tensor Core GPUs.",
    )
    parser.add_argument(
        "--gui_progress",
        action="store_true",
        help="Emit machine-readable JSON progress lines to stdout (used by the desktop app).",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify --dataset_name or --train_data_dir.")

    return args


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard" and not is_tensorboard_available():
        raise ImportError("Install tensorboard for logging.")
    elif args.logger == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb for logging.")

    # Ampere+ GPUs (RTX 30-series and newer) expose TF32 tensor-core math for
    # fp32 matmuls/convolutions. Same numerical range as fp32, ~10 bits less
    # mantissa precision, and up to ~3x throughput. No extra VRAM or power
    # draw cost since it reuses the existing fp32 storage path; it's simply
    # an unused fast path on Turing/Pascal cards (no-op there, safe to leave on).
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model
            for i in range(len(models)):
                model = models.pop()
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.storage_saver and not args.resume_from_checkpoint:
            # Clear old bulky training folders from a previous run in this same output directory.
            cleanup_training_artifacts(
                args.output_dir,
                gui_progress=args.gui_progress,
                keep_latest_checkpoint=args.keep_latest_resume_checkpoint,
            )

    if args.push_to_hub and accelerator.is_main_process:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name,
            exist_ok=True,
            token=args.hub_token,
        ).repo_id

    if args.pretrained_model_path:
        model_path = Path(args.pretrained_model_path)
        if not (model_path / "model_index.json").exists():
            raise ValueError(
                f"Selected model is not a complete DDPMPipeline folder: {model_path}. "
                "Pick a folder that contains model_index.json."
            )
        logger.info(f"Loading trained model to continue/fine-tune: {model_path}")
        pipeline = DDPMPipeline.from_pretrained(str(model_path))
        model = pipeline.unet
        loaded_size = getattr(model.config, "sample_size", None)
        if loaded_size is not None and int(loaded_size) != int(args.resolution):
            logger.warning(
                f"Loaded model sample_size={loaded_size}, but training resolution={args.resolution}. "
                "For safest continuation, use the same resolution as the trained model."
            )
        del pipeline
    elif args.model_config_name_or_path is None:
        model = UNet2DModel(
            sample_size=args.resolution,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        model = UNet2DModel.from_config(config)

    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # channels_last lets cuDNN pick faster tensor-core kernels for the
    # conv-heavy UNet on Ampere+ GPUs. Safe no-op on CPU/older GPUs.
    if args.channels_last and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
        else:
            logger.warning(
                "xformers not available; falling back to PyTorch's built-in "
                "scaled_dot_product_attention (already efficient on Ampere+)."
            )

    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        # Scan the folder ourselves rather than delegating to imagefolder's
        # split auto-detection (which expects subfolders literally named
        # train/test/validation and errors out otherwise). This also lets
        # us filter out GIFs and unreadable files up front.
        image_paths = scan_image_folder(args.train_data_dir, gui_progress=args.gui_progress)
        dataset = Dataset.from_dict({"image": image_paths}).cast_column("image", datasets.Image())

    spatial_augmentations = [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(_identity),
    ]

    augmentations = transforms.Compose(
        spatial_augmentations
        + [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    precision_augmentations = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Lambda(_ensure_three_channels),
            transforms.ConvertImageDtype(torch.float32),
        ]
        + spatial_augmentations
        + [transforms.Normalize([0.5], [0.5])]
    )

    transform_images = ImageTransform(augmentations, precision_augmentations, args.preserve_input_precision)

    logger.info(f"Dataset size: {len(dataset)}")
    dataset.set_transform(transform_images)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        persistent_workers=args.dataloader_num_workers > 0,
        prefetch_factor=2 if args.dataloader_num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    if accelerator.is_main_process:
        run = os.path.splitext(os.path.basename(__file__))[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    emit_progress(
        args.gui_progress and accelerator.is_main_process,
        event="start",
        num_epochs=args.num_epochs,
        steps_per_epoch=num_update_steps_per_epoch,
        total_steps=max_train_steps,
    )

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1])) if dirs else []
            path = dirs[-1] if dirs else None

        if path is None:
            logger.info("Checkpoint not found. Starting fresh.")
            args.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    else:
        resume_step = 0

    if args.stop_signal_file and accelerator.is_main_process:
        try:
            os.remove(args.stop_signal_file)
        except FileNotFoundError:
            pass

    epoch_durations = []  # rolling history, used to smooth the ETA estimate
    stop_requested = False

    def eta_reference_durations(durations):
        """Durations to average for ETA purposes.

        Epoch 0 includes one-time startup costs that never recur on later
        epochs: CUDA kernel compilation, cuDNN algorithm benchmarking, pinned
        memory buffer setup, and a cold OS file cache for the dataset. Left
        in the average, epoch 0 alone can make the ETA look many times worse
        than the run actually is (e.g. showing "2 hours remaining" early on,
        for a run that settles into 35 minutes once warmed up).

        Once we have at least one post-warmup epoch to use instead, drop
        epoch 0 from the average. While still inside epoch 0 itself, there's
        nothing else to go on yet, so use it anyway rather than show no ETA.
        """
        if len(durations) > 1:
            return durations[1:]
        return durations

    for epoch in range(first_epoch, args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            if is_stop_requested(args):
                stop_requested = True
                logger.info("GUI stop requested. Saving current model before exiting...")
                save_pipeline_snapshot(accelerator, model, noise_scheduler, args, ema_model if args.use_ema else None, epoch=epoch, stopped=True)
                emit_progress(args.gui_progress and accelerator.is_main_process, event="stopped", output_dir=args.output_dir, epoch=epoch)
                break
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(model):
                clean_images = batch["input"].to(weight_dtype)
                if args.channels_last and clean_images.is_cuda:
                    clean_images = clean_images.to(memory_format=torch.channels_last)
                noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
                bsz = clean_images.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
                ).long()

                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                model_output = model(noisy_images, timesteps).sample

                if args.prediction_type == "epsilon":
                    loss = F.mse_loss(model_output.float(), noise.float())
                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = (snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")).mean()
                else:
                    raise ValueError(f"Unsupported prediction_type: {args.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    if args.use_ema:
                        ema_model.step(model.parameters())
                    progress_bar.update(1)
                    global_step += 1

                    if (
                        accelerator.is_main_process
                        and args.checkpointing_steps
                        and args.checkpointing_steps > 0
                        and global_step % args.checkpointing_steps == 0
                    ):
                        if args.checkpoints_total_limit is not None:
                            checkpoints = sorted(
                                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else -1,
                            )
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                for cp in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                    shutil.rmtree(os.path.join(args.output_dir, cp))

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved resume checkpoint to {save_path}")

                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                    if args.use_ema:
                        logs["ema_decay"] = ema_model.cur_decay_value
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    if args.gui_progress and accelerator.is_main_process:
                        eta_durations = eta_reference_durations(epoch_durations)
                        avg_epoch_s = sum(eta_durations) / len(eta_durations) if eta_durations else None
                        epochs_remaining = args.num_epochs - epoch - 1
                        eta_seconds = None
                        if avg_epoch_s is not None:
                            frac_done_this_epoch = min(1.0, (step + 1) / max(1, num_update_steps_per_epoch))
                            eta_seconds = avg_epoch_s * (epochs_remaining + (1 - frac_done_this_epoch))
                        emit_progress(
                            True,
                            event="step",
                            epoch=epoch,
                            global_step=global_step,
                            total_steps=max_train_steps,
                            loss=logs["loss"],
                            lr=logs["lr"],
                            eta_seconds=eta_seconds,
                        )

        progress_bar.close()
        if stop_requested:
            break
        accelerator.wait_for_everyone()

        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)

        if args.gui_progress and accelerator.is_main_process:
            eta_durations = eta_reference_durations(epoch_durations)
            avg_epoch_s = sum(eta_durations) / len(eta_durations)
            epochs_remaining = args.num_epochs - epoch - 1
            emit_progress(
                True,
                event="epoch_end",
                epoch=epoch,
                epoch_duration_seconds=epoch_duration,
                avg_epoch_seconds=avg_epoch_s,
                eta_seconds=avg_epoch_s * epochs_remaining,
            )

        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)
                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                images = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.preview_num_inference_steps,
                    output_type="np",
                ).images

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                images_processed = (images * 255).round().astype("uint8")
                save_preview_grid(images, args.output_dir, epoch, gui_progress=args.gui_progress)
                if args.logger == "tensorboard":
                    tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                pipeline = save_pipeline_snapshot(
                    accelerator, model, noise_scheduler, args, ema_model if args.use_ema else None, epoch=epoch
                )

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

    if stop_requested:
        accelerator.end_training()
        return

    accelerator.end_training()
    if args.storage_saver and accelerator.is_main_process:
        cleanup_training_artifacts(
            args.output_dir,
            gui_progress=args.gui_progress,
            keep_latest_checkpoint=args.keep_latest_resume_checkpoint,
        )
    emit_progress(args.gui_progress and accelerator.is_main_process, event="done", output_dir=args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as exc:
        emit_progress(args.gui_progress, event="error", message=str(exc)[:500])
        raise
