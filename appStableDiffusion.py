"""
Desktop app for unconditional image generation (DDPM diffusion).

Two tabs:
  - Train:    configure and launch training (runs train.py as a subprocess),
              with a live progress bar and an ETA that gets more accurate
              every epoch.
  - Generate: pick a trained checkpoint (or the latest one automatically)
              and generate an image.
"""

import json
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
BG = "#1b1d23"
BG_PANEL = "#22252e"
BG_FIELD = "#2a2e38"
FG = "#e8e9ed"
FG_DIM = "#9a9ea8"
ACCENT = "#5b8cff"
ACCENT_DIM = "#3d4a6b"
GOOD = "#5ec98f"
WARN = "#e0a23c"
BAD = "#e0615f"
FONT_FAMILY = "Segoe UI"

# ----------------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT_DIR / "train.py"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "model"
DEFAULT_MODELS_ROOT = ROOT_DIR / "output"
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_GENERATIONS_DIR = ROOT_DIR / "output" / "generations"
MODEL_INFO_FILE = "model_info.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
PRETRAINED_MODEL = "anton-l/ddpm-butterflies-128"

PIPELINE = None  # lazily-loaded diffusers pipeline, cached across generations
PIPELINE_SOURCE = None  # path/name the cached pipeline was loaded from
PIPELINE_SCHEDULER_CONFIG = None  # original scheduler config, used when swapping samplers

SAMPLER_DESCRIPTIONS = {
    "DDPM": (
        "Classic denoising diffusion sampler. Slower, very stable, and a good baseline. "
        "Use this when you want the normal DDPM dream look."
    ),
    "DDIM": (
        "Denoising Diffusion Implicit Model sampler. Usually much faster at lower step counts, "
        "more deterministic with the same seed, and great for quick previews/generation."
    ),
}


# ----------------------------------------------------------------------------
# Model discovery
# ----------------------------------------------------------------------------
def is_loadable_model_dir(path: Path) -> bool:
    """A complete diffusers DDPMPipeline folder contains model_index.json."""
    path = Path(path).expanduser()
    return path.exists() and path.is_dir() and (path / "model_index.json").exists()


def looks_like_resume_checkpoint(path: Path) -> bool:
    """Accelerate checkpoint-* folders are resume states, not Generate-ready models."""
    path = Path(path).expanduser()
    return path.is_dir() and path.name.startswith("checkpoint-")


def find_loadable_model_inside(path: Path, max_depth: int = 4):
    """Return a loadable DDPMPipeline folder from path or one of its children.

    This fixes the annoying case where you pick output/, an experiment folder,
    or a parent folder and the actual model_index.json is one level deeper.
    """
    path = Path(path).expanduser()
    if is_loadable_model_dir(path):
        return path
    if not path.exists() or not path.is_dir():
        return None

    best = None
    best_mtime = -1
    root_parts = len(path.parts)
    for model_index in path.rglob("model_index.json"):
        model_dir = model_index.parent
        if len(model_dir.parts) - root_parts > max_depth:
            continue
        try:
            mtime = model_dir.stat().st_mtime
        except OSError:
            mtime = 0
        if mtime > best_mtime:
            best = model_dir
            best_mtime = mtime
    return best


def describe_model_problem(path: Path) -> str:
    path = Path(path).expanduser()
    if looks_like_resume_checkpoint(path):
        return (
            "That looks like an Accelerate resume checkpoint folder.\n\n"
            "checkpoint-* folders are for continuing training, not for generating images. "
            "Pick the saved model folder that contains model_index.json instead."
        )
    if path.exists() and path.is_dir():
        return (
            "That folder is not a complete Diffusers DDPMPipeline model.\n\n"
            "A loadable model folder needs model_index.json at the top level. "
            "Try selecting the experiment/model folder that contains model_index.json, "
            "or select a parent folder and let the app scan inside it."
        )
    return f"Folder does not exist:\n{path}"


def get_model_resolution(path: Path):
    config_path = Path(path) / "unet" / "config.json"
    try:
        data = json.loads(config_path.read_text())
        sample_size = data.get("sample_size")
        if isinstance(sample_size, (list, tuple)):
            return "x".join(str(x) for x in sample_size)
        if sample_size:
            return f"{sample_size}x{sample_size}"
    except Exception:
        pass
    return "?"


def get_model_modified(path: Path) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(Path(path).stat().st_mtime))
    except Exception:
        return "?"


def sanitize_filename_component(name: str, fallback="Model") -> str:
    """Make a friendly model/generation name safe for Windows/macOS/Linux folders."""
    name = str(name or "").strip()
    bad_chars = '<>:"/\\|?*'
    cleaned = []
    for ch in name:
        if ch in bad_chars or ord(ch) < 32:
            cleaned.append("_")
        else:
            cleaned.append(ch)
    cleaned = "".join(cleaned).strip(" ._")
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    if not cleaned:
        cleaned = fallback
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if cleaned.upper() in reserved:
        cleaned = f"{cleaned}_Model"
    return cleaned[:80]


def unique_child_dir(parent: Path, desired_name: str) -> Path:
    """Return parent/desired_name, or parent/desired_name 1, 2, 3... if needed."""
    parent = Path(parent).expanduser()
    base = sanitize_filename_component(desired_name)
    candidate = parent / base
    counter = 1
    while candidate.exists():
        candidate = parent / f"{base} {counter}"
        counter += 1
    return candidate


def read_model_metadata(path: Path):
    path = Path(path).expanduser()
    metadata_path = path / MODEL_INFO_FILE
    try:
        if metadata_path.exists():
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def get_friendly_model_name(path: Path) -> str:
    metadata = read_model_metadata(path)
    name = metadata.get("model_name") or Path(path).name or "Model"
    return sanitize_filename_component(name, fallback="Model")


def get_model_info(path: Path):
    path = Path(path).expanduser()
    metadata = read_model_metadata(path)
    model_name = sanitize_filename_component(metadata.get("model_name") or path.name or str(path), fallback="Model")
    try:
        display = str(path.relative_to(ROOT_DIR))
    except ValueError:
        display = str(path)
    return {
        "name": model_name,
        "model_name": model_name,
        "path": str(path),
        "display": display,
        "resolution": get_model_resolution(path),
        "modified": get_model_modified(path),
        "metadata": metadata,
    }


def model_label(path: Path, prefix="Model") -> str:
    info = get_model_info(path)
    return f"{prefix}: {info['model_name']}  ({info['resolution']}, {info['modified']})"


def find_checkpoints(output_dir: Path = DEFAULT_MODELS_ROOT):
    """Return every loadable DDPMPipeline folder under output_dir, newest first."""
    roots = []
    for candidate in [DEFAULT_OUTPUT_DIR, output_dir, DEFAULT_MODELS_ROOT]:
        candidate = Path(candidate).expanduser()
        if candidate.exists() and candidate not in roots:
            roots.append(candidate)

    found = {}
    for root in roots:
        resolved = find_loadable_model_inside(root) if not is_loadable_model_dir(root) else root
        if resolved is not None and is_loadable_model_dir(resolved):
            found[str(resolved.resolve())] = resolved
        if root.exists():
            for model_index in root.rglob("model_index.json"):
                path = model_index.parent
                found[str(path.resolve())] = path

    results = []
    for path in found.values():
        mtime = path.stat().st_mtime
        info = get_model_info(path)
        label = f"Latest trained model: {info['model_name']}" if path.resolve() == DEFAULT_OUTPUT_DIR.resolve() else model_label(path)
        results.append((label, str(path), mtime, info))
    results.sort(key=lambda r: r[2], reverse=True)
    return results


def copy_model_folder(src: Path, dst: Path):
    src = Path(src).expanduser()
    dst = Path(dst).expanduser()
    src = find_loadable_model_inside(src) or src
    if not is_loadable_model_dir(src):
        raise ValueError(describe_model_problem(src))
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"Destination already exists and is not empty: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def open_folder(path: Path):
    path = Path(path).expanduser()
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        messagebox.showerror("Could not open folder", str(exc))


# ----------------------------------------------------------------------------
# Generation (runs in a background thread, in-process — inference is quick
# and doesn't need subprocess isolation the way training does)
# ----------------------------------------------------------------------------
def load_pipeline(model_path_or_name: str):
    global PIPELINE, PIPELINE_SOURCE, PIPELINE_SCHEDULER_CONFIG
    if PIPELINE is not None and PIPELINE_SOURCE == model_path_or_name:
        return PIPELINE

    import torch
    from diffusers import DDPMPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = DDPMPipeline.from_pretrained(model_path_or_name)
    pipeline = pipeline.to(device, dtype)

    if device == "cuda":
        try:
            pipeline.enable_attention_slicing()
        except Exception:
            pass

    PIPELINE = pipeline
    PIPELINE_SOURCE = model_path_or_name
    PIPELINE_SCHEDULER_CONFIG = dict(pipeline.scheduler.config)
    return pipeline


def apply_sampler_to_pipeline(pipeline, sampler="DDPM"):
    """Swap only the inference scheduler; the trained UNet/model stays the same."""
    global PIPELINE_SCHEDULER_CONFIG
    from diffusers import DDIMScheduler, DDPMScheduler

    sampler = (sampler or "DDPM").upper()
    config = PIPELINE_SCHEDULER_CONFIG or dict(pipeline.scheduler.config)
    if sampler == "DDIM":
        pipeline.scheduler = DDIMScheduler.from_config(config)
    else:
        pipeline.scheduler = DDPMScheduler.from_config(config)
    return pipeline


def generate_images(model_path_or_name: str, seed=None, num_inference_steps=50, batch_size=1, sampler="DDPM"):
    """Run inference and return a list of PIL Images.

    Resolution is baked into the trained UNet sample_size, so the Generate tab
    shows it as model info instead of pretending it can be changed safely.
    """
    import torch

    pipeline = load_pipeline(model_path_or_name)
    pipeline = apply_sampler_to_pipeline(pipeline, sampler=sampler)
    device = next(pipeline.unet.parameters()).device
    generator = None
    if seed is not None:
        # For a batch, make deterministic-but-different samples from the same seed.
        generator = [torch.Generator(device=device).manual_seed(int(seed) + i) for i in range(int(batch_size))]

    images = pipeline(
        generator=generator,
        batch_size=int(batch_size),
        num_inference_steps=int(num_inference_steps),
        output_type="pil",
    ).images
    return images


def generate_image(model_path_or_name: str, seed=None, sampler="DDPM"):
    return generate_images(model_path_or_name, seed=seed, num_inference_steps=50, batch_size=1, sampler=sampler)[0]


def _get_model_sample_size(unet):
    sample_size = getattr(unet.config, "sample_size", 128)
    if isinstance(sample_size, (list, tuple)) and len(sample_size) >= 2:
        return int(sample_size[0]), int(sample_size[1])
    return int(sample_size), int(sample_size)


def _normalize_noise_tensor(noise):
    import torch
    flat = noise.view(noise.shape[0], -1)
    denom = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    flat = flat / denom
    return flat.view_as(noise)


def _pil_to_model_tensor(image, size, device, dtype):
    import numpy as np
    import torch
    from PIL import Image as PILImage

    if isinstance(size, int):
        width = height = size
    else:
        width, height = size
    image = image.convert("RGB").resize((width, height), PILImage.BILINEAR)
    arr = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device=device, dtype=dtype)


def _model_tensor_to_pil(image_tensor):
    import numpy as np
    from PIL import Image as PILImage

    image = (image_tensor / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    return PILImage.fromarray(image)


def _sample_with_pipeline_from_noise(pipeline, initial_noise, num_inference_steps=50):
    image = initial_noise.clone()
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(int(num_inference_steps))
    for t in scheduler.timesteps:
        model_output = pipeline.unet(image, t).sample
        image = scheduler.step(model_output, t, image).prev_sample
    return image


def _sample_with_pipeline_from_reference(pipeline, reference_image, initial_noise, num_inference_steps=50, reference_influence=0.6):
    import torch

    device = next(pipeline.unet.parameters()).device
    dtype = next(pipeline.unet.parameters()).dtype
    sample_w, sample_h = _get_model_sample_size(pipeline.unet)
    reference_tensor = _pil_to_model_tensor(reference_image, (sample_w, sample_h), device, dtype)

    scheduler = pipeline.scheduler
    scheduler.set_timesteps(int(num_inference_steps))
    timesteps = scheduler.timesteps
    if len(timesteps) == 0:
        raise RuntimeError("Scheduler produced no timesteps.")

    # Higher reference influence means we start closer to the clean reference image
    # (less added noise, more preserved structure).
    influence = max(0.0, min(1.0, float(reference_influence)))
    start_index = int(round(influence * max(0, len(timesteps) - 1)))
    start_index = max(0, min(start_index, len(timesteps) - 1))
    start_t = timesteps[start_index]

    if not isinstance(start_t, torch.Tensor):
        start_t_tensor = torch.tensor([int(start_t)], device=device, dtype=torch.long)
    else:
        start_t_tensor = start_t.reshape(1).to(device=device, dtype=torch.long)

    image = scheduler.add_noise(reference_tensor, initial_noise, start_t_tensor)
    for t in timesteps[start_index:]:
        model_output = pipeline.unet(image, t).sample
        image = scheduler.step(model_output, t, image).prev_sample
    return image


def sample_reference_video_frames(video_path, target_frames):
    import imageio.v2 as imageio
    from PIL import Image as PILImage

    if target_frames <= 0:
        return []

    reader = imageio.get_reader(str(video_path))
    frames = []
    try:
        try:
            length = reader.count_frames()
        except Exception:
            length = reader.get_length()

        if length is None or length <= 0 or length == float("inf"):
            cached = []
            for frame in reader:
                cached.append(frame)
                if len(cached) >= max(target_frames, 1):
                    break
            if not cached:
                return []
            if len(cached) >= target_frames:
                indices = [round(i * (len(cached) - 1) / max(1, target_frames - 1)) for i in range(target_frames)]
                return [PILImage.fromarray(cached[idx]).convert("RGB") for idx in indices]
            while len(cached) < target_frames:
                cached.append(cached[-1])
            return [PILImage.fromarray(frame).convert("RGB") for frame in cached[:target_frames]]

        indices = [round(i * (length - 1) / max(1, target_frames - 1)) for i in range(target_frames)]
        for idx in indices:
            frame = reader.get_data(int(idx))
            frames.append(PILImage.fromarray(frame).convert("RGB"))
        return frames
    finally:
        try:
            reader.close()
        except Exception:
            pass


def generate_video_frames(
    model_path_or_name: str,
    seconds=4.0,
    fps=24,
    seed=None,
    num_inference_steps=60,
    smoothness=85,
    reference_video_path=None,
    reference_influence=0.6,
    reference_mode="Dreamify/Reconstruct",
    source_preservation=0.55,
    progress_callback=None,
):
    import torch

    pipeline = load_pipeline(model_path_or_name)
    device = next(pipeline.unet.parameters()).device
    dtype = next(pipeline.unet.parameters()).dtype
    sample_w, sample_h = _get_model_sample_size(pipeline.unet)

    total_frames = max(1, int(round(float(seconds) * float(fps))))
    base_seed = int(seed) if seed is not None else random.randint(0, 2_147_483_647)
    generator = torch.Generator(device=device).manual_seed(base_seed)

    current_noise = torch.randn((1, 3, sample_h, sample_w), generator=generator, device=device, dtype=dtype)
    current_noise = _normalize_noise_tensor(current_noise)

    # High smoothness = slower drift in latent space, lower smoothness = more change per frame.
    smoothness = max(1, min(100, int(smoothness)))
    drift_alpha = max(0.03, 1.0 - (smoothness / 100.0) * 0.95)

    reference_frames = None
    if reference_video_path:
        reference_frames = sample_reference_video_frames(reference_video_path, total_frames)
        if not reference_frames:
            raise RuntimeError("Could not read frames from the reference video.")

    frames = []
    for frame_index in range(total_frames):
        if frame_index > 0:
            fresh_noise = torch.randn(current_noise.shape, generator=generator, device=device, dtype=dtype)
            current_noise = _normalize_noise_tensor((1.0 - drift_alpha) * current_noise + drift_alpha * fresh_noise)

        ref_image = reference_frames[min(frame_index, len(reference_frames) - 1)] if reference_frames else None
        if ref_image is not None:
            frame_tensor = _sample_with_pipeline_from_reference(
                pipeline,
                ref_image,
                current_noise,
                num_inference_steps=num_inference_steps,
                reference_influence=reference_influence,
            )
        else:
            frame_tensor = _sample_with_pipeline_from_noise(
                pipeline,
                current_noise,
                num_inference_steps=num_inference_steps,
            )

        frame_image = _model_tensor_to_pil(frame_tensor)

        # APVD-inspired Dreamify/Reconstruct mode:
        # keep the reference frame as the structural anchor and let DDPM only
        # push it into the model's dream style. This avoids the "same silhouette,
        # totally different object" problem caused by full DDPM reimagination.
        if ref_image is not None and str(reference_mode).lower().startswith("dreamify"):
            preserve = max(0.0, min(1.0, float(source_preservation)))
            dream_amount = 1.0 - preserve
            ref_resized = ref_image.convert("RGB").resize(frame_image.size)
            
            if dream_amount <= 0:
                frame_image = ref_resized
            else:
                from PIL import Image as PILImage
                frame_image = PILImage.blend(ref_resized, frame_image.convert("RGB"), dream_amount)

        frames.append(frame_image)

        if progress_callback is not None:
            progress_callback(frame_index + 1, total_frames, frame_image)

    return frames


def save_video_frames(frames, output_path, fps=24):
    import numpy as np
    import imageio.v2 as imageio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(frame.convert("RGB")) for frame in frames]

    if output_path.suffix.lower() == ".gif":
        imageio.mimsave(str(output_path), arrays, fps=fps, loop=0)
        return str(output_path)

    try:
        writer = imageio.get_writer(str(output_path), fps=fps)
        for arr in arrays:
            writer.append_data(arr)
        writer.close()
        return str(output_path)
    except Exception:
        fallback = output_path.with_suffix('.gif')
        imageio.mimsave(str(fallback), arrays, fps=fps, loop=0)
        return str(fallback)


def scale_for_display(image, target_size=512):
    """Resize a generated image to a comfortably viewable size for the GUI.

    DDPM models commonly generate at small native resolutions (64-256px).
    PIL's Image.thumbnail() only ever shrinks an image that's larger than
    the target — it does nothing if the image is already smaller, which is
    why a 64x64 generation used to show up tiny in a big window. This
    always scales to fit target_size in the larger dimension, scaling up
    when needed. Uses NEAREST so small images enlarge as crisp visible
    blocks rather than a blurry mess — appropriate since the goal here is
    visibility, not photographic upscaling quality.
    """
    from PIL import Image as PILImage

    width, height = image.size
    scale = target_size / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resample = PILImage.NEAREST if scale > 1 else PILImage.LANCZOS
    return image.resize(new_size, resample=resample)


# ----------------------------------------------------------------------------
# Reusable small widgets
# ----------------------------------------------------------------------------
class LabeledEntry(ttk.Frame):
    """A label + entry box pair, e.g. for epochs, resolution, learning rate."""

    def __init__(self, parent, label, default="", width=12, tooltip=None):
        super().__init__(parent, style="Panel.TFrame")
        self.var = tk.StringVar(value=str(default))
        lbl = ttk.Label(self, text=label, style="Field.TLabel")
        lbl.pack(anchor="w")
        entry = ttk.Entry(self, textvariable=self.var, width=width, style="Field.TEntry")
        entry.pack(anchor="w", pady=(2, 0), fill="x")
        if tooltip:
            Tooltip(entry, tooltip)
            Tooltip(lbl, tooltip)

    def get(self):
        return self.var.get().strip()


class LabeledCombo(ttk.Frame):
    """A label + dropdown pair, e.g. for mixed precision."""

    def __init__(self, parent, label, values, default=None, width=12, tooltip=None):
        super().__init__(parent, style="Panel.TFrame")
        self.var = tk.StringVar(value=default if default is not None else values[0])
        lbl = ttk.Label(self, text=label, style="Field.TLabel")
        lbl.pack(anchor="w")
        combo = ttk.Combobox(
            self, textvariable=self.var, values=values, state="readonly", width=width
        )
        combo.pack(anchor="w", pady=(2, 0), fill="x")
        if tooltip:
            Tooltip(combo, tooltip)
            Tooltip(lbl, tooltip)

    def get(self):
        return self.var.get().strip()


class LabeledScale(ttk.Frame):
    """Label + slider + value readout."""

    def __init__(self, parent, label, from_, to, default, width=160, tooltip=None):
        super().__init__(parent, style="Panel.TFrame")
        self.var = tk.IntVar(value=int(default))
        row = ttk.Frame(self, style="Panel.TFrame")
        row.pack(fill="x")
        lbl = ttk.Label(row, text=label, style="Field.TLabel")
        lbl.pack(side="left")
        self.value_label = ttk.Label(row, text=str(default), style="Field.TLabel")
        self.value_label.pack(side="right")
        scale = ttk.Scale(
            self, from_=from_, to=to, orient="horizontal", variable=self.var,
            command=lambda _v: self.value_label.config(text=str(self.get())), length=width,
        )
        scale.pack(fill="x", pady=(2, 0))
        if tooltip:
            Tooltip(scale, tooltip)
            Tooltip(lbl, tooltip)

    def get(self):
        return int(round(self.var.get()))


class Tooltip:
    """Minimal hover tooltip so the settings panel can explain itself
    without needing a manual."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tip,
            text=self.text,
            background="#33384a",
            foreground=FG,
            font=(FONT_FAMILY, 9),
            padx=8,
            pady=4,
            wraplength=260,
            justify="left",
        )
        label.pack()

    def _hide(self, _event=None):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def format_eta(seconds):
    if seconds is None or seconds < 0:
        return "Calculating..."
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"~{h}h {m}m remaining"
    if m:
        return f"~{m}m {s}s remaining"
    return f"~{s}s remaining"


def parse_progress_line(line: str):
    """Extract a PROGRESS_JSON payload from a line of subprocess output.

    tqdm redraws its bar in place using \\r instead of \\n, so our
    PROGRESS_JSON: print can end up appended to the *same* physical line
    as a tqdm bar (e.g. "...31.90s/it, step=1]PROGRESS_JSON:{...}") rather
    than starting a fresh line. We therefore search for the prefix
    anywhere in the line instead of requiring it at the start.
    """
    prefix = "PROGRESS_JSON:"
    idx = line.find(prefix)
    if idx == -1:
        return None
    payload = line[idx + len(prefix):].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


CUDA_CPU_WARNING_MESSAGE = (
    "CUDA GPU not available. PyTorch is running on CPU. "
    "Training DDPM models on CPU will be extremely slow. "
    "Your GPU may be too old for modern PyTorch CUDA."
)


def _run_nvidia_smi_gpu_names():
    """Best-effort Windows/Linux NVIDIA GPU name lookup, even when PyTorch CUDA is unavailable."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0,
        )
        if completed.returncode == 0:
            names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if names:
                return names
    except Exception:
        pass
    return []


def get_cuda_startup_report():
    """Return a small CUDA/PyTorch report for user-facing startup warnings."""
    report = {
        "torch_imported": False,
        "cuda_available": False,
        "torch_version": "unknown",
        "torch_cuda_version": None,
        "device_name": None,
        "nvidia_smi_names": [],
        "error": None,
    }
    try:
        import torch
        report["torch_imported"] = True
        report["torch_version"] = getattr(torch, "__version__", "unknown")
        report["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            report["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        report["error"] = str(exc)

    if not report["cuda_available"]:
        report["nvidia_smi_names"] = _run_nvidia_smi_gpu_names()
    return report


def build_cuda_warning_text(report):
    lines = [
        CUDA_CPU_WARNING_MESSAGE,
        "",
        "Possible reasons:",
        "• Your NVIDIA GPU is too old for this version of CUDA/PyTorch.",
        "• You installed the CPU-only version of PyTorch.",
        "• Your NVIDIA driver/CUDA setup is missing or outdated.",
        "• You are using an AMD or Intel GPU. CUDA is NVIDIA-only.",
        "",
        "Recommended hardware:",
        "• GTX 1060 / GTX 1660 or newer",
        "• RTX 20, 30, 40, or newer",
        "",
        "Detected device:",
    ]
    if report.get("device_name"):
        lines.append(f"• {report['device_name']}")
    elif report.get("nvidia_smi_names"):
        for name in report["nvidia_smi_names"]:
            lines.append(f"• {name} (seen by NVIDIA driver, but not usable by PyTorch CUDA)")
    else:
        lines.append("• No supported CUDA GPU detected by PyTorch")

    if report.get("torch_imported"):
        lines.extend([
            "",
            f"PyTorch version: {report.get('torch_version')}",
            f"PyTorch CUDA build: {report.get('torch_cuda_version') or 'CPU-only / unavailable'}",
        ])
    elif report.get("error"):
        lines.extend(["", f"PyTorch import error: {report['error']}"])

    lines.extend(["", "Continue anyway on CPU?"])
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Train tab
# ----------------------------------------------------------------------------
class TrainTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="Panel.TFrame")
        self.app = app
        self.proc = None
        self.reader_thread = None
        self.event_queue = queue.Queue()
        self.training_active = False
        self.start_time = None
        self.user_stopped = False
        self.stop_signal_file = None
        self.last_saved_model_dir = None
        self.preview_photo_ref = None
        self.dataset_photo_ref = None
        self.conveyor_state = None
        self.conveyor_prefetch_thread = None

        self._build_ui()
        self.after(100, self._poll_queue)

    # -- UI construction -----------------------------------------------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        header = ttk.Label(self, text="Training settings", style="Heading.TLabel")
        header.grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(16, 8))

        settings = ttk.Frame(self, style="Panel.TFrame")
        settings.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16)
        for c in range(3):
            settings.columnconfigure(c, weight=1, uniform="settings")

        self.epochs = LabeledEntry(
            settings, "Epochs", default=100,
            tooltip="How many full passes over your dataset. More epochs = better results, but longer training.",
        )
        self.resolution = LabeledEntry(
            settings, "Resolution", default=128,
            tooltip="Image size in pixels (e.g. 128 = 128x128). Higher uses more VRAM and is slower per step.",
        )
        self.batch_size = LabeledEntry(
            settings, "Batch size", default=28,
            tooltip="Images processed per step. 16 is the practical RTX 3060 12GB sweet spot at 128px if it fits. "
                    "Odd/non-power-of-two values like 20, 28, or 30 are allowed too, but VRAM decides if they survive.",
        )
        self.learning_rate = LabeledEntry(
            settings, "Learning rate", default="2e-4",
            tooltip="How fast the model updates. 2e-4 is a faster generalization preset; lower to 1e-4 if loss becomes unstable.",
        )
        self.mixed_precision = LabeledCombo(
            settings, "Mixed precision", values=["fp16", "bf16", "no"], default="fp16",
            tooltip="fp16 speeds up training on most NVIDIA GPUs with little quality loss. "
                    "Use 'no' only if you hit numerical issues.",
        )
        self.preview_every = LabeledEntry(
            settings, "Preview every N epochs", default=5,
            tooltip="How often to generate a preview image and save a checkpoint during "
                    "training. Generating a preview re-runs the model's sampling process, "
                    "which costs real time on top of training itself — previewing every "
                    "epoch on a short run can roughly double total training time. "
                    "Every 5-10 epochs gives plenty of visibility without the overhead.",
        )
        self.training_intensity = LabeledScale(
            settings, "Training intensity", from_=10, to=100, default=75, width=180,
            tooltip="Soft GPU throttle for background or overnight training. "
                    "10% adds big cooldown pauses for gaming/multitasking, 50% is the quiet overnight/background zone, "
                    "75% is serious training with small breathers, and 100% lets DDPM fully focus on training.",
        )
        self.training_intensity.var.trace_add("write", lambda *_: self._update_intensity_hint())

        self.dataloader_workers = LabeledCombo(
            settings, "DataLoader workers", values=["Auto", "4", "6", "8", "2", "0"], default="6", width=10,
            tooltip="CPU image-loading workers. For your i5-10400, test 4, 6, and 8. 6 is the new default sweet spot."
        )
        self.gradient_accumulation = LabeledEntry(
            settings, "Grad accumulation", default=1, width=8,
            tooltip="Simulates a larger effective batch without extra VRAM. It can stabilize training, but usually does not make wall-clock time faster."
        )
        self.preview_steps = LabeledEntry(
            settings, "Preview steps", default=50, width=8,
            tooltip="Denoising steps used only for training preview images. Lower means previews interrupt training less; final model quality is unaffected."
        )
        self.preview_sampler = LabeledCombo(
            settings, "Preview sampler", values=["DDIM", "DDPM"], default="DDIM", width=10,
            tooltip="Sampler used only for in-training preview images. DDIM is usually faster; DDPM is the classic baseline."
        )
        self.pin_memory_var = tk.BooleanVar(value=True)

        self.epochs.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        self.resolution.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        self.batch_size.grid(row=0, column=2, sticky="ew", padx=6, pady=6)
        self.learning_rate.grid(row=1, column=0, sticky="ew", padx=6, pady=6)
        self.mixed_precision.grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        self.preview_every.grid(row=1, column=2, sticky="ew", padx=6, pady=6)
        self.training_intensity.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(8, 2))
        self.dataloader_workers.grid(row=3, column=0, sticky="ew", padx=6, pady=6)
        self.gradient_accumulation.grid(row=3, column=1, sticky="ew", padx=6, pady=6)
        self.preview_steps.grid(row=3, column=2, sticky="ew", padx=6, pady=6)
        self.preview_sampler.grid(row=4, column=0, sticky="ew", padx=6, pady=6)
        self.intensity_hint = ttk.Label(settings, text="", style="Meta.TLabel")
        self.intensity_hint.grid(row=5, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 6))
        self._update_intensity_hint()

        # Dataset folder picker
        data_row = ttk.Frame(settings, style="Panel.TFrame")
        data_row.grid(row=6, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
        ttk.Label(data_row, text="Dataset folder", style="Field.TLabel").pack(anchor="w")
        picker_row = ttk.Frame(data_row, style="Panel.TFrame")
        picker_row.pack(fill="x", pady=(2, 0))
        self.data_dir_var = tk.StringVar(value=str(DEFAULT_DATA_DIR))
        data_entry = ttk.Entry(picker_row, textvariable=self.data_dir_var, style="Field.TEntry")
        data_entry.pack(side="left", fill="x", expand=True)
        browse_btn = ttk.Button(picker_row, text="Browse...", command=self._browse_data_dir, style="Secondary.TButton")
        browse_btn.pack(side="left", padx=(6, 0))
        Tooltip(data_entry, "Folder containing your training images. The model name can auto-fill from this folder name.")

        conveyor_row = ttk.Frame(settings, style="Panel.TFrame")
        conveyor_row.grid(row=7, column=0, columnspan=3, sticky="ew", padx=6, pady=(2, 6))
        conveyor_row.columnconfigure(1, weight=1)
        self.dataset_conveyor_var = tk.BooleanVar(value=False)
        conveyor_check = ttk.Checkbutton(
            conveyor_row, text="Dataset Conveyor Mode",
            variable=self.dataset_conveyor_var, style="Dark.TCheckbutton",
        )
        conveyor_check.grid(row=0, column=0, sticky="w")
        Tooltip(
            conveyor_check,
            "For huge datasets. The app splits the image paths into chunk folders, trains one chunk at a time, "
            "and preloads the next chunk so train.py does not scan/load the entire dataset at once.",
        )

        conveyor_controls = ttk.Frame(conveyor_row, style="Panel.TFrame")
        conveyor_controls.grid(row=0, column=1, sticky="ew", padx=(12, 0))
        self.conveyor_chunks = LabeledEntry(
            conveyor_controls, "Chunks", default=50, width=8,
            tooltip="How many chunks to split the dataset into. 203,000 / 50 = about 4,060 images per chunk.",
        )
        self.conveyor_prefetch_var = tk.BooleanVar(value=True)
        self.conveyor_chunks.pack(side="left", padx=(0, 10))
        prefetch_check = ttk.Checkbutton(
            conveyor_controls, text="Prefetch next chunk",
            variable=self.conveyor_prefetch_var, style="Dark.TCheckbutton",
        )
        prefetch_check.pack(side="left", pady=(18, 0))
        Tooltip(prefetch_check, "Keeps the next chunk ready while the current chunk trains. Uses disk links/copies, not full image RAM loading.")

        name_row = ttk.Frame(settings, style="Panel.TFrame")
        name_row.grid(row=8, column=0, columnspan=3, sticky="ew", padx=6, pady=(2, 6))
        name_row.columnconfigure(1, weight=1)
        ttk.Label(name_row, text="Model name", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.model_name_var = tk.StringVar(value=self._suggest_model_name(DEFAULT_DATA_DIR))
        name_entry = ttk.Entry(name_row, textvariable=self.model_name_var, style="Field.TEntry")
        name_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(name_row, text="Use Dataset Name", command=self._use_dataset_name, style="Secondary.TButton").grid(row=0, column=2, padx=(6, 0))
        Tooltip(name_entry, "Friendly name saved inside model_info.json. Used by the library and per-model Generations folders.")

        # Model loading / output saving
        model_row = ttk.Frame(settings, style="Panel.TFrame")
        model_row.grid(row=9, column=0, columnspan=3, sticky="ew", padx=6, pady=(8, 2))
        model_row.columnconfigure(1, weight=1)
        ttk.Label(model_row, text="Start from trained model", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.base_model_var = tk.StringVar(value="")
        base_entry = ttk.Entry(model_row, textvariable=self.base_model_var, style="Field.TEntry")
        base_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(model_row, text="Load Model...", command=self._browse_base_model, style="Secondary.TButton").grid(row=0, column=2, padx=(6, 0))
        ttk.Button(model_row, text="Clear", command=lambda: self.base_model_var.set(""), style="Secondary.TButton").grid(row=0, column=3, padx=(6, 0))
        Tooltip(base_entry, "Optional: pick an older trained DDPMPipeline folder to continue/fine-tune from. Leave blank to train from scratch.")

        out_row = ttk.Frame(settings, style="Panel.TFrame")
        out_row.grid(row=10, column=0, columnspan=3, sticky="ew", padx=6, pady=(2, 6))
        out_row.columnconfigure(1, weight=1)
        ttk.Label(out_row, text="Save training to", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.output_dir_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        out_entry = ttk.Entry(out_row, textvariable=self.output_dir_var, style="Field.TEntry")
        out_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(out_row, text="Browse...", command=self._browse_output_dir, style="Secondary.TButton").grid(row=0, column=2, padx=(6, 0))
        Tooltip(out_entry, "This exact folder becomes the saved model folder. Use different folders for different experiments.")

        # Advanced toggle (everything else keeps sane defaults under the hood)
        adv_row = ttk.Frame(self, style="Panel.TFrame")
        adv_row.grid(row=2, column=0, columnspan=2, sticky="w", padx=16, pady=(4, 12))
        self.use_ema_var = tk.BooleanVar(value=False)
        ema_check = ttk.Checkbutton(
            adv_row, text="Use EMA (smoother results, slightly slower)",
            variable=self.use_ema_var, style="Dark.TCheckbutton",
        )
        ema_check.pack(side="left")

        self.storage_saver_var = tk.BooleanVar(value=True)
        storage_check = ttk.Checkbutton(
            adv_row, text="Storage Saver (small final model)",
            variable=self.storage_saver_var, style="Dark.TCheckbutton",
        )
        storage_check.pack(side="left", padx=(16, 0))
        Tooltip(
            storage_check,
            "Recommended. Saves only the Generate-ready DDPM pipeline and removes heavy "
            "checkpoint/log training artifacts that can turn one model into 100+ GB.",
        )

        pin_check = ttk.Checkbutton(
            adv_row, text="Pin Memory (faster RAM → GPU transfer)",
            variable=self.pin_memory_var, style="Dark.TCheckbutton",
        )
        pin_check.pack(side="left", padx=(16, 0))
        Tooltip(
            pin_check,
            "Recommended for CUDA training. Locks staging RAM so batches move to VRAM faster. "
            "Turn off only if it causes system stutter."
        )

        # Start/stop controls
        control_row = ttk.Frame(self, style="Panel.TFrame")
        control_row.grid(row=3, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 8))
        self.start_btn = ttk.Button(
            control_row, text="Start Training", command=self._on_start, style="Accent.TButton"
        )
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(
            control_row, text="Stop", command=self._on_stop, style="Secondary.TButton", state=tk.DISABLED
        )
        self.stop_btn.pack(side="left", padx=(8, 0))
        self.save_btn = ttk.Button(
            control_row, text="Save Model As...", command=self._save_current_model_as,
            style="Secondary.TButton", state=tk.DISABLED
        )
        self.save_btn.pack(side="left", padx=(8, 0))

        # Progress panel
        progress_panel = ttk.Frame(self, style="Card.TFrame")
        progress_panel.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=16, pady=(8, 16))
        self.rowconfigure(4, weight=1)
        progress_panel.columnconfigure(0, weight=1)
        progress_panel.columnconfigure(1, weight=0)

        self.status_label = ttk.Label(progress_panel, text="Idle. Configure settings and click Start Training.",
                                       style="Status.TLabel")
        self.status_label.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))

        self.progress_bar = ttk.Progressbar(progress_panel, orient="horizontal", mode="determinate")
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 4))

        meta_row = ttk.Frame(progress_panel, style="Card.TFrame")
        meta_row.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 4))
        self.epoch_label = ttk.Label(meta_row, text="Epoch: -- / --", style="Meta.TLabel")
        self.epoch_label.pack(side="left")
        self.eta_label = ttk.Label(meta_row, text="", style="Meta.TLabel")
        self.eta_label.pack(side="right")

        loss_row = ttk.Frame(progress_panel, style="Card.TFrame")
        loss_row.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.loss_label = ttk.Label(loss_row, text="", style="Meta.TLabel")
        self.loss_label.pack(side="left")

        self.log_text = tk.Text(
            progress_panel, height=8, bg=BG_FIELD, fg=FG_DIM, insertbackground=FG,
            relief="flat", font=("Consolas", 9), wrap="word",
        )
        self.log_text.grid(row=4, column=0, sticky="nsew", padx=14, pady=(0, 14))
        progress_panel.rowconfigure(4, weight=1)

        preview_box = ttk.Frame(progress_panel, style="Card.TFrame")
        preview_box.grid(row=0, column=1, rowspan=5, sticky="ns", padx=(0, 14), pady=14)
        ttk.Label(preview_box, text="Training Preview", style="Meta.TLabel").pack(anchor="w", pady=(0, 6))
        self.preview_label = tk.Label(
            preview_box, text="Dataset examples and generated previews will appear here",
            bg=BG_FIELD, fg=FG_DIM, width=28, height=14, wraplength=180, justify="center",
            font=(FONT_FAMILY, 9),
        )
        self.preview_label.pack(fill="both", expand=True)
        self.log_text.configure(state=tk.DISABLED)

    def _get_intensity_description(self, intensity=None):
        intensity = int(intensity if intensity is not None else self.training_intensity.get())
        if intensity <= 20:
            return "10% Power — heavy cooldown pauses for gaming or extreme multitasking."
        if intensity <= 60:
            return "50% Balanced — good for overnight/background training without the fan goblin screaming."
        if intensity <= 85:
            return "75% Serious — trains hard, with small breathers so the PC stays usable."
        return "100% Full Send — DDPM gets the GPU and your fans get the microphone."

    def _update_intensity_hint(self):
        if hasattr(self, "intensity_hint"):
            self.intensity_hint.config(text=self._get_intensity_description())

    def _worker_count_for_intensity(self, intensity):
        intensity = int(intensity)
        if intensity <= 20:
            return 1
        if intensity <= 60:
            return 2
        if intensity <= 85:
            return 4
        return 6

    def _suggest_model_name(self, data_dir):
        name = Path(data_dir).expanduser().name or "DDPM Model"
        if name.lower() == "train" and Path(data_dir).parent.name:
            name = Path(data_dir).parent.name
        return sanitize_filename_component(name, fallback="DDPM Model")

    def _use_dataset_name(self):
        self.model_name_var.set(self._suggest_model_name(self.data_dir_var.get()))

    def _browse_data_dir(self):
        old_suggestion = self._suggest_model_name(self.data_dir_var.get())
        chosen = filedialog.askdirectory(initialdir=self.data_dir_var.get() or str(ROOT_DIR))
        if chosen:
            self.data_dir_var.set(chosen)
            current_name = self.model_name_var.get().strip()
            if not current_name or current_name == old_suggestion:
                self.model_name_var.set(self._suggest_model_name(chosen))

    def _browse_base_model(self):
        chosen = filedialog.askdirectory(initialdir=str(DEFAULT_MODELS_ROOT if DEFAULT_MODELS_ROOT.exists() else ROOT_DIR))
        if not chosen:
            return
        path = Path(chosen)
        resolved = find_loadable_model_inside(path)
        if resolved is None:
            messagebox.showerror("Not a loadable model", describe_model_problem(path))
            return
        if resolved != path:
            messagebox.showinfo(
                "Model found inside folder",
                f"I found the loadable model here instead:\n{resolved}"
            )
        self.base_model_var.set(str(resolved))
        self._try_apply_model_resolution(resolved)

    def _browse_output_dir(self):
        chosen = filedialog.askdirectory(initialdir=str(DEFAULT_MODELS_ROOT if DEFAULT_MODELS_ROOT.exists() else ROOT_DIR))
        if chosen:
            self.output_dir_var.set(chosen)

    def _try_apply_model_resolution(self, model_dir: Path):
        # Small convenience: if the selected model has a saved UNet config, copy its sample_size
        # into the Resolution field so continuing training does not accidentally mismatch sizes.
        config_path = model_dir / "unet" / "config.json"
        try:
            data = json.loads(config_path.read_text())
            sample_size = data.get("sample_size")
            if sample_size:
                self.resolution.var.set(str(sample_size))
        except Exception:
            pass

    def _save_current_model_as(self):
        src = Path(self.last_saved_model_dir or self.output_dir_var.get())
        if not is_loadable_model_dir(src):
            messagebox.showerror("Nothing to save yet", "No complete trained model has been saved yet.")
            return
        chosen = filedialog.askdirectory(initialdir=str(DEFAULT_MODELS_ROOT if DEFAULT_MODELS_ROOT.exists() else ROOT_DIR))
        if not chosen:
            return
        dst = Path(chosen)
        if dst == src:
            messagebox.showinfo("Already saved", "That is already the current model folder.")
            return
        try:
            copy_model_folder(src, dst)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        self.last_saved_model_dir = str(dst)
        self.app.notify_training_complete()
        messagebox.showinfo("Model saved", f"Saved model to:\n{dst}")

    def _show_preview_image(self, path, caption=None):
        try:
            from PIL import Image, ImageTk
            img = Image.open(path).convert("RGB")
            img = scale_for_display(img, target_size=210)
            photo = ImageTk.PhotoImage(img)
            self.preview_photo_ref = photo
            self.preview_label.config(image=photo, text=caption or "")
        except Exception as e:
            self.preview_label.config(image="", text=f"Preview failed:\n{e}")

    def _show_dataset_examples(self, paths):
        if not paths:
            return
        # Show one example quickly so the training screen feels alive before first generated preview.
        self._show_preview_image(random.choice(paths), caption="Dataset example")

    # -- Validation -------------------------------------------------------
    def _validate_settings(self):
        errors = []
        try:
            epochs = int(self.epochs.get())
            if epochs <= 0:
                errors.append("Epochs must be a positive whole number.")
        except ValueError:
            errors.append("Epochs must be a whole number (e.g. 100).")
            epochs = None

        try:
            resolution = int(self.resolution.get())
            if resolution <= 0 or resolution % 8 != 0:
                errors.append("Resolution should be a positive multiple of 8 (e.g. 64, 128, 256).")
        except ValueError:
            errors.append("Resolution must be a whole number (e.g. 128).")
            resolution = None

        try:
            batch_size = int(self.batch_size.get())
            if batch_size <= 0:
                errors.append("Batch size must be a positive whole number.")
        except ValueError:
            errors.append("Batch size must be a whole number (e.g. 4).")
            batch_size = None

        try:
            lr = float(self.learning_rate.get())
            if lr <= 0:
                errors.append("Learning rate must be a positive number.")
        except ValueError:
            errors.append("Learning rate must be a number (e.g. 1e-4 or 0.0001).")
            lr = None

        try:
            preview_every = int(self.preview_every.get())
            if preview_every <= 0:
                errors.append("Preview every N epochs must be a positive whole number.")
        except ValueError:
            errors.append("Preview every N epochs must be a whole number (e.g. 5).")
            preview_every = None

        conveyor_enabled = bool(getattr(self, "dataset_conveyor_var", tk.BooleanVar(value=False)).get())
        try:
            conveyor_chunks = int(self.conveyor_chunks.get()) if conveyor_enabled else 1
            if conveyor_enabled and conveyor_chunks <= 0:
                errors.append("Dataset Conveyor chunks must be a positive whole number.")
        except ValueError:
            errors.append("Dataset Conveyor chunks must be a whole number, like 50.")
            conveyor_chunks = 1

        intensity = int(self.training_intensity.get())
        intensity = max(10, min(100, intensity))

        workers_raw = self.dataloader_workers.get().strip()
        if workers_raw.lower() == "auto" or workers_raw == "":
            dataloader_workers = None
        else:
            try:
                dataloader_workers = int(workers_raw)
                if dataloader_workers < 0:
                    errors.append("DataLoader workers must be zero or a positive whole number.")
            except ValueError:
                errors.append("DataLoader workers must be 'Auto' or a whole number (e.g. 6).")
                dataloader_workers = None

        try:
            gradient_accumulation = int(self.gradient_accumulation.get())
            if gradient_accumulation <= 0:
                errors.append("Grad accumulation must be a positive whole number.")
        except ValueError:
            errors.append("Grad accumulation must be a whole number (e.g. 1).")
            gradient_accumulation = 1

        try:
            preview_steps = int(self.preview_steps.get())
            if preview_steps <= 0:
                errors.append("Preview steps must be a positive whole number.")
        except ValueError:
            errors.append("Preview steps must be a whole number (e.g. 50).")
            preview_steps = 50

        data_dir = Path(self.data_dir_var.get())
        if not data_dir.exists():
            errors.append(f"Dataset folder does not exist: {data_dir}")

        raw_model_name = self.model_name_var.get().strip() or self._suggest_model_name(data_dir)
        model_name = sanitize_filename_component(raw_model_name, fallback="DDPM Model")
        if model_name != raw_model_name:
            self.model_name_var.set(model_name)

        output_dir = Path(self.output_dir_var.get()).expanduser()
        if not output_dir:
            errors.append("Pick an output folder to save the trained model.")
        base_model = self.base_model_var.get().strip()
        should_auto_pick_folder = output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()
        try:
            under_default_output = output_dir.parent.resolve() == DEFAULT_MODELS_ROOT.resolve()
        except OSError:
            under_default_output = False
        if under_default_output and output_dir.exists() and any(output_dir.iterdir()) and not base_model:
            # Collision handling: Test -> Test 1 -> Test 2, etc.
            should_auto_pick_folder = True
        if should_auto_pick_folder:
            # Default output/model is too easy to overwrite. When the user leaves it
            # at default, or a same-name model already exists, create output/<Model Name>,
            # output/<Model Name 1>, etc.
            output_dir = unique_child_dir(DEFAULT_MODELS_ROOT, model_name)
            self.output_dir_var.set(str(output_dir))

        if base_model:
            resolved_base = find_loadable_model_inside(Path(base_model))
            if resolved_base is None:
                errors.append("Start-from model must be a complete trained model folder containing model_index.json.")
            else:
                base_model = str(resolved_base)
                self.base_model_var.set(base_model)

        if errors:
            return None, errors
        return {
            "epochs": epochs,
            "resolution": resolution,
            "batch_size": batch_size,
            "learning_rate": lr,
            "mixed_precision": self.mixed_precision.get(),
            "preview_every": preview_every,
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "model_name": model_name,
            "base_model": base_model,
            "use_ema": self.use_ema_var.get(),
            "storage_saver": self.storage_saver_var.get(),
            "training_intensity": intensity,
            "dataloader_workers": dataloader_workers if dataloader_workers is not None else self._worker_count_for_intensity(intensity),
            "gradient_accumulation_steps": gradient_accumulation,
            "preview_steps": preview_steps,
            "preview_sampler": self.preview_sampler.get(),
            "pin_memory": bool(self.pin_memory_var.get()),
            "dataset_conveyor": conveyor_enabled,
            "conveyor_chunks": conveyor_chunks,
            "conveyor_prefetch": bool(getattr(self, "conveyor_prefetch_var", tk.BooleanVar(value=True)).get()),
        }, []

    # -- Process control --------------------------------------------------
    def _on_start(self):
        if self.training_active:
            return
        settings, errors = self._validate_settings()
        if errors:
            messagebox.showerror("Check your settings", "\n".join(errors))
            return

        output_dir = Path(settings["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.stop_signal_file = output_dir / "gui_stop_training.flag"
        try:
            self.stop_signal_file.unlink()
        except FileNotFoundError:
            pass

        self.training_active = True
        self.start_time = time.time()
        self.user_stopped = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_bar.config(value=0, maximum=100)
        self.last_saved_model_dir = str(output_dir)
        self.save_btn.config(state=tk.DISABLED)
        self.preview_label.config(image="", text="Scanning dataset...")
        self._clear_log()
        self.app.set_training_active(True)

        if settings.get("dataset_conveyor"):
            try:
                self._start_conveyor_training(settings, output_dir)
            except Exception as exc:
                self._append_log(f"Dataset Conveyor failed to start: {exc}")
                messagebox.showerror("Dataset Conveyor failed", str(exc))
                self._finish_training(success=False)
            return

        cmd = self._build_training_command(
            settings=settings,
            train_data_dir=settings["data_dir"],
            output_dir=output_dir,
            num_epochs=settings["epochs"],
            base_model=settings.get("base_model") or "",
        )
        self.status_label.config(text=f"Starting... {self._get_intensity_description(settings['training_intensity'])}")
        self._launch_training_command(cmd)

    def _build_training_command(self, settings, train_data_dir, output_dir, num_epochs, base_model=""):
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--train_data_dir", str(train_data_dir),
            "--output_dir", str(output_dir),
            "--model_name", settings["model_name"],
            "--resolution", str(settings["resolution"]),
            "--train_batch_size", str(settings["batch_size"]),
            "--num_epochs", str(num_epochs),
            "--learning_rate", str(settings["learning_rate"]),
            "--mixed_precision", settings["mixed_precision"],
            "--save_images_epochs", str(settings.get("save_images_epochs", settings["preview_every"])),
            "--save_model_epochs", str(settings.get("save_model_epochs", settings["preview_every"])),
            "--training_intensity", str(settings["training_intensity"]),
            "--dataloader_num_workers", str(settings["dataloader_workers"]),
            "--gradient_accumulation_steps", str(settings.get("gradient_accumulation_steps", 1)),
            "--preview_num_inference_steps", str(settings.get("preview_steps", 50)),
            "--preview_sampler", str(settings.get("preview_sampler", "DDIM")),
            "--pin_memory", "true" if settings.get("pin_memory", True) else "false",
            "--stop_signal_file", str(self.stop_signal_file),
            "--gui_progress",
        ]
        if settings["storage_saver"]:
            cmd.extend([
                "--storage_saver", "true",
                "--checkpointing_steps", "0",
                "--checkpoints_total_limit", "0",
            ])
        else:
            cmd.extend([
                "--storage_saver", "false",
                "--checkpointing_steps", "1000",
                "--checkpoints_total_limit", "2",
            ])
        if base_model:
            cmd.extend(["--pretrained_model_path", str(base_model)])
        if settings["use_ema"]:
            cmd.append("--use_ema")
        return cmd

    def _launch_training_command(self, cmd):
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(ROOT_DIR),
            )
        except OSError as e:
            messagebox.showerror("Couldn't start training", str(e))
            self._finish_training(success=False)
            return False

        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()
        return True

    def _scan_training_images(self, data_dir: Path):
        paths = []
        for file_path in Path(data_dir).expanduser().rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(file_path)
        paths.sort()
        return paths

    def _split_paths_into_chunks(self, paths, chunk_count):
        if not paths:
            return []
        chunk_count = max(1, min(int(chunk_count), len(paths)))
        chunk_size = max(1, (len(paths) + chunk_count - 1) // chunk_count)
        return [paths[i:i + chunk_size] for i in range(0, len(paths), chunk_size)]

    def _link_or_copy_image(self, src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return
        try:
            os.link(src, dst)
        except Exception:
            try:
                os.symlink(src, dst)
            except Exception:
                shutil.copy2(src, dst)

    def _prepare_conveyor_chunk(self, chunk_index):
        state = self.conveyor_state
        if not state:
            return None
        prepared = state.setdefault("prepared_chunks", {})
        if chunk_index in prepared:
            return prepared[chunk_index]
        chunks = state["chunks"]
        if chunk_index < 0 or chunk_index >= len(chunks):
            return None

        chunk_dir = state["work_dir"] / f"chunk_{chunk_index + 1:04d}"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir, ignore_errors=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        for local_index, src in enumerate(chunks[chunk_index]):
            # Prefix with an index to avoid filename collisions from nested folders.
            safe_name = f"{local_index:06d}_{sanitize_filename_component(src.stem, 'image')}{src.suffix.lower()}"
            self._link_or_copy_image(src, chunk_dir / safe_name)
        prepared[chunk_index] = chunk_dir
        return chunk_dir

    def _prefetch_conveyor_chunk(self, chunk_index):
        if not self.conveyor_state or not self.conveyor_state.get("prefetch"):
            return
        if chunk_index >= len(self.conveyor_state.get("chunks", [])):
            return
        if self.conveyor_prefetch_thread and self.conveyor_prefetch_thread.is_alive():
            return
        def work():
            try:
                self._prepare_conveyor_chunk(chunk_index)
            except Exception as exc:
                self.event_queue.put(("log", f"Dataset Conveyor prefetch failed for chunk {chunk_index + 1}: {exc}"))
        self.conveyor_prefetch_thread = threading.Thread(target=work, daemon=True)
        self.conveyor_prefetch_thread.start()

    def _cleanup_old_conveyor_chunks(self):
        state = self.conveyor_state
        if not state:
            return
        current = state.get("chunk_index", 0)
        prepared = state.setdefault("prepared_chunks", {})
        for idx in list(prepared.keys()):
            if idx < current - 1:
                shutil.rmtree(prepared[idx], ignore_errors=True)
                prepared.pop(idx, None)

    def _start_conveyor_training(self, settings, output_dir: Path):
        image_paths = self._scan_training_images(Path(settings["data_dir"]))
        if not image_paths:
            raise RuntimeError("No usable images were found for Dataset Conveyor Mode.")
        random.shuffle(image_paths)
        chunks = self._split_paths_into_chunks(image_paths, settings["conveyor_chunks"])
        work_dir = output_dir / "_dataset_conveyor_cache"
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        self.conveyor_state = {
            "settings": settings,
            "output_dir": output_dir,
            "work_dir": work_dir,
            "chunks": chunks,
            "pass_index": 0,
            "chunk_index": 0,
            "total_passes": int(settings["epochs"]),
            "prefetch": bool(settings.get("conveyor_prefetch", True)),
            "prepared_chunks": {},
            "base_model": settings.get("base_model") or "",
        }
        chunk_size = len(chunks[0]) if chunks else 0
        self._append_log(
            f"Dataset Conveyor enabled: {len(image_paths)} image(s) split into {len(chunks)} chunk(s), "
            f"about {chunk_size} image(s) per chunk. Epochs now mean full conveyor passes."
        )
        self._append_log("Preparing chunk 1 before training starts...")
        self._prepare_conveyor_chunk(0)
        self._prefetch_conveyor_chunk(1)
        self._run_current_conveyor_chunk()

    def _run_current_conveyor_chunk(self):
        state = self.conveyor_state
        if not state:
            return False
        settings = dict(state["settings"])
        # Conveyor mode needs the model saved after every chunk so the next
        # subprocess can continue from the newly trained output. Keep preview
        # frequency independent so it does not generate a preview after every
        # small chunk unless the user asked for that.
        settings["save_model_epochs"] = 1
        settings["save_images_epochs"] = state["settings"].get("preview_every", 1)
        output_dir = state["output_dir"]
        chunk_index = state["chunk_index"]
        pass_index = state["pass_index"]
        total_chunks = len(state["chunks"])
        total_passes = state["total_passes"]

        # Shuffle the chunk order after the first pass so the model does not see
        # the same conveyor order forever.
        if chunk_index == 0 and pass_index > 0:
            random.shuffle(state["chunks"])
            state["prepared_chunks"].clear()

        chunk_dir = self._prepare_conveyor_chunk(chunk_index)
        self._prefetch_conveyor_chunk(chunk_index + 1)
        self._cleanup_old_conveyor_chunks()
        base_model = state["base_model"] if (pass_index == 0 and chunk_index == 0) else str(output_dir)
        cmd = self._build_training_command(
            settings=settings,
            train_data_dir=chunk_dir,
            output_dir=output_dir,
            num_epochs=1,
            base_model=base_model,
        )
        self.status_label.config(
            text=f"Dataset Conveyor: pass {pass_index + 1}/{total_passes}, chunk {chunk_index + 1}/{total_chunks}..."
        )
        self._append_log(f"Training conveyor chunk {chunk_index + 1}/{total_chunks} for pass {pass_index + 1}/{total_passes}.")
        return self._launch_training_command(cmd)

    def _advance_conveyor_or_finish(self):
        state = self.conveyor_state
        if not state:
            self.status_label.config(text="Training complete!")
            self._finish_training(success=True)
            return

        if self.user_stopped:
            self.status_label.config(text="Training stopped.")
            self._finish_training(success=False)
            return

        state["chunk_index"] += 1
        if state["chunk_index"] >= len(state["chunks"]):
            state["chunk_index"] = 0
            state["pass_index"] += 1
            self._append_log(f"Dataset Conveyor pass {state['pass_index']}/{state['total_passes']} finished.")

        if state["pass_index"] >= state["total_passes"]:
            self.status_label.config(text="Dataset Conveyor training complete!")
            self.progress_bar.config(value=100)
            self.eta_label.config(text="Done")
            self._finish_training(success=True)
            return

        self._run_current_conveyor_chunk()

    def _on_stop(self):
        if self.proc is not None and self.training_active:
            self.status_label.config(text="Stop requested — saving current model after the current step...")
            self.stop_btn.config(state=tk.DISABLED)
            self.user_stopped = True
            try:
                if self.stop_signal_file is not None:
                    Path(self.stop_signal_file).write_text("stop")
            except Exception as e:
                self._append_log(f"Could not write stop signal: {e}")
                try:
                    self.proc.terminate()
                except Exception:
                    pass

    def _read_process_output(self):
        """Runs in a background thread: reads subprocess stdout line by
        line and forwards parsed events (or raw log lines) to the GUI
        thread via a thread-safe queue."""
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            evt = parse_progress_line(line)
            if evt is not None:
                self.event_queue.put(("progress", evt))
            else:
                stripped = line.rstrip()
                if stripped:
                    self.event_queue.put(("log", stripped))
        returncode = self.proc.wait()
        self.event_queue.put(("process_exit", returncode))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "progress":
                    self._handle_progress_event(payload)
                elif kind == "log":
                    self._append_log(payload)
                elif kind == "process_exit":
                    self._handle_process_exit(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _handle_progress_event(self, evt):
        kind = evt.get("event")
        if kind == "dataset_scan":
            usable = evt.get("usable_images", 0)
            total = evt.get("total_files", 0)
            self._append_log(f"Dataset scan: {usable} usable image(s) out of {total} file(s).")
            self._show_dataset_examples(evt.get("example_images") or [])
        elif kind == "hardware_warning":
            message = evt.get("message") or CUDA_CPU_WARNING_MESSAGE
            self._append_log(f"WARNING: {message}")
            self.status_label.config(text="CUDA unavailable — training will run on CPU and may be extremely slow.")
            details = evt.get("details")
            if details:
                self._append_log(details)
        elif kind == "start":
            total_epochs = evt.get("num_epochs", "?")
            self.total_epochs = evt.get("num_epochs")
            intensity = evt.get("training_intensity", "?")
            self.status_label.config(text=f"Training started ({total_epochs} epochs) at {intensity}% intensity...")
            self.epoch_label.config(text=f"Epoch: 0 / {total_epochs}")
            self._append_log(f"Training intensity: {intensity}% | GPU breather sleep ratio: {evt.get('throttle_sleep_ratio', 0):.2f}x step time")
        elif kind == "step":
            total_steps = evt.get("total_steps") or 1
            step = evt.get("global_step", 0)
            pct = min(100, int(100 * step / total_steps))
            self.progress_bar.config(value=pct)
            epoch = evt.get("epoch", 0)
            total_epochs = getattr(self, "total_epochs", "?")
            self.epoch_label.config(text=f"Epoch: {epoch} / {total_epochs}  (step {step}/{total_steps})")
            loss = evt.get("loss")
            if loss is not None:
                self.loss_label.config(text=f"Loss: {loss:.4f}")
            eta = format_eta(evt.get("eta_seconds"))
            self.eta_label.config(text=eta)
            self.status_label.config(text="Training...")
        elif kind == "epoch_end":
            eta = format_eta(evt.get("eta_seconds"))
            self.eta_label.config(text=eta)
            self._append_log(
                f"-- Epoch {evt.get('epoch')} finished in {evt.get('epoch_duration_seconds', 0):.1f}s --"
            )
        elif kind == "preview_image":
            path = evt.get("path")
            if path:
                self._show_preview_image(path, caption=f"Generated preview - epoch {evt.get('epoch')}")
        elif kind == "storage_cleanup":
            removed = evt.get("removed_human", "0 B")
            final_size = evt.get("final_size_human", "?")
            self._append_log(f"Storage Saver removed {removed} of training-only files. Current folder size: {final_size}.")
        elif kind == "model_saved":
            out = evt.get("output_dir")
            if out:
                self.last_saved_model_dir = out
                self.save_btn.config(state=tk.NORMAL)
                size = evt.get("size_human")
                model_name = evt.get("model_name")
                if size:
                    self._append_log(f"Saved lightweight loadable model to: {out}  ({size})")
                else:
                    self._append_log(f"Saved loadable model to: {out}")
                if model_name:
                    self._append_log(f"Model name: {model_name}")
        elif kind == "stopped":
            self.status_label.config(text="Training stopped and saved.")
            self.eta_label.config(text="Stopped")
            self._finish_training(success=True)
        elif kind == "done":
            if self.conveyor_state:
                self.status_label.config(text="Dataset Conveyor chunk finished; preparing next chunk...")
            else:
                self.status_label.config(text="Training complete!", )
                self.progress_bar.config(value=100)
                self.eta_label.config(text="Done")
                self._finish_training(success=True)
        elif kind == "error":
            self._append_log(f"ERROR: {evt.get('message', 'unknown error')}")

    def _handle_process_exit(self, returncode):
        if not self.training_active:
            return  # already handled via the "done" event
        if returncode == 0:
            self._advance_conveyor_or_finish()
        elif self.user_stopped:
            self.status_label.config(text="Training stopped.")
            self._finish_training(success=False)
        else:
            self.status_label.config(text=f"Training stopped with an error (exit code {returncode}). See log below.")
            self._finish_training(success=False)

    def _finish_training(self, success):
        if self.conveyor_state:
            work_dir = self.conveyor_state.get("work_dir")
            if work_dir and success:
                shutil.rmtree(work_dir, ignore_errors=True)
            self.conveyor_state = None
        self.training_active = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.app.set_training_active(False)
        if success:
            self.save_btn.config(state=tk.NORMAL)
            self.app.notify_training_complete()

    def _append_log(self, text):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_log(self):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)


# ----------------------------------------------------------------------------
# Generate tab
# ----------------------------------------------------------------------------
class GenerateTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="Panel.TFrame")
        self.app = app
        self.photo_ref = None  # keep a reference so Tk doesn't garbage-collect it
        self.last_images = []
        self.last_model_path = None
        self._checkpoint_paths = {}
        self._model_infos = {}
        self._build_ui()
        self.refresh_checkpoints()

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Label(self, text="Model Library", style="Heading.TLabel")
        header.grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(16, 8))

        library = ttk.Frame(self, style="Card.TFrame")
        library.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 10))
        library.columnconfigure(0, weight=1)

        self.model_tree = ttk.Treeview(
            library,
            columns=("name", "resolution", "modified", "path"),
            show="headings",
            height=6,
            selectmode="browse",
        )
        self.model_tree.heading("name", text="Model name")
        self.model_tree.heading("resolution", text="Resolution")
        self.model_tree.heading("modified", text="Modified")
        self.model_tree.heading("path", text="Model folder")
        self.model_tree.column("name", width=180, stretch=False)
        self.model_tree.column("resolution", width=90, stretch=False)
        self.model_tree.column("modified", width=135, stretch=False)
        self.model_tree.column("path", width=340, stretch=True)
        self.model_tree.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        self.model_tree.bind("<<TreeviewSelect>>", self._on_model_tree_select)
        self.model_tree.bind("<Double-1>", lambda _e: self._load_selected_tree_model())

        library_buttons = ttk.Frame(library, style="Card.TFrame")
        library_buttons.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        ttk.Button(library_buttons, text="Refresh Library", command=self.refresh_checkpoints, style="Secondary.TButton").pack(side="left")
        ttk.Button(library_buttons, text="Scan/Load Folder...", command=self._browse_model, style="Secondary.TButton").pack(side="left", padx=(8, 0))
        ttk.Button(library_buttons, text="Use Selected Model", command=self._load_selected_tree_model, style="Accent.TButton").pack(side="left", padx=(8, 0))
        ttk.Button(library_buttons, text="Open Folder", command=self._open_selected_model_folder, style="Secondary.TButton").pack(side="left", padx=(8, 0))
        ttk.Button(library_buttons, text="Save Selected As...", command=self._save_selected_model_as, style="Secondary.TButton").pack(side="left", padx=(8, 0))

        main = ttk.Frame(self, style="Panel.TFrame")
        main.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=16)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)

        # The image generation controls can be taller than the visible window,
        # especially on smaller monitors or with Windows display scaling.
        # Put the control panel inside a scrollable canvas so the Save button
        # can no longer sneak below the bottom of the app like cursed UI archaeology.
        controls_outer = ttk.Frame(main, style="Card.TFrame")
        controls_outer.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 12), pady=(0, 16))
        controls_outer.rowconfigure(0, weight=1)
        controls_outer.columnconfigure(0, weight=1)

        self.image_controls_canvas = tk.Canvas(
            controls_outer,
            bg=BG_PANEL,
            highlightthickness=0,
            bd=0,
            width=260,
        )
        self.image_controls_scrollbar = ttk.Scrollbar(
            controls_outer,
            orient="vertical",
            command=self.image_controls_canvas.yview,
        )
        self.image_controls_canvas.configure(yscrollcommand=self.image_controls_scrollbar.set)

        self.image_controls_canvas.grid(row=0, column=0, sticky="ns")
        self.image_controls_scrollbar.grid(row=0, column=1, sticky="ns")

        controls = ttk.Frame(self.image_controls_canvas, style="Card.TFrame")
        self.image_controls_window = self.image_controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw"
        )

        def _sync_image_controls_scrollregion(_event=None):
            self.image_controls_canvas.configure(scrollregion=self.image_controls_canvas.bbox("all"))

        def _sync_image_controls_width(event):
            # Keep the inner ttk.Frame the same width as the canvas viewport.
            self.image_controls_canvas.itemconfigure(self.image_controls_window, width=event.width)

        def _wheel_image_controls(event):
            if getattr(event, "num", None) == 4:
                self.image_controls_canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                self.image_controls_canvas.yview_scroll(1, "units")
            else:
                delta = -1 * int(event.delta / 120) if event.delta else 0
                self.image_controls_canvas.yview_scroll(delta, "units")
            return "break"

        def _bind_image_controls_wheel(_event=None):
            self.image_controls_canvas.bind_all("<MouseWheel>", _wheel_image_controls)
            self.image_controls_canvas.bind_all("<Button-4>", _wheel_image_controls)
            self.image_controls_canvas.bind_all("<Button-5>", _wheel_image_controls)

        def _unbind_image_controls_wheel(_event=None):
            self.image_controls_canvas.unbind_all("<MouseWheel>")
            self.image_controls_canvas.unbind_all("<Button-4>")
            self.image_controls_canvas.unbind_all("<Button-5>")

        controls.bind("<Configure>", _sync_image_controls_scrollregion)
        self.image_controls_canvas.bind("<Configure>", _sync_image_controls_width)
        controls_outer.bind("<Enter>", _bind_image_controls_wheel)
        controls_outer.bind("<Leave>", _unbind_image_controls_wheel)

        ttk.Label(controls, text="Generation Controls", style="Meta.TLabel").pack(anchor="w", padx=12, pady=(12, 8))
        self.selected_model_label = ttk.Label(controls, text="Selected: --", style="Meta.TLabel", wraplength=240, justify="left")
        self.selected_model_label.pack(anchor="w", padx=12, pady=(0, 10))

        self.preset = LabeledCombo(
            controls, "Preset", values=["Balanced", "Dreamy", "Clean", "Chaotic", "Fast Preview"], default="Balanced", width=18,
            tooltip="Presets mainly adjust denoising steps and seed behavior. They don't fake a new training resolution."
        )
        self.preset.pack(fill="x", padx=12, pady=6)
        self.preset.var.trace_add("write", lambda *_: self._apply_preset())

        self.steps = LabeledScale(
            controls, "Inference steps", from_=10, to=350, default=75,
            tooltip="More steps usually means cleaner/more settled images, but slower generation."
        )
        self.steps.pack(fill="x", padx=12, pady=6)

        self.sampler = LabeledCombo(
            controls, "Sampler", values=["DDPM", "DDIM"], default="DDIM", width=10,
            tooltip=(
                "DDPM: classic slower sampler, very stable baseline. "
                "DDIM: faster sampler that usually works well at lower step counts and is great for quick dream previews."
            )
        )
        self.sampler.pack(fill="x", padx=12, pady=6)
        self.sampler.var.trace_add("write", lambda *_: self._update_sampler_hint())
        self.sampler_hint = ttk.Label(controls, text="", style="Meta.TLabel", wraplength=240, justify="left")
        self.sampler_hint.pack(fill="x", padx=12, pady=(0, 4))
        self._update_sampler_hint()

        self.batch_count = LabeledCombo(
            controls, "Images", values=["1", "2", "4", "8"], default="1", width=8,
            tooltip="Generate several dream samples at once. Uses more VRAM while generating."
        )
        self.batch_count.pack(fill="x", padx=12, pady=6)

        self.display_size = LabeledCombo(
            controls, "Preview size", values=["256", "384", "512", "768"], default="512", width=8,
            tooltip="Only changes how big the image appears in the app. The model's real resolution stays the same."
        )
        self.display_size.pack(fill="x", padx=12, pady=6)

        seed_frame = ttk.Frame(controls, style="Card.TFrame")
        seed_frame.pack(fill="x", padx=12, pady=(8, 4))
        ttk.Label(seed_frame, text="Seed (optional)", style="Meta.TLabel").pack(anchor="w")
        self.seed_var = tk.StringVar(value="")
        ttk.Entry(seed_frame, textvariable=self.seed_var, width=18, style="Field.TEntry").pack(anchor="w", fill="x", pady=(2, 0))
        ttk.Button(seed_frame, text="Randomize Seed", command=self._randomize_seed, style="Secondary.TButton").pack(anchor="w", pady=(6, 0))

        out_frame = ttk.Frame(controls, style="Card.TFrame")
        out_frame.pack(fill="x", padx=12, pady=(8, 4))
        ttk.Label(out_frame, text="Save generated images to", style="Meta.TLabel").pack(anchor="w")
        self.image_output_dir_var = tk.StringVar(value=str(DEFAULT_GENERATIONS_DIR))
        ttk.Entry(out_frame, textvariable=self.image_output_dir_var, width=24, style="Field.TEntry").pack(anchor="w", fill="x", pady=(2, 0))
        ttk.Button(out_frame, text="Browse...", command=self._browse_image_output_dir, style="Secondary.TButton").pack(anchor="w", pady=(6, 0))

        self.generate_btn = ttk.Button(controls, text="Generate", command=self._on_generate, style="Accent.TButton")
        self.generate_btn.pack(fill="x", padx=12, pady=(14, 6))
        ttk.Button(controls, text="Save Last Image(s)", command=self._save_last_images, style="Secondary.TButton").pack(fill="x", padx=12, pady=(0, 12))

        self.gen_status_label = ttk.Label(main, text="Ready.", style="Status.TLabel")
        self.gen_status_label.grid(row=0, column=1, sticky="w", pady=(0, 8))

        image_frame = ttk.Frame(main, style="Card.TFrame")
        image_frame.grid(row=1, column=1, sticky="nsew", pady=(0, 16))
        self.image_label = tk.Label(
            image_frame, text="Image will appear here", bg=BG_FIELD, fg=FG_DIM, font=(FONT_FAMILY, 12),
        )
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)

    def refresh_checkpoints(self):
        checkpoints = find_checkpoints(DEFAULT_MODELS_ROOT)
        previous_path = self.get_selected_model_path()
        self._checkpoint_paths = {}
        self._model_infos = {}
        self.model_tree.delete(*self.model_tree.get_children())

        for idx, item in enumerate(checkpoints):
            label, path, _mtime, info = item
            key = f"model_{idx}"
            self._checkpoint_paths[key] = path
            self._model_infos[key] = info
            self.model_tree.insert("", "end", iid=key, values=(info["model_name"], info["resolution"], info["modified"], info["display"]))

        if not checkpoints:
            key = "pretrained_demo"
            self._checkpoint_paths[key] = PRETRAINED_MODEL
            self._model_infos[key] = {"model_name": "Pretrained demo", "display": "Pretrained demo model", "resolution": "128x128", "modified": "online", "path": PRETRAINED_MODEL}
            self.model_tree.insert("", "end", iid=key, values=("Pretrained demo", "128x128", "online", "Pretrained demo model"))

        # Keep previous selection when possible.
        target_key = None
        if previous_path:
            for key, path in self._checkpoint_paths.items():
                if path == previous_path:
                    target_key = key
                    break
        if target_key is None and self.model_tree.get_children():
            target_key = self.model_tree.get_children()[0]
        if target_key:
            self.model_tree.selection_set(target_key)
            self.model_tree.focus(target_key)
            self._on_model_tree_select()

    def get_selected_model_key(self):
        selection = self.model_tree.selection()
        if selection:
            return selection[0]
        focus = self.model_tree.focus()
        return focus or None

    def get_selected_model_path(self):
        key = self.get_selected_model_key()
        if key:
            return self._checkpoint_paths.get(key)
        return None

    def _on_model_tree_select(self, _event=None):
        key = self.get_selected_model_key()
        info = self._model_infos.get(key, {})
        text = f"Selected: {info.get('display', '--')}\nResolution: {info.get('resolution', '?')}"
        self.selected_model_label.config(text=text)

    def _load_selected_tree_model(self):
        path = self.get_selected_model_path()
        if not path:
            messagebox.showerror("No model selected", "Select a model from the library first.")
            return
        self.last_model_path = path
        info = self._model_infos.get(self.get_selected_model_key(), {})
        self.gen_status_label.config(text=f"Ready to generate with: {info.get('model_name', info.get('display', path))}")

    def _browse_model(self):
        chosen = filedialog.askdirectory(initialdir=str(DEFAULT_MODELS_ROOT if DEFAULT_MODELS_ROOT.exists() else ROOT_DIR))
        if not chosen:
            return
        path = Path(chosen)
        resolved = find_loadable_model_inside(path)
        if resolved is None:
            messagebox.showerror("Not a loadable model", describe_model_problem(path))
            return
        if resolved != path:
            messagebox.showinfo("Model found inside folder", f"I found the loadable model here instead:\n{resolved}")

        key = f"loaded_{len(self._checkpoint_paths)}"
        info = get_model_info(resolved)
        self._checkpoint_paths[key] = str(resolved)
        self._model_infos[key] = info
        if key not in self.model_tree.get_children():
            self.model_tree.insert("", 0, iid=key, values=(info["model_name"], info["resolution"], info["modified"], info["display"]))
        self.model_tree.selection_set(key)
        self.model_tree.focus(key)
        self._on_model_tree_select()
        self._load_selected_tree_model()

    def _open_selected_model_folder(self):
        path = self.get_selected_model_path()
        if not path or path == PRETRAINED_MODEL:
            messagebox.showerror("No local folder", "Select a local trained model first.")
            return
        open_folder(path)

    def _save_selected_model_as(self):
        src = self.get_selected_model_path()
        if not src or src == PRETRAINED_MODEL:
            messagebox.showerror("Cannot save this", "Pick a local trained model folder first.")
            return
        src_path = Path(src)
        resolved = find_loadable_model_inside(src_path)
        if resolved is None:
            messagebox.showerror("Cannot save this", describe_model_problem(src_path))
            return
        chosen = filedialog.askdirectory(initialdir=str(DEFAULT_MODELS_ROOT if DEFAULT_MODELS_ROOT.exists() else ROOT_DIR))
        if not chosen:
            return
        dst = Path(chosen)
        try:
            copy_model_folder(resolved, dst)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        self.refresh_checkpoints()
        self.gen_status_label.config(text=f"Saved selected model to: {dst}")

    def _browse_image_output_dir(self):
        chosen = filedialog.askdirectory(initialdir=self.image_output_dir_var.get() or str(DEFAULT_GENERATIONS_DIR))
        if chosen:
            self.image_output_dir_var.set(chosen)

    def _randomize_seed(self):
        self.seed_var.set(str(random.randint(0, 2_147_483_647)))

    def _update_sampler_hint(self):
        sampler = self.sampler.get() if hasattr(self, "sampler") else "DDPM"
        self.sampler_hint.config(text=SAMPLER_DESCRIPTIONS.get(sampler, SAMPLER_DESCRIPTIONS["DDPM"]))

    def _apply_preset(self):
        preset = self.preset.get()
        if preset == "Fast Preview":
            self.steps.var.set(25)
        elif preset == "Clean":
            self.steps.var.set(150)
        elif preset == "Dreamy":
            self.steps.var.set(100)
        elif preset == "Chaotic":
            self.steps.var.set(40)
            self.seed_var.set("")
        else:
            self.steps.var.set(75)
        self.steps.value_label.config(text=str(self.steps.get()))

    def _validated_generation_settings(self):
        model_path = self.last_model_path or self.get_selected_model_path() or PRETRAINED_MODEL
        seed_val = self.seed_var.get().strip()
        try:
            seed = int(seed_val) if seed_val else None
        except ValueError:
            messagebox.showerror("Invalid seed", "Seed must be a whole number, or left blank.")
            return None
        try:
            batch_count = int(self.batch_count.get())
            preview_size = int(self.display_size.get())
            steps = max(1, int(self.steps.get()))
        except ValueError:
            messagebox.showerror("Invalid generation settings", "Generation settings must be whole numbers.")
            return None
        return {
            "model_path": model_path,
            "seed": seed,
            "steps": steps,
            "sampler": self.sampler.get(),
            "batch_count": batch_count,
            "preview_size": preview_size,
        }

    def _on_generate(self):
        settings = self._validated_generation_settings()
        if settings is None:
            return

        self.generate_btn.config(state=tk.DISABLED)
        self.gen_status_label.config(text=f"Generating {settings['batch_count']} image(s) with {settings['sampler']} at {settings['steps']} steps...")

        def work():
            try:
                images = generate_images(
                    settings["model_path"],
                    seed=settings["seed"],
                    num_inference_steps=settings["steps"],
                    batch_size=settings["batch_count"],
                    sampler=settings["sampler"],
                )
                from PIL import Image as PILImage, ImageTk

                # Make a simple preview grid for the app display.
                display_images = [scale_for_display(img, target_size=settings["preview_size"]) for img in images]
                cols = min(4, len(display_images))
                rows = (len(display_images) + cols - 1) // cols
                w = max(img.width for img in display_images)
                h = max(img.height for img in display_images)
                grid = PILImage.new("RGB", (cols * w, rows * h), (30, 34, 43))
                for i, img in enumerate(display_images):
                    grid.paste(img.convert("RGB"), ((i % cols) * w, (i // cols) * h))
                photo = ImageTk.PhotoImage(grid)

                def update_ui():
                    self.last_images = images
                    self.photo_ref = photo
                    self.image_label.config(image=photo, text="")
                    first = images[0]
                    self.gen_status_label.config(
                        text=f"Done. Native model output: {first.width}x{first.height}. Preview shown enlarged."
                    )
                    self.generate_btn.config(state=tk.NORMAL)

                self.after(0, update_ui)
            except Exception as e:
                error_text = str(e)[:500]

                def show_error():
                    friendly = error_text
                    if "model_index.json" in error_text or "Error no file named" in error_text:
                        friendly = (
                            "Could not load that folder as a complete model. "
                            "Try selecting the saved model folder that contains model_index.json, "
                            "not a checkpoint-* resume folder.\n\n"
                            + error_text[:250]
                        )
                    self.gen_status_label.config(text=f"Error: {friendly}")
                    self.generate_btn.config(state=tk.NORMAL)

                self.after(0, show_error)

        threading.Thread(target=work, daemon=True).start()

    def _get_active_model_info_for_generations(self):
        model_path = self.last_model_path or self.get_selected_model_path()
        for key, path in self._checkpoint_paths.items():
            if path == model_path:
                return self._model_infos.get(key, {})
        if model_path and model_path != PRETRAINED_MODEL:
            return get_model_info(Path(model_path))
        return {"model_name": "Pretrained demo"}

    def _save_last_images(self):
        if not self.last_images:
            messagebox.showerror("No images yet", "Generate an image first.")
            return
        root_out_dir = Path(self.image_output_dir_var.get()).expanduser()
        model_info = self._get_active_model_info_for_generations()
        model_name = sanitize_filename_component(model_info.get("model_name") or model_info.get("name") or "Model")
        out_dir = root_out_dir / f"{model_name} Generations"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        saved = []
        for idx, img in enumerate(self.last_images, start=1):
            path = out_dir / f"{model_name}_generation_{stamp}_{idx:02d}.png"
            img.save(path)
            saved.append(path)
        self.gen_status_label.config(text=f"Saved {len(saved)} image(s) to: {out_dir}")


# ----------------------------------------------------------------------------
# Video tab
# ----------------------------------------------------------------------------
class VideoTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="Panel.TFrame")
        self.app = app
        self.video_photo_ref = None
        self.last_video_path = None
        self._model_paths_by_label = {}
        self._build_ui()
        self.refresh_models()

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Label(self, text="Dream Video Generator", style="Heading.TLabel")
        header.grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(16, 8))

        controls = ttk.Frame(self, style="Card.TFrame")
        controls.grid(row=1, column=0, sticky="ns", padx=(16, 10), pady=(0, 16))
        preview = ttk.Frame(self, style="Card.TFrame")
        preview.grid(row=1, column=1, sticky="nsew", padx=(0, 16), pady=(0, 16))
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(2, weight=1)

        ttk.Label(controls, text="Video Controls", style="Meta.TLabel").pack(anchor="w", padx=12, pady=(12, 8))

        model_wrap = ttk.Frame(controls, style="Card.TFrame")
        model_wrap.pack(fill="x", padx=12, pady=6)
        ttk.Label(model_wrap, text="Model", style="Meta.TLabel").pack(anchor="w")
        self.video_model_var = tk.StringVar()
        self.video_model_combo = ttk.Combobox(model_wrap, textvariable=self.video_model_var, state="readonly", width=28)
        self.video_model_combo.pack(fill="x", pady=(2, 4))
        buttons = ttk.Frame(model_wrap, style="Card.TFrame")
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Refresh", command=self.refresh_models, style="Secondary.TButton").pack(side="left")
        ttk.Button(buttons, text="Use Generate Selection", command=self._use_generate_model, style="Secondary.TButton").pack(side="left", padx=(6, 0))

        ref_wrap = ttk.Frame(controls, style="Card.TFrame")
        ref_wrap.pack(fill="x", padx=12, pady=6)
        ttk.Label(ref_wrap, text="Reference video (optional)", style="Meta.TLabel").pack(anchor="w")
        self.reference_video_var = tk.StringVar(value="")
        ttk.Entry(ref_wrap, textvariable=self.reference_video_var, style="Field.TEntry").pack(fill="x", pady=(2, 4))
        ref_btns = ttk.Frame(ref_wrap, style="Card.TFrame")
        ref_btns.pack(fill="x")
        ttk.Button(ref_btns, text="Browse...", command=self._browse_reference_video, style="Secondary.TButton").pack(side="left")
        ttk.Button(ref_btns, text="Clear", command=lambda: self.reference_video_var.set(""), style="Secondary.TButton").pack(side="left", padx=(6, 0))

        length_wrap = ttk.Frame(controls, style="Card.TFrame")
        length_wrap.pack(fill="x", padx=12, pady=6)
        ttk.Label(length_wrap, text="Length / timing", style="Meta.TLabel").pack(anchor="w")
        row = ttk.Frame(length_wrap, style="Card.TFrame")
        row.pack(fill="x", pady=(2, 0))
        self.seconds_var = tk.StringVar(value="4")
        self.fps_var = tk.StringVar(value="24")
        ttk.Label(row, text="Seconds", style="Meta.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=self.seconds_var, width=8, style="Field.TEntry").grid(row=1, column=0, sticky="w", padx=(0, 10))
        ttk.Label(row, text="FPS", style="Meta.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Entry(row, textvariable=self.fps_var, width=8, style="Field.TEntry").grid(row=1, column=1, sticky="w")

        self.video_steps = LabeledScale(
            controls, "Inference steps", from_=10, to=120, default=60,
            tooltip="More denoising steps usually gives cleaner frames, but each frame takes longer."
        )
        self.video_steps.pack(fill="x", padx=12, pady=6)

        self.video_smoothness = LabeledScale(
            controls, "Motion smoothness", from_=1, to=100, default=85,
            tooltip="Higher values keep frames closer together in latent space for a smoother dream walk."
        )
        self.video_smoothness.pack(fill="x", padx=12, pady=6)

        self.reference_influence = LabeledScale(
            controls, "Reference influence", from_=0, to=100, default=70,
            tooltip="How close the DDPM denoising starts to the reference frame. Higher values preserve more structure."
        )
        self.reference_influence.pack(fill="x", padx=12, pady=6)

        self.reference_mode = LabeledCombo(
            controls, "Reference mode",
            values=["Dreamify/Reconstruct", "Reimagine Shape"],
            default="Dreamify/Reconstruct", width=20,
            tooltip="Dreamify/Reconstruct keeps the source frame visible, APVD-style. Reimagine Shape is the older troll-face/object-morph style."
        )
        self.reference_mode.pack(fill="x", padx=12, pady=6)

        self.source_preservation = LabeledScale(
            controls, "Source preservation", from_=0, to=100, default=55,
            tooltip="Only used in Dreamify/Reconstruct mode. Higher keeps more of the original frame; lower lets DDPM dream harder."
        )
        self.source_preservation.pack(fill="x", padx=12, pady=6)

        seed_wrap = ttk.Frame(controls, style="Card.TFrame")
        seed_wrap.pack(fill="x", padx=12, pady=6)
        ttk.Label(seed_wrap, text="Seed (optional)", style="Meta.TLabel").pack(anchor="w")
        self.video_seed_var = tk.StringVar(value="")
        ttk.Entry(seed_wrap, textvariable=self.video_seed_var, style="Field.TEntry").pack(fill="x", pady=(2, 4))
        ttk.Button(seed_wrap, text="Randomize Seed", command=self._randomize_seed, style="Secondary.TButton").pack(anchor="w")

        out_wrap = ttk.Frame(controls, style="Card.TFrame")
        out_wrap.pack(fill="x", padx=12, pady=6)
        ttk.Label(out_wrap, text="Output", style="Meta.TLabel").pack(anchor="w")
        self.video_output_dir_var = tk.StringVar(value=str(ROOT_DIR / "output" / "videos"))
        ttk.Entry(out_wrap, textvariable=self.video_output_dir_var, style="Field.TEntry").pack(fill="x", pady=(2, 4))
        out_buttons = ttk.Frame(out_wrap, style="Card.TFrame")
        out_buttons.pack(fill="x")
        ttk.Button(out_buttons, text="Browse...", command=self._browse_video_output_dir, style="Secondary.TButton").pack(side="left")
        ttk.Label(out_buttons, text="Format", style="Meta.TLabel").pack(side="left", padx=(10, 6))
        self.video_format_var = tk.StringVar(value="mp4")
        ttk.Combobox(out_buttons, textvariable=self.video_format_var, values=["mp4", "gif"], state="readonly", width=6).pack(side="left")

        self.generate_video_btn = ttk.Button(controls, text="Generate Video", command=self._on_generate_video, style="Accent.TButton")
        self.generate_video_btn.pack(fill="x", padx=12, pady=(12, 6))
        ttk.Button(controls, text="Open Output Folder", command=self._open_output_folder, style="Secondary.TButton").pack(fill="x", padx=12, pady=(0, 12))

        self.video_status_label = ttk.Label(preview, text="Ready. Generate a DDPM dream video.", style="Status.TLabel")
        self.video_status_label.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))
        self.video_progress = ttk.Progressbar(preview, orient="horizontal", mode="determinate")
        self.video_progress.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        self.video_preview_label = tk.Label(
            preview, text="Preview frame will appear here",
            bg=BG_FIELD, fg=FG_DIM, font=(FONT_FAMILY, 11),
        )
        self.video_preview_label.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 8))
        self.video_info_label = ttk.Label(
            preview,
            text="Uses latent walk generation. Dreamify/Reconstruct mode keeps the reference frame visible and lets DDPM repaint it instead of replacing it.",
            style="Meta.TLabel", wraplength=520, justify="left",
        )
        self.video_info_label.grid(row=3, column=0, sticky="w", padx=14, pady=(0, 14))

    def refresh_models(self):
        checkpoints = find_checkpoints(DEFAULT_MODELS_ROOT)
        self._model_paths_by_label = {}
        labels = []
        for label, path, _mtime, info in checkpoints:
            combo_label = f"{info['display']} ({info['resolution']})"
            labels.append(combo_label)
            self._model_paths_by_label[combo_label] = path
        if not labels:
            labels = ["Pretrained demo model (128x128)"]
            self._model_paths_by_label[labels[0]] = PRETRAINED_MODEL
        self.video_model_combo.config(values=labels)
        if self.video_model_var.get() not in labels:
            self.video_model_var.set(labels[0])

    def _use_generate_model(self):
        model_path = self.app.get_preferred_model_path()
        if not model_path:
            messagebox.showerror("No model selected", "Select or load a model in the Generate tab first, or pick one from this tab.")
            return
        for label, path in self._model_paths_by_label.items():
            if path == model_path:
                self.video_model_var.set(label)
                self.video_status_label.config(text=f"Using model from Generate tab: {label}")
                return
        resolved = find_loadable_model_inside(Path(model_path)) if model_path != PRETRAINED_MODEL else None
        if resolved is not None:
            label = f"{get_model_info(resolved)['display']} ({get_model_info(resolved)['resolution']})"
            self._model_paths_by_label[label] = str(resolved)
            self.video_model_combo.config(values=list(self._model_paths_by_label.keys()))
            self.video_model_var.set(label)
            self.video_status_label.config(text=f"Using model from Generate tab: {label}")

    def _browse_reference_video(self):
        chosen = filedialog.askopenfilename(
            initialdir=str(ROOT_DIR),
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.webm *.gif"), ("All files", "*.*")],
        )
        if chosen:
            self.reference_video_var.set(chosen)

    def _browse_video_output_dir(self):
        chosen = filedialog.askdirectory(initialdir=self.video_output_dir_var.get() or str(ROOT_DIR))
        if chosen:
            self.video_output_dir_var.set(chosen)

    def _open_output_folder(self):
        out_dir = Path(self.video_output_dir_var.get()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        open_folder(out_dir)

    def _randomize_seed(self):
        self.video_seed_var.set(str(random.randint(0, 2_147_483_647)))

    def _validated_video_settings(self):
        model_label = self.video_model_var.get().strip()
        model_path = self._model_paths_by_label.get(model_label, PRETRAINED_MODEL)

        ref_path = self.reference_video_var.get().strip()
        if ref_path and not Path(ref_path).exists():
            messagebox.showerror("Missing reference video", f"Reference video does not exist:\n{ref_path}")
            return None

        try:
            seconds = float(self.seconds_var.get().strip())
            fps = int(self.fps_var.get().strip())
            steps = int(self.video_steps.get())
            smoothness = int(self.video_smoothness.get())
            ref_influence = int(self.reference_influence.get())
            source_preservation = int(self.source_preservation.get())
            reference_mode = self.reference_mode.get()
        except ValueError:
            messagebox.showerror("Invalid settings", "Seconds, FPS, and sliders must contain valid numbers.")
            return None

        if seconds <= 0:
            messagebox.showerror("Invalid length", "Seconds must be greater than 0.")
            return None
        if fps <= 0:
            messagebox.showerror("Invalid FPS", "FPS must be greater than 0.")
            return None

        total_frames = int(round(seconds * fps))
        if total_frames > 240:
            proceed = messagebox.askyesno(
                "Large render",
                f"This will render {total_frames} frames with DDPM sampling, which may take a while. Continue?"
            )
            if not proceed:
                return None

        seed_val = self.video_seed_var.get().strip()
        try:
            seed = int(seed_val) if seed_val else None
        except ValueError:
            messagebox.showerror("Invalid seed", "Seed must be a whole number, or left blank.")
            return None

        out_dir = Path(self.video_output_dir_var.get()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        extension = self.video_format_var.get().strip().lower() or "mp4"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"dream_video_{stamp}.{extension}"

        return {
            "model_path": model_path,
            "reference_video_path": ref_path or None,
            "seconds": seconds,
            "fps": fps,
            "steps": steps,
            "smoothness": smoothness,
            "reference_influence": ref_influence / 100.0,
            "reference_mode": reference_mode,
            "source_preservation": source_preservation / 100.0,
            "seed": seed,
            "output_path": out_path,
            "total_frames": total_frames,
        }

    def _on_generate_video(self):
        settings = self._validated_video_settings()
        if settings is None:
            return

        self.generate_video_btn.config(state=tk.DISABLED)
        self.video_progress.config(value=0, maximum=max(1, settings["total_frames"]))
        self.video_status_label.config(text=f"Generating {settings['total_frames']} frame(s)...")
        self.video_info_label.config(text="")

        def progress_callback(done, total, frame_image):
            def update():
                self.video_progress.config(value=done, maximum=total)
                try:
                    from PIL import ImageTk
                    preview_img = scale_for_display(frame_image, target_size=512)
                    photo = ImageTk.PhotoImage(preview_img)
                    self.video_photo_ref = photo
                    self.video_preview_label.config(image=photo, text="")
                except Exception:
                    pass
                self.video_status_label.config(text=f"Generating frame {done}/{total}...")
            self.after(0, update)

        def worker():
            try:
                frames = generate_video_frames(
                    settings["model_path"],
                    seconds=settings["seconds"],
                    fps=settings["fps"],
                    seed=settings["seed"],
                    num_inference_steps=settings["steps"],
                    smoothness=settings["smoothness"],
                    reference_video_path=settings["reference_video_path"],
                    reference_influence=settings["reference_influence"],
                    reference_mode=settings["reference_mode"],
                    source_preservation=settings["source_preservation"],
                    progress_callback=progress_callback,
                )
                saved_path = save_video_frames(frames, settings["output_path"], fps=settings["fps"])

                def finish():
                    self.last_video_path = saved_path
                    self.generate_video_btn.config(state=tk.NORMAL)
                    self.video_progress.config(value=settings["total_frames"], maximum=settings["total_frames"])
                    self.video_status_label.config(text=f"Done. Saved video to: {saved_path}")
                    self.video_info_label.config(text=(
                        f"Rendered {len(frames)} frame(s) at {settings['fps']} FPS using {settings['reference_mode']} mode. "
                        f"If MP4 export wasn't available, the app automatically fell back to GIF."
                    ))
                self.after(0, finish)
            except Exception as exc:
                error_text = str(exc)[:700]
                def fail():
                    self.generate_video_btn.config(state=tk.NORMAL)
                    self.video_status_label.config(text=f"Error: {error_text}")
                    self.video_info_label.config(text="Video generation failed. If this happened during MP4 export, try GIF output first.")
                self.after(0, fail)

        threading.Thread(target=worker, daemon=True).start()



# ----------------------------------------------------------------------------
# App shell
# ----------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Diffusion Styler")
        root.geometry("980x840")
        root.minsize(820, 680)
        root.configure(bg=BG)

        self._setup_style()

        notebook = ttk.Notebook(root, style="Dark.TNotebook")
        notebook.pack(fill="both", expand=True, padx=0, pady=0)

        self.train_tab = TrainTab(notebook, self)
        self.generate_tab = GenerateTab(notebook, self)
        self.video_tab = VideoTab(notebook, self)

        notebook.add(self.train_tab, text="  Train  ")
        notebook.add(self.generate_tab, text="  Generate  ")
        notebook.add(self.video_tab, text="  Video  ")
        self.notebook = notebook

        # Warn immediately when PyTorch cannot use CUDA. This prevents users
        # with unsupported/very old GPUs from thinking the app is broken after
        # a slow CPU-only training run or a mysterious exit code 1.
        root.after(350, self._show_cuda_warning_if_needed)

    def _show_cuda_warning_if_needed(self):
        report = get_cuda_startup_report()
        if report.get("cuda_available"):
            return
        should_continue = messagebox.askyesno(
            "CUDA GPU Not Available",
            build_cuda_warning_text(report),
            icon="warning",
            default="no",
        )
        if not should_continue:
            self.root.destroy()

    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=BG, foreground=FG, font=(FONT_FAMILY, 10))
        style.configure("Panel.TFrame", background=BG)
        style.configure("Card.TFrame", background=BG_PANEL)
        style.configure("Heading.TLabel", background=BG, foreground=FG, font=(FONT_FAMILY, 14, "bold"))
        style.configure("Field.TLabel", background=BG, foreground=FG_DIM, font=(FONT_FAMILY, 9))
        style.configure("Status.TLabel", background=BG, foreground=FG, font=(FONT_FAMILY, 10))
        style.configure("Meta.TLabel", background=BG_PANEL, foreground=FG_DIM, font=(FONT_FAMILY, 9))

        style.configure(
            "Field.TEntry",
            fieldbackground=BG_FIELD, background=BG_FIELD, foreground=FG,
            insertcolor=FG, borderwidth=0,
        )
        style.configure(
            "Accent.TButton",
            background=ACCENT, foreground="#0c0e12", font=(FONT_FAMILY, 11, "bold"),
            borderwidth=0, padding=(16, 8),
        )
        style.map("Accent.TButton", background=[("active", "#7aa2ff"), ("disabled", ACCENT_DIM)])
        style.configure(
            "Secondary.TButton",
            background=BG_FIELD, foreground=FG, borderwidth=0, padding=(12, 6),
        )
        style.map("Secondary.TButton", background=[("active", "#383d4a")])
        style.configure("Dark.TCheckbutton", background=BG, foreground=FG)

        style.configure("Dark.TNotebook", background=BG, borderwidth=0)
        style.configure(
            "Dark.TNotebook.Tab",
            background=BG_PANEL, foreground=FG_DIM, padding=(14, 8), font=(FONT_FAMILY, 10, "bold"),
        )
        style.map(
            "Dark.TNotebook.Tab",
            background=[("selected", BG)],
            foreground=[("selected", FG)],
        )

        style.configure("TCombobox", fieldbackground=BG_FIELD, background=BG_FIELD, foreground=FG)
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=BG_FIELD, borderwidth=0)
        style.configure("Treeview", background=BG_FIELD, fieldbackground=BG_FIELD, foreground=FG, rowheight=24, borderwidth=0)
        style.configure("Treeview.Heading", background=BG_PANEL, foreground=FG, font=(FONT_FAMILY, 9, "bold"))
        style.map("Treeview", background=[("selected", ACCENT_DIM)], foreground=[("selected", FG)])

    # -- Cross-tab coordination ------------------------------------------
    def set_training_active(self, active: bool):
        """Disable generation tabs while training is running. They share the
        GPU and the output directory, so it's safer to avoid sampling mid-save."""
        state = tk.DISABLED if active else tk.NORMAL
        self.generate_tab.generate_btn.config(state=state)
        self.video_tab.generate_video_btn.config(state=state)
        if active:
            self.generate_tab.gen_status_label.config(text="Training is running — Generate will unlock when it's done.")
            self.video_tab.video_status_label.config(text="Training is running — Video will unlock when it's done.")
        else:
            self.generate_tab.gen_status_label.config(text="Ready.")
            self.video_tab.video_status_label.config(text="Ready. Generate a DDPM dream video.")

    def notify_training_complete(self):
        self.generate_tab.refresh_checkpoints()
        self.video_tab.refresh_models()

    def get_preferred_model_path(self):
        return self.generate_tab.last_model_path or self.generate_tab.get_selected_model_path()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
