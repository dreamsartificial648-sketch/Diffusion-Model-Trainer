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
PRETRAINED_MODEL = "anton-l/ddpm-butterflies-128"

PIPELINE = None  # lazily-loaded diffusers pipeline, cached across generations
PIPELINE_SOURCE = None  # path/name the cached pipeline was loaded from


# ----------------------------------------------------------------------------
# Model discovery
# ----------------------------------------------------------------------------
def is_loadable_model_dir(path: Path) -> bool:
    """A complete diffusers DDPMPipeline folder contains model_index.json."""
    return path.exists() and path.is_dir() and (path / "model_index.json").exists()


def model_label(path: Path, prefix="Model") -> str:
    try:
        display = path.relative_to(ROOT_DIR)
    except ValueError:
        display = path
    return f"{prefix}: {display}"


def find_checkpoints(output_dir: Path = DEFAULT_MODELS_ROOT):
    """Return every loadable DDPMPipeline folder under output_dir, newest first.

    This is deliberately broader than the old app. Instead of only allowing
    output/model, it finds older saved models too, as long as the folder has
    model_index.json. Raw checkpoint-* folders are still ignored because those
    are accelerator resume states, not Generate-ready pipelines.
    """
    roots = []
    for candidate in [DEFAULT_OUTPUT_DIR, output_dir, DEFAULT_MODELS_ROOT]:
        candidate = Path(candidate)
        if candidate.exists() and candidate not in roots:
            roots.append(candidate)

    found = {}
    for root in roots:
        if is_loadable_model_dir(root):
            found[str(root.resolve())] = root
        if root.exists():
            for model_index in root.rglob("model_index.json"):
                path = model_index.parent
                found[str(path.resolve())] = path

    results = []
    for path in found.values():
        mtime = path.stat().st_mtime
        label = "Latest trained model" if path.resolve() == DEFAULT_OUTPUT_DIR.resolve() else model_label(path)
        results.append((label, str(path), mtime))
    results.sort(key=lambda r: r[2], reverse=True)
    return results


def copy_model_folder(src: Path, dst: Path):
    src = Path(src)
    dst = Path(dst)
    if not is_loadable_model_dir(src):
        raise ValueError(f"Not a complete model folder: {src}")
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"Destination already exists and is not empty: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


# ----------------------------------------------------------------------------
# Generation (runs in a background thread, in-process — inference is quick
# and doesn't need subprocess isolation the way training does)
# ----------------------------------------------------------------------------
def load_pipeline(model_path_or_name: str):
    global PIPELINE, PIPELINE_SOURCE
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
    return pipeline


def generate_image(model_path_or_name: str, seed=None):
    """Run inference and return a PIL Image.

    Note: resolution is NOT a parameter here — it's baked into the trained
    UNet's sample_size and can't be changed at generation time. Resolution
    is a *training*-time setting (see TrainTab), not a generation-time one.
    """
    import torch

    pipeline = load_pipeline(model_path_or_name)
    device = next(pipeline.unet.parameters()).device
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    image = pipeline(
        generator=generator,
        batch_size=1,
        num_inference_steps=50,
        output_type="pil",
    ).images[0]
    return image


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
            settings, "Batch size", default=8,
            tooltip="Images processed per step. Higher uses more VRAM but trains faster per epoch. "
                    "8 is a good starting point on a 12GB Ampere card (RTX 3060) at 128px; "
                    "lower this if you run out of memory.",
        )
        self.learning_rate = LabeledEntry(
            settings, "Learning rate", default="1e-4",
            tooltip="How fast the model updates. 1e-4 is a solid default; lower it if loss becomes unstable.",
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

        self.epochs.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        self.resolution.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        self.batch_size.grid(row=0, column=2, sticky="ew", padx=6, pady=6)
        self.learning_rate.grid(row=1, column=0, sticky="ew", padx=6, pady=6)
        self.mixed_precision.grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        self.preview_every.grid(row=1, column=2, sticky="ew", padx=6, pady=6)

        # Dataset folder picker
        data_row = ttk.Frame(settings, style="Panel.TFrame")
        data_row.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
        ttk.Label(data_row, text="Dataset folder", style="Field.TLabel").pack(anchor="w")
        picker_row = ttk.Frame(data_row, style="Panel.TFrame")
        picker_row.pack(fill="x", pady=(2, 0))
        self.data_dir_var = tk.StringVar(value=str(DEFAULT_DATA_DIR))
        data_entry = ttk.Entry(picker_row, textvariable=self.data_dir_var, style="Field.TEntry")
        data_entry.pack(side="left", fill="x", expand=True)
        browse_btn = ttk.Button(picker_row, text="Browse...", command=self._browse_data_dir, style="Secondary.TButton")
        browse_btn.pack(side="left", padx=(6, 0))
        Tooltip(data_entry, "Folder containing your training images (subfolder layout: data_dir/train/*.png).")

        # Model loading / output saving
        model_row = ttk.Frame(settings, style="Panel.TFrame")
        model_row.grid(row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=(8, 2))
        model_row.columnconfigure(1, weight=1)
        ttk.Label(model_row, text="Start from trained model", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.base_model_var = tk.StringVar(value="")
        base_entry = ttk.Entry(model_row, textvariable=self.base_model_var, style="Field.TEntry")
        base_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(model_row, text="Load Model...", command=self._browse_base_model, style="Secondary.TButton").grid(row=0, column=2, padx=(6, 0))
        ttk.Button(model_row, text="Clear", command=lambda: self.base_model_var.set(""), style="Secondary.TButton").grid(row=0, column=3, padx=(6, 0))
        Tooltip(base_entry, "Optional: pick an older trained DDPMPipeline folder to continue/fine-tune from. Leave blank to train from scratch.")

        out_row = ttk.Frame(settings, style="Panel.TFrame")
        out_row.grid(row=4, column=0, columnspan=3, sticky="ew", padx=6, pady=(2, 6))
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

    def _browse_data_dir(self):
        chosen = filedialog.askdirectory(initialdir=self.data_dir_var.get() or str(ROOT_DIR))
        if chosen:
            self.data_dir_var.set(chosen)

    def _browse_base_model(self):
        chosen = filedialog.askdirectory(initialdir=str(DEFAULT_MODELS_ROOT if DEFAULT_MODELS_ROOT.exists() else ROOT_DIR))
        if not chosen:
            return
        path = Path(chosen)
        if not is_loadable_model_dir(path):
            messagebox.showerror("Not a loadable model", "Pick a trained model folder that contains model_index.json.")
            return
        self.base_model_var.set(str(path))
        self._try_apply_model_resolution(path)

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

        data_dir = Path(self.data_dir_var.get())
        if not data_dir.exists():
            errors.append(f"Dataset folder does not exist: {data_dir}")

        output_dir = Path(self.output_dir_var.get()).expanduser()
        if not output_dir:
            errors.append("Pick an output folder to save the trained model.")

        base_model = self.base_model_var.get().strip()
        if base_model and not is_loadable_model_dir(Path(base_model)):
            errors.append("Start-from model must be a complete trained model folder containing model_index.json.")

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
            "base_model": base_model,
            "use_ema": self.use_ema_var.get(),
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

        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--train_data_dir", settings["data_dir"],
            "--output_dir", str(output_dir),
            "--resolution", str(settings["resolution"]),
            "--train_batch_size", str(settings["batch_size"]),
            "--num_epochs", str(settings["epochs"]),
            "--learning_rate", str(settings["learning_rate"]),
            "--mixed_precision", settings["mixed_precision"],
            "--save_images_epochs", str(settings["preview_every"]),
            "--save_model_epochs", str(settings["preview_every"]),
            "--dataloader_num_workers", "4",
            "--stop_signal_file", str(self.stop_signal_file),
            "--gui_progress",
        ]
        if settings["base_model"]:
            cmd.extend(["--pretrained_model_path", settings["base_model"]])
        if settings["use_ema"]:
            cmd.append("--use_ema")

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
            return

        self.training_active = True
        self.start_time = time.time()
        self.user_stopped = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Starting...")
        self.progress_bar.config(value=0, maximum=100)
        self.last_saved_model_dir = str(output_dir)
        self.save_btn.config(state=tk.DISABLED)
        self.preview_label.config(image="", text="Scanning dataset...")
        self._clear_log()
        self.app.set_training_active(True)

        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()

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
        elif kind == "start":
            total_epochs = evt.get("num_epochs", "?")
            self.total_epochs = evt.get("num_epochs")
            self.status_label.config(text=f"Training started ({total_epochs} epochs)...")
            self.epoch_label.config(text=f"Epoch: 0 / {total_epochs}")
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
        elif kind == "model_saved":
            out = evt.get("output_dir")
            if out:
                self.last_saved_model_dir = out
                self.save_btn.config(state=tk.NORMAL)
                self._append_log(f"Saved loadable model to: {out}")
        elif kind == "stopped":
            self.status_label.config(text="Training stopped and saved.")
            self.eta_label.config(text="Stopped")
            self._finish_training(success=True)
        elif kind == "done":
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
            self.status_label.config(text="Training complete!")
            self._finish_training(success=True)
        elif self.user_stopped:
            self.status_label.config(text="Training stopped.")
            self._finish_training(success=False)
        else:
            self.status_label.config(text=f"Training stopped with an error (exit code {returncode}). See log below.")
            self._finish_training(success=False)

    def _finish_training(self, success):
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
        self._build_ui()
        self.refresh_checkpoints()

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, style="Panel.TFrame")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=16)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Model", style="Field.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, state="readonly", width=40)
        self.model_combo.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        refresh_btn = ttk.Button(top, text="Refresh", command=self.refresh_checkpoints, style="Secondary.TButton")
        refresh_btn.grid(row=0, column=2)
        browse_btn = ttk.Button(top, text="Load Any Model...", command=self._browse_model, style="Secondary.TButton")
        browse_btn.grid(row=0, column=3, padx=(6, 0))
        save_btn = ttk.Button(top, text="Save Selected As...", command=self._save_selected_model_as, style="Secondary.TButton")
        save_btn.grid(row=0, column=4, padx=(6, 0))

        seed_row = ttk.Frame(self, style="Panel.TFrame")
        seed_row.grid(row=2, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 8))
        ttk.Label(seed_row, text="Seed (optional):", style="Field.TLabel").pack(side="left", padx=(0, 6))
        self.seed_var = tk.StringVar(value="")
        ttk.Entry(seed_row, textvariable=self.seed_var, width=12, style="Field.TEntry").pack(side="left")

        self.generate_btn = ttk.Button(
            seed_row, text="Generate", command=self._on_generate, style="Accent.TButton"
        )
        self.generate_btn.pack(side="left", padx=(16, 0))

        self.gen_status_label = ttk.Label(self, text="Ready.", style="Status.TLabel")
        self.gen_status_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 8))

        image_frame = ttk.Frame(self, style="Card.TFrame")
        image_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=16, pady=(0, 16))
        self.rowconfigure(4, weight=1)
        self.image_label = tk.Label(
            image_frame, text="Image will appear here", bg=BG_FIELD, fg=FG_DIM, font=(FONT_FAMILY, 12),
        )
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)

    def refresh_checkpoints(self):
        checkpoints = find_checkpoints(DEFAULT_MODELS_ROOT)
        labels = [c[0] for c in checkpoints]
        self._checkpoint_paths = {c[0]: c[1] for c in checkpoints}
        if not labels:
            labels = ["No trained model yet (will use pretrained demo model)"]
            self._checkpoint_paths[labels[0]] = PRETRAINED_MODEL
        self.model_combo.config(values=labels)
        if self.model_var.get() not in labels:
            self.model_var.set(labels[0])

    def _browse_model(self):
        chosen = filedialog.askdirectory(initialdir=str(DEFAULT_MODELS_ROOT if DEFAULT_MODELS_ROOT.exists() else ROOT_DIR))
        if not chosen:
            return
        path = Path(chosen)
        if not is_loadable_model_dir(path):
            messagebox.showerror("Not a loadable model", "Pick a trained model folder that contains model_index.json.")
            return
        label = model_label(path, prefix="Loaded")
        self._checkpoint_paths[label] = str(path)
        values = list(self.model_combo.cget("values"))
        if label not in values:
            values.insert(0, label)
            self.model_combo.config(values=values)
        self.model_var.set(label)
        self.gen_status_label.config(text=f"Loaded model folder: {path}")

    def _save_selected_model_as(self):
        label = self.model_var.get()
        src = self._checkpoint_paths.get(label)
        if not src or src == PRETRAINED_MODEL:
            messagebox.showerror("Cannot save this", "Pick a local trained model folder first.")
            return
        src_path = Path(src)
        if not is_loadable_model_dir(src_path):
            messagebox.showerror("Cannot save this", "The selected item is not a complete local model folder.")
            return
        chosen = filedialog.askdirectory(initialdir=str(DEFAULT_MODELS_ROOT if DEFAULT_MODELS_ROOT.exists() else ROOT_DIR))
        if not chosen:
            return
        dst = Path(chosen)
        try:
            copy_model_folder(src_path, dst)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        self.refresh_checkpoints()
        self.gen_status_label.config(text=f"Saved selected model to: {dst}")

    def _on_generate(self):
        label = self.model_var.get()
        model_path = self._checkpoint_paths.get(label, PRETRAINED_MODEL)
        seed_val = self.seed_var.get().strip()
        try:
            seed = int(seed_val) if seed_val else None
        except ValueError:
            messagebox.showerror("Invalid seed", "Seed must be a whole number, or left blank.")
            return

        self.generate_btn.config(state=tk.DISABLED)
        self.gen_status_label.config(text="Generating...")

        def work():
            try:
                image = generate_image(model_path, seed=seed)
                from PIL import ImageTk

                display_image = scale_for_display(image, target_size=512)
                photo = ImageTk.PhotoImage(display_image)

                def update_ui():
                    self.photo_ref = photo
                    self.image_label.config(image=photo, text="")
                    self.gen_status_label.config(
                        text=f"Done. (generated at {image.width}x{image.height}, shown enlarged)"
                    )
                    self.generate_btn.config(state=tk.NORMAL)

                self.after(0, update_ui)
            except Exception as e:
                error_text = str(e)[:200]

                def show_error():
                    self.gen_status_label.config(text=f"Error: {error_text}")
                    self.generate_btn.config(state=tk.NORMAL)

                self.after(0, show_error)

        threading.Thread(target=work, daemon=True).start()


# ----------------------------------------------------------------------------
# App shell
# ----------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Diffusion Styler")
        root.geometry("760x780")
        root.minsize(640, 600)
        root.configure(bg=BG)

        self._setup_style()

        notebook = ttk.Notebook(root, style="Dark.TNotebook")
        notebook.pack(fill="both", expand=True, padx=0, pady=0)

        self.train_tab = TrainTab(notebook, self)
        self.generate_tab = GenerateTab(notebook, self)

        notebook.add(self.train_tab, text="  Train  ")
        notebook.add(self.generate_tab, text="  Generate  ")
        self.notebook = notebook

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

    # -- Cross-tab coordination ------------------------------------------
    def set_training_active(self, active: bool):
        """Disable Generate while training is running. The two share the
        GPU and, more importantly, the same output_dir; generating mid-save
        could read a half-written checkpoint."""
        state = tk.DISABLED if active else tk.NORMAL
        self.generate_tab.generate_btn.config(state=state)
        if active:
            self.generate_tab.gen_status_label.config(text="Training is running — Generate will unlock when it's done.")
        else:
            self.generate_tab.gen_status_label.config(text="Ready.")

    def notify_training_complete(self):
        self.generate_tab.refresh_checkpoints()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
