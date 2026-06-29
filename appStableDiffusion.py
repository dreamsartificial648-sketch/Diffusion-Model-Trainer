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
PRETRAINED_MODEL = "anton-l/ddpm-butterflies-128"

PIPELINE = None  # lazily-loaded diffusers pipeline, cached across generations
PIPELINE_SOURCE = None  # path/name the cached pipeline was loaded from


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


def get_model_info(path: Path):
    path = Path(path).expanduser()
    try:
        display = str(path.relative_to(ROOT_DIR))
    except ValueError:
        display = str(path)
    return {
        "name": path.name or str(path),
        "path": str(path),
        "display": display,
        "resolution": get_model_resolution(path),
        "modified": get_model_modified(path),
    }


def model_label(path: Path, prefix="Model") -> str:
    info = get_model_info(path)
    return f"{prefix}: {info['display']}  ({info['resolution']}, {info['modified']})"


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
        label = "Latest trained model" if path.resolve() == DEFAULT_OUTPUT_DIR.resolve() else model_label(path)
        results.append((label, str(path), mtime, get_model_info(path)))
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


def generate_images(model_path_or_name: str, seed=None, num_inference_steps=50, batch_size=1):
    """Run inference and return a list of PIL Images.

    Resolution is baked into the trained UNet sample_size, so the Generate tab
    shows it as model info instead of pretending it can be changed safely.
    """
    import torch

    pipeline = load_pipeline(model_path_or_name)
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


def generate_image(model_path_or_name: str, seed=None):
    return generate_images(model_path_or_name, seed=seed, num_inference_steps=50, batch_size=1)[0]


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

        data_dir = Path(self.data_dir_var.get())
        if not data_dir.exists():
            errors.append(f"Dataset folder does not exist: {data_dir}")

        output_dir = Path(self.output_dir_var.get()).expanduser()
        if not output_dir:
            errors.append("Pick an output folder to save the trained model.")

        base_model = self.base_model_var.get().strip()
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
            columns=("resolution", "modified", "path"),
            show="headings",
            height=6,
            selectmode="browse",
        )
        self.model_tree.heading("resolution", text="Resolution")
        self.model_tree.heading("modified", text="Modified")
        self.model_tree.heading("path", text="Model folder")
        self.model_tree.column("resolution", width=90, stretch=False)
        self.model_tree.column("modified", width=135, stretch=False)
        self.model_tree.column("path", width=420, stretch=True)
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

        controls = ttk.Frame(main, style="Card.TFrame")
        controls.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 12), pady=(0, 16))

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
            controls, "Inference steps", from_=10, to=250, default=75,
            tooltip="More steps usually means cleaner/more settled images, but slower generation."
        )
        self.steps.pack(fill="x", padx=12, pady=6)

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
        self.image_output_dir_var = tk.StringVar(value=str(ROOT_DIR / "output" / "generations"))
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
            self.model_tree.insert("", "end", iid=key, values=(info["resolution"], info["modified"], info["display"]))

        if not checkpoints:
            key = "pretrained_demo"
            self._checkpoint_paths[key] = PRETRAINED_MODEL
            self._model_infos[key] = {"display": "Pretrained demo model", "resolution": "128x128", "modified": "online", "path": PRETRAINED_MODEL}
            self.model_tree.insert("", "end", iid=key, values=("128x128", "online", "Pretrained demo model"))

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
        self.gen_status_label.config(text=f"Ready to generate with: {info.get('display', path)}")

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
            self.model_tree.insert("", 0, iid=key, values=(info["resolution"], info["modified"], info["display"]))
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
        chosen = filedialog.askdirectory(initialdir=self.image_output_dir_var.get() or str(ROOT_DIR))
        if chosen:
            self.image_output_dir_var.set(chosen)

    def _randomize_seed(self):
        self.seed_var.set(str(random.randint(0, 2_147_483_647)))

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
            "batch_count": batch_count,
            "preview_size": preview_size,
        }

    def _on_generate(self):
        settings = self._validated_generation_settings()
        if settings is None:
            return

        self.generate_btn.config(state=tk.DISABLED)
        self.gen_status_label.config(text=f"Generating {settings['batch_count']} image(s) at {settings['steps']} steps...")

        def work():
            try:
                images = generate_images(
                    settings["model_path"],
                    seed=settings["seed"],
                    num_inference_steps=settings["steps"],
                    batch_size=settings["batch_count"],
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

    def _save_last_images(self):
        if not self.last_images:
            messagebox.showerror("No images yet", "Generate an image first.")
            return
        out_dir = Path(self.image_output_dir_var.get()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        saved = []
        for idx, img in enumerate(self.last_images, start=1):
            path = out_dir / f"generation_{stamp}_{idx:02d}.png"
            img.save(path)
            saved.append(path)
        self.gen_status_label.config(text=f"Saved {len(saved)} image(s) to: {out_dir}")


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
        style.configure("Treeview", background=BG_FIELD, fieldbackground=BG_FIELD, foreground=FG, rowheight=24, borderwidth=0)
        style.configure("Treeview.Heading", background=BG_PANEL, foreground=FG, font=(FONT_FAMILY, 9, "bold"))
        style.map("Treeview", background=[("selected", ACCENT_DIM)], foreground=[("selected", FG)])

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
