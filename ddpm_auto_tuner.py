"""
DDPM Training Auto-Tuner
------------------------
A small standalone helper for the Diffusion Styler/DDPM app.

What it does:
  1. Lets you choose your dataset folder and train.py file.
  2. Randomly copies up to 300 images into a temporary benchmark dataset.
  3. Runs short 1-epoch training trials with different speed settings.
  4. Measures actual training step speed from PROGRESS_JSON lines.
  5. Prints the recommended settings for you to copy into the main DDPM GUI.

It DOES NOT edit ddpm_gui_settings.json yet. This first version is read-only
for your real app settings.
"""

from __future__ import annotations

import json
import os
import queue
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SAMPLE_IMAGES = 300
APP_TITLE = "DDPM Training Auto-Tuner"

BG = "#1b1d23"
PANEL = "#22252e"
FIELD = "#2a2e38"
FG = "#e8e9ed"
DIM = "#9a9ea8"
GOOD = "#5ec98f"
BAD = "#e0615f"
WARN = "#e0a23c"
ACCENT = "#5b8cff"
FONT = "Segoe UI"


def here() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


@dataclass
class TrialConfig:
    batch_size: int
    workers: int
    prefetch: int
    pin_memory: bool
    mixed_precision: str = "fp16"
    tf32: bool = True

    def label(self) -> str:
        pin = "pin" if self.pin_memory else "no-pin"
        tf32 = "tf32" if self.tf32 else "no-tf32"
        return f"batch={self.batch_size}, workers={self.workers}, prefetch={self.prefetch}, {pin}, {self.mixed_precision}, {tf32}"


@dataclass
class TrialResult:
    config: TrialConfig
    ok: bool
    images_per_second: float = 0.0
    seconds_per_step: float = 0.0
    measured_steps: int = 0
    total_elapsed_seconds: float = 0.0
    error: str = ""
    return_code: int | None = None

    def safety_score(self) -> float:
        """Prefer fast settings, but slightly penalize extreme batch sizes.

        This cannot directly know VRAM headroom because train.py does not emit
        max CUDA memory yet, so this uses a conservative speed score. If two
        trials are close, the smaller batch wins to avoid living at VRAM's edge.
        """
        if not self.ok or self.images_per_second <= 0:
            return -1.0
        batch_penalty = max(0, self.config.batch_size - 28) * 0.015
        return self.images_per_second * (1.0 - batch_penalty)


def find_images(data_dir: Path) -> list[Path]:
    data_dir = Path(data_dir).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {data_dir}")
    images = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort()
    if not images:
        raise RuntimeError(f"No supported images found in: {data_dir}")
    return images


def prepare_sample_dataset(source_dir: Path, sample_count: int, log) -> Path:
    images = find_images(source_dir)
    sample_count = max(1, min(int(sample_count), len(images)))
    chosen = random.sample(images, sample_count) if len(images) > sample_count else images

    temp_root = Path(tempfile.mkdtemp(prefix="ddpm_auto_tune_dataset_"))
    sample_dir = temp_root / "data"
    sample_dir.mkdir(parents=True, exist_ok=True)

    log(f"Creating benchmark dataset with {len(chosen)} random image(s)...")
    for index, src in enumerate(chosen):
        safe_name = f"sample_{index:04d}{src.suffix.lower()}"
        dst = sample_dir / safe_name
        try:
            shutil.copy2(src, dst)
        except Exception:
            shutil.copy(src, dst)
    return sample_dir


def parse_progress_json(line: str) -> dict | None:
    prefix = "PROGRESS_JSON:"
    idx = line.find(prefix)
    if idx < 0:
        return None
    payload = line[idx + len(prefix):].strip()
    try:
        return json.loads(payload)
    except Exception:
        return None


def run_trial(
    train_py: Path,
    sample_dir: Path,
    output_root: Path,
    cfg: TrialConfig,
    resolution: int,
    learning_rate: str,
    timeout_seconds: int,
    stop_event: threading.Event,
    log,
) -> TrialResult:
    trial_name = f"trial_b{cfg.batch_size}_w{cfg.workers}_p{cfg.prefetch}_{'pin' if cfg.pin_memory else 'nopin'}_{int(time.time() * 1000)}"
    output_dir = output_root / trial_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(train_py),
        "--train_data_dir", str(sample_dir),
        "--output_dir", str(output_dir),
        "--model_name", trial_name,
        "--resolution", str(int(resolution)),
        "--num_epochs", "1",
        "--train_batch_size", str(cfg.batch_size),
        "--dataloader_num_workers", str(cfg.workers),
        "--prefetch_factor", str(max(1, cfg.prefetch)),
        "--pin_memory", "true" if cfg.pin_memory else "false",
        "--mixed_precision", cfg.mixed_precision,
        "--tf32", "true" if cfg.tf32 else "false",
        "--training_intensity", "100",
        "--learning_rate", str(learning_rate),
        "--checkpointing_steps", "0",
        "--save_images_epochs", "999999",
        "--save_model_epochs", "999999",
        "--eval_batch_size", "1",
        "--preview_num_inference_steps", "5",
        "--storage_saver", "true",
        "--gui_progress",
    ]

    start = time.time()
    step_times: list[float] = []
    error_message = ""
    return_code = None

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(train_py.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            creationflags=creationflags,
        )
    except Exception as exc:
        return TrialResult(cfg, ok=False, error=f"Could not start train.py: {exc}")

    try:
        assert proc.stdout is not None
        while True:
            if stop_event.is_set():
                try:
                    proc.terminate()
                except Exception:
                    pass
                return TrialResult(cfg, ok=False, error="Stopped by user", return_code=None)

            if time.time() - start > timeout_seconds:
                try:
                    proc.kill()
                except Exception:
                    pass
                return TrialResult(cfg, ok=False, error=f"Timed out after {timeout_seconds}s", return_code=None)

            line = proc.stdout.readline()
            if line:
                low = line.lower()
                if "out of memory" in low or "cuda error" in low or "nan" in low:
                    error_message = line.strip()[-400:]
                event = parse_progress_json(line)
                if event:
                    if event.get("event") == "step":
                        step_times.append(float(event.get("ts", time.time())))
                    elif event.get("event") == "error":
                        error_message = str(event.get("message", "Unknown train.py error"))
                    elif event.get("event") == "hardware_warning":
                        log("WARNING: train.py says CUDA is not available. Benchmark will not represent GPU speed.")
            elif proc.poll() is not None:
                break
            else:
                time.sleep(0.03)

        return_code = proc.poll()
    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

    elapsed = time.time() - start

    # Focus on steady training step speed. Ignore the first interval because it
    # can include CUDA/cudnn warmup, worker startup, and cold file cache weirdness.
    deltas = [b - a for a, b in zip(step_times, step_times[1:]) if b > a]
    if len(deltas) >= 3:
        useful = deltas[1:]
    else:
        useful = deltas

    if useful:
        seconds_per_step = statistics.median(useful)
        images_per_second = cfg.batch_size / seconds_per_step if seconds_per_step > 0 else 0.0
    elif step_times:
        # Fallback for very tiny datasets / giant batches.
        seconds_per_step = elapsed / max(1, len(step_times))
        images_per_second = cfg.batch_size / seconds_per_step if seconds_per_step > 0 else 0.0
    else:
        seconds_per_step = 0.0
        images_per_second = 0.0

    ok = (return_code == 0) and images_per_second > 0 and not error_message
    if return_code not in (0, None) and not error_message:
        error_message = f"train.py exited with code {return_code}"

    # Delete the benchmark output model so repeated tuning does not create a storage monster.
    try:
        shutil.rmtree(output_dir, ignore_errors=True)
    except Exception:
        pass

    return TrialResult(
        config=cfg,
        ok=ok,
        images_per_second=images_per_second,
        seconds_per_step=seconds_per_step,
        measured_steps=len(step_times),
        total_elapsed_seconds=elapsed,
        error=error_message,
        return_code=return_code,
    )


def build_trial_plan(max_batch: int, workers_hint: int, prefetch_hint: int) -> list[TrialConfig]:
    """Staged-ish plan without requiring user interaction mid-run.

    A true adaptive search is possible, but this static plan is easier to trust:
      - Several batch sizes around common RTX 3060 values.
      - Several worker/prefetch variants around the likely best CPU feeder setup.
      - A pin-memory off test, because occasionally Windows setups act weird.
    """
    batches = [8, 12, 16, 20, 24, 28, 32]
    batches = [b for b in batches if b <= max_batch]
    if not batches:
        batches = [min(8, max_batch)]

    plan: list[TrialConfig] = []

    # Batch-size sweep using the user's current-ish feeder settings.
    for b in batches:
        plan.append(TrialConfig(batch_size=b, workers=workers_hint, prefetch=prefetch_hint, pin_memory=True))

    # CPU feeder sweep at the strongest normal batch sizes.
    target_batches = [b for b in [16, 20, 24, 28, 32] if b <= max_batch]
    target_batch = target_batches[-1] if target_batches else batches[-1]
    worker_values = sorted(set([0, 2, 4, 6, 8, workers_hint]))
    for w in worker_values:
        if w == 0:
            plan.append(TrialConfig(batch_size=target_batch, workers=0, prefetch=1, pin_memory=True))
        else:
            for p in sorted(set([2, 4, 8, prefetch_hint])):
                plan.append(TrialConfig(batch_size=target_batch, workers=w, prefetch=p, pin_memory=True))

    # Pin memory comparison.
    plan.append(TrialConfig(batch_size=target_batch, workers=workers_hint, prefetch=prefetch_hint, pin_memory=False))

    # Deduplicate while preserving order.
    seen = set()
    unique: list[TrialConfig] = []
    for cfg in plan:
        key = (cfg.batch_size, cfg.workers, cfg.prefetch, cfg.pin_memory, cfg.mixed_precision, cfg.tf32)
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def format_seconds(seconds: float) -> str:
    seconds = float(seconds or 0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes}m {sec}s"


class AutoTunerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("980x760")
        self.root.minsize(860, 640)
        self.root.configure(bg=BG)
        self.q: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.results: list[TrialResult] = []
        self.report_path: Path | None = None

        self.train_py_var = tk.StringVar(value=str(here() / "train.py"))
        self.data_dir_var = tk.StringVar(value=str(here() / "data"))
        self.sample_count_var = tk.StringVar(value=str(DEFAULT_SAMPLE_IMAGES))
        self.resolution_var = tk.StringVar(value="128")
        self.learning_rate_var = tk.StringVar(value="2e-4")
        self.max_batch_var = tk.StringVar(value="32")
        self.workers_var = tk.StringVar(value="6")
        self.prefetch_var = tk.StringVar(value="4")
        self.timeout_var = tk.StringVar(value="240")

        self._style()
        self._build_ui()
        self.root.after(100, self._poll)

    def _style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=(FONT, 10))
        style.configure("Dim.TLabel", background=BG, foreground=DIM, font=(FONT, 9))
        style.configure("Panel.TLabel", background=PANEL, foreground=FG, font=(FONT, 10))
        style.configure("Heading.TLabel", background=BG, foreground=FG, font=(FONT, 16, "bold"))
        style.configure("Good.TLabel", background=BG, foreground=GOOD, font=(FONT, 10, "bold"))
        style.configure("Bad.TLabel", background=BG, foreground=BAD, font=(FONT, 10, "bold"))
        style.configure("TButton", font=(FONT, 10))
        style.configure("Treeview", background=FIELD, foreground=FG, fieldbackground=FIELD, rowheight=26)
        style.configure("Treeview.Heading", font=(FONT, 10, "bold"))

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=16, pady=16)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(3, weight=1)

        ttk.Label(main, text="DDPM Training Auto-Tuner", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            main,
            text="Benchmarks short 1-epoch trials on a 300-image sample and recommends speed settings. It will not edit your real app settings.",
            style="Dim.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 14))

        form = ttk.Frame(main, style="Panel.TFrame")
        form.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        for c in range(4):
            form.columnconfigure(c, weight=1)

        self._path_row(form, 0, "train.py", self.train_py_var, self._browse_train)
        self._path_row(form, 1, "Dataset folder", self.data_dir_var, self._browse_data)
        self._entry(form, 2, 0, "Sample images", self.sample_count_var)
        self._entry(form, 2, 1, "Resolution", self.resolution_var)
        self._entry(form, 2, 2, "Learning rate", self.learning_rate_var)
        self._entry(form, 2, 3, "Max batch", self.max_batch_var)
        self._entry(form, 3, 0, "Worker hint", self.workers_var)
        self._entry(form, 3, 1, "Prefetch hint", self.prefetch_var)
        self._entry(form, 3, 2, "Timeout/trial seconds", self.timeout_var)

        buttons = ttk.Frame(form, style="Panel.TFrame")
        buttons.grid(row=3, column=3, sticky="ew", padx=8, pady=8)
        self.start_btn = ttk.Button(buttons, text="Start Auto-Tune", command=self.start)
        self.start_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left")

        table_frame = ttk.Frame(main)
        table_frame.grid(row=3, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = ("status", "ips", "step", "batch", "workers", "prefetch", "pin", "elapsed", "error")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)
        headings = {
            "status": "Status",
            "ips": "Images/sec",
            "step": "Sec/step",
            "batch": "Batch",
            "workers": "Workers",
            "prefetch": "Prefetch",
            "pin": "Pin RAM",
            "elapsed": "Elapsed",
            "error": "Error / Notes",
        }
        widths = {"status": 80, "ips": 95, "step": 90, "batch": 70, "workers": 80, "prefetch": 85, "pin": 75, "elapsed": 80, "error": 320}
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.grid(row=0, column=1, sticky="ns")

        self.recommendation = tk.Text(main, height=8, bg=FIELD, fg=FG, insertbackground=FG, wrap="word", relief="flat")
        self.recommendation.grid(row=4, column=0, sticky="ew", pady=(12, 8))
        self.recommendation.insert("end", "Recommended settings will appear here after the benchmark.\n")
        self.recommendation.configure(state="disabled")

        self.log_box = tk.Text(main, height=9, bg="#111318", fg=FG, insertbackground=FG, wrap="word", relief="flat")
        self.log_box.grid(row=5, column=0, sticky="ew")
        self._log("Ready.")

    def _path_row(self, parent, row, label, var, command):
        ttk.Label(parent, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=(8, 2))
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=8, pady=(8, 2))
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=3, sticky="ew", padx=8, pady=(8, 2))

    def _entry(self, parent, row, col, label, var):
        box = ttk.Frame(parent, style="Panel.TFrame")
        box.grid(row=row, column=col, sticky="ew", padx=8, pady=8)
        ttk.Label(box, text=label, style="Panel.TLabel").pack(anchor="w")
        ttk.Entry(box, textvariable=var).pack(fill="x", pady=(2, 0))

    def _browse_train(self):
        path = filedialog.askopenfilename(title="Select train.py", filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if path:
            self.train_py_var.set(path)

    def _browse_data(self):
        path = filedialog.askdirectory(title="Select dataset folder")
        if path:
            self.data_dir_var.set(path)

    def _log(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{time.strftime('%H:%M:%S')}  {text}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def log_from_thread(self, text: str):
        self.q.put(("log", text))

    def _set_recommendation(self, text: str):
        self.recommendation.configure(state="normal")
        self.recommendation.delete("1.0", "end")
        self.recommendation.insert("end", text)
        self.recommendation.configure(state="disabled")

    def _add_result_row(self, result: TrialResult):
        cfg = result.config
        status = "OK" if result.ok else "FAILED"
        values = (
            status,
            f"{result.images_per_second:.2f}" if result.ok else "-",
            f"{result.seconds_per_step:.3f}" if result.ok else "-",
            cfg.batch_size,
            cfg.workers,
            cfg.prefetch,
            "on" if cfg.pin_memory else "off",
            format_seconds(result.total_elapsed_seconds),
            result.error[:220],
        )
        self.tree.insert("", "end", values=values)

    def start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.tree.delete(*self.tree.get_children())
        self.results.clear()
        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._set_recommendation("Benchmark running...\n")
        self.worker_thread = threading.Thread(target=self._run_benchmark_thread, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.stop_event.set()
        self._log("Stop requested. Current trial will be terminated.")

    def _run_benchmark_thread(self):
        try:
            train_py = Path(self.train_py_var.get()).expanduser()
            data_dir = Path(self.data_dir_var.get()).expanduser()
            if not train_py.exists():
                raise FileNotFoundError(f"train.py not found: {train_py}")

            sample_count = int(self.sample_count_var.get())
            resolution = int(self.resolution_var.get())
            max_batch = int(self.max_batch_var.get())
            workers_hint = int(self.workers_var.get())
            prefetch_hint = int(self.prefetch_var.get())
            timeout_seconds = int(self.timeout_var.get())
            learning_rate = self.learning_rate_var.get().strip() or "2e-4"

            sample_dir = prepare_sample_dataset(data_dir, sample_count, self.log_from_thread)
            output_root = Path(tempfile.mkdtemp(prefix="ddpm_auto_tune_runs_"))
            self.log_from_thread(f"Benchmark output temp folder: {output_root}")

            plan = build_trial_plan(max_batch=max_batch, workers_hint=workers_hint, prefetch_hint=prefetch_hint)
            self.log_from_thread(f"Planned {len(plan)} trial(s).")

            for i, cfg in enumerate(plan, 1):
                if self.stop_event.is_set():
                    break
                self.log_from_thread(f"Trial {i}/{len(plan)}: {cfg.label()}")
                result = run_trial(
                    train_py=train_py,
                    sample_dir=sample_dir,
                    output_root=output_root,
                    cfg=cfg,
                    resolution=resolution,
                    learning_rate=learning_rate,
                    timeout_seconds=timeout_seconds,
                    stop_event=self.stop_event,
                    log=self.log_from_thread,
                )
                self.results.append(result)
                self.q.put(("result", result))
                if result.ok:
                    self.log_from_thread(f"  OK: {result.images_per_second:.2f} images/sec ({result.seconds_per_step:.3f}s/step)")
                else:
                    self.log_from_thread(f"  FAILED: {result.error or 'unknown error'}")

            # Clean sample/output temp data.
            try:
                shutil.rmtree(sample_dir.parent, ignore_errors=True)
                shutil.rmtree(output_root, ignore_errors=True)
            except Exception:
                pass

            self.q.put(("done", None))
        except Exception as exc:
            self.q.put(("fatal", f"{exc}\n\n{traceback.format_exc()}"))

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(str(payload))
                elif kind == "result":
                    self._add_result_row(payload)
                elif kind == "done":
                    self._finish()
                elif kind == "fatal":
                    self._log(str(payload))
                    messagebox.showerror(APP_TITLE, str(payload)[:1200])
                    self._finish(failed=True)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _finish(self, failed: bool = False):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if failed:
            self._set_recommendation("Benchmark failed. Check the log box for details.\n")
            return
        if not self.results:
            self._set_recommendation("No trials completed.\n")
            return
        recommendation = self._build_recommendation_text()
        self._set_recommendation(recommendation)
        self._write_report(recommendation)
        self._log("Benchmark finished.")
        if self.report_path:
            self._log(f"Report saved: {self.report_path}")

    def _build_recommendation_text(self) -> str:
        successful = [r for r in self.results if r.ok]
        failed = [r for r in self.results if not r.ok]
        if not successful:
            return (
                "No stable settings were found.\n\n"
                "Try lowering Max batch to 16 or 20, using resolution 128, and confirming train.py can train normally from the main app first."
            )

        successful.sort(key=lambda r: r.safety_score(), reverse=True)
        best = successful[0]
        cfg = best.config
        lines = [
            "Recommended DDPM training settings to copy into the GUI:",
            "",
            f"Batch size: {cfg.batch_size}",
            f"DataLoader workers: {cfg.workers}",
            f"Prefetch factor: {cfg.prefetch}",
            f"Pin memory: {'Enabled' if cfg.pin_memory else 'Disabled'}",
            f"Mixed precision: {cfg.mixed_precision}",
            f"TF32 compute: {'Enabled' if cfg.tf32 else 'Disabled'}",
            "Training intensity: 100% for max speed, or 75% for cooler/overnight training",
            "",
            f"Measured speed: {best.images_per_second:.2f} images/sec",
            f"Median step time: {best.seconds_per_step:.3f}s/step",
            f"Measured steps: {best.measured_steps}",
            "",
            "Top stable trials:",
        ]
        for rank, r in enumerate(successful[:5], 1):
            lines.append(
                f"{rank}. {r.images_per_second:.2f} img/s | batch {r.config.batch_size}, workers {r.config.workers}, "
                f"prefetch {r.config.prefetch}, pin {'on' if r.config.pin_memory else 'off'}"
            )
        if failed:
            lines.extend(["", f"Failed/unstable trials: {len(failed)}. Usually this means too much VRAM, worker issues, or timeout."])
        lines.extend([
            "",
            "Note: this is a speed benchmark, not a quality benchmark. Keep your usual epochs/loss/noise schedule settings for model quality.",
        ])
        return "\n".join(lines)

    def _write_report(self, recommendation: str):
        try:
            report_dir = here()
            path = report_dir / f"ddpm_auto_tune_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
            data = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "recommendation_text": recommendation,
                "results": [
                    {
                        **asdict(r),
                        "config": asdict(r.config),
                    }
                    for r in self.results
                ],
            }
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.report_path = path
        except Exception as exc:
            self._log(f"Could not write report: {exc}")


def main():
    root = tk.Tk()
    AutoTunerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
