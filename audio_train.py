"""Small, dependency-free waveform reconstruction trainer for Sound tab.

This intentionally establishes a reliable audio baseline before attempting
generative diffusion: train on fixed short clips, then reconstruct a held clip.
If this cannot reproduce recognizable audio, a diffusion sampler will not help.
"""
import argparse
import json
import math
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def progress(**values):
    print("AUDIO_PROGRESS_JSON:" + json.dumps({"ts": time.time(), **values}), flush=True)


def read_wav(path):
    with wave.open(str(path), "rb") as handle:
        channels, width, rate, frames = handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()
        raw = handle.readframes(frames)
    if width == 1:
        data = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 3:
        packed = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        ints = packed[:, 0].astype(np.int32) | (packed[:, 1].astype(np.int32) << 8) | (packed[:, 2].astype(np.int32) << 16)
        ints[ints >= 0x800000] -= 0x1000000
        data = ints.astype(np.float32) / 8388608.0
    elif width == 4:
        data = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV bit depth: {width * 8}-bit")
    return data.reshape(-1, channels).mean(axis=1), rate


def write_wav(path, samples, rate):
    samples = np.clip(np.asarray(samples), -1, 1)
    pcm = (samples * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(rate); handle.writeframes(pcm)


def resample(samples, source_rate, target_rate):
    if source_rate == target_rate:
        return samples
    target_length = max(1, round(len(samples) * target_rate / source_rate))
    return np.interp(np.linspace(0, len(samples) - 1, target_length), np.arange(len(samples)), samples).astype(np.float32)


def read_audio(path, temporary_dir, index):
    """Read WAV directly, or decode other ffmpeg-supported audio to a temporary WAV.

    The converted file lives only inside TemporaryDirectory and never changes
    the user's original audio file.
    """
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return read_wav(path)
    converted = Path(temporary_dir) / f"decoded_{index:04d}.wav"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-c:a", "pcm_s16le", str(converted)],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("MP3 decoding needs ffmpeg, but ffmpeg was not found on this computer.") from exc
    if result.returncode != 0 or not converted.exists():
        detail = (result.stderr or "unknown decoder error").strip()
        raise RuntimeError(f"Could not decode {path.name}: {detail}")
    return read_wav(converted)


class WaveAutoencoder(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, width, 9, stride=2, padding=4), nn.LeakyReLU(0.2),
            nn.Conv1d(width, width * 2, 9, stride=2, padding=4), nn.LeakyReLU(0.2),
            nn.Conv1d(width * 2, width * 3, 9, stride=2, padding=4), nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(width * 3, width * 2, 8, stride=2, padding=3), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(width * 2, width, 8, stride=2, padding=3), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(width, 1, 8, stride=2, padding=3), nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=[], help="One or more WAV or MP3 files (MP3 is decoded temporarily with ffmpeg).")
    parser.add_argument("--inputs_file", default="", help="UTF-8 text file containing one WAV or MP3 path per line; avoids OS command-line length limits.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clip_seconds", type=float, default=1.0); parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--sample_rate", type=int, default=8000); parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8); parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--stop_signal_file", default="")
    args = parser.parse_args()
    if args.inputs_file:
        input_list = Path(args.inputs_file)
        if not input_list.is_file():
            parser.error(f"Input list file was not found: {input_list}")
        args.inputs.extend(line.strip() for line in input_list.read_text(encoding="utf-8").splitlines() if line.strip())
    if not args.inputs:
        parser.error("Provide --inputs or --inputs_file with at least one WAV or MP3 file.")
    sources = []
    source_details = []
    with tempfile.TemporaryDirectory(prefix="ddpm_audio_decode_") as temporary_dir:
        for index, input_path in enumerate(args.inputs):
            source, original_rate = read_audio(input_path, temporary_dir, index)
            source = resample(source, original_rate, args.sample_rate)
            sources.append(source)
            source_details.append({"path": str(Path(input_path).resolve()), "original_rate": original_rate, "seconds": round(len(source) / args.sample_rate, 2)})
    clip_len = max(256, int(args.clip_seconds * args.sample_rate))
    stride = max(1, int(clip_len * (1 - min(0.95, max(0, args.overlap)))))
    clips = []
    clip_source_ids = []
    for source_index, source in enumerate(sources):
        if len(source) < clip_len:
            source = np.pad(source, (0, clip_len - len(source)))
        file_clips = [source[start:start + clip_len] for start in range(0, len(source) - clip_len + 1, stride)]
        if (len(source) - clip_len) % stride:
            file_clips.append(source[-clip_len:])
        clips.extend(file_clips)
        clip_source_ids.extend([source_index] * len(file_clips))
    clips = np.stack(clips).astype(np.float32)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    progress(event="dataset_ready", files=len(sources), clips=len(clips), sample_rate=args.sample_rate, clip_seconds=args.clip_seconds, original_seconds=round(sum(len(source) for source in sources) / args.sample_rate, 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaveAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(TensorDataset(torch.from_numpy(clips).unsqueeze(1)), batch_size=max(1, args.batch_size), shuffle=True)
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for (batch,) in loader:
            batch = batch.to(device); noisy = (batch + torch.randn_like(batch) * 0.015).clamp(-1, 1)
            prediction = model(noisy)[..., :batch.shape[-1]]
            loss = (prediction - batch).abs().mean() + 0.15 * (torch.diff(prediction, dim=-1) - torch.diff(batch, dim=-1)).abs().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += loss.item()
        if epoch == 1 or epoch % max(1, args.epochs // 100) == 0 or epoch == args.epochs:
            elapsed = time.time() - start; eta = elapsed / epoch * (args.epochs - epoch)
            progress(event="epoch", epoch=epoch, total_epochs=args.epochs, loss=total / len(loader), eta_seconds=eta)
        if args.stop_signal_file and os.path.exists(args.stop_signal_file):
            progress(event="stopped"); break
    model.eval()
    with torch.no_grad():
        example = torch.from_numpy(clips[:1]).unsqueeze(1).to(device)
        reconstructed = model(example)[0, 0, :clip_len].cpu().numpy()
        # Keep a bounded bank of learned temporal representations. Generation
        # blends these learned latents instead of simply replaying clip zero.
        bank_count = min(64, len(clips))
        bank_indices = np.linspace(0, len(clips) - 1, bank_count, dtype=np.int64)
        latent_source_ids = torch.tensor(np.asarray(clip_source_ids)[bank_indices], dtype=torch.long)
        latent_batches = []
        for start_index in range(0, bank_count, max(1, args.batch_size)):
            selected = clips[bank_indices[start_index:start_index + max(1, args.batch_size)]]
            encoded = model.encoder(torch.from_numpy(selected).unsqueeze(1).to(device))
            latent_batches.append(encoded.cpu().to(torch.float16))
        latent_bank = torch.cat(latent_batches, dim=0)
    preview_path = output / "reconstruction_preview.wav"
    write_wav(preview_path, reconstructed, args.sample_rate)
    torch.save({
        "state_dict": model.state_dict(),
        "sample_rate": args.sample_rate,
        "clip_seconds": args.clip_seconds,
        "latent_bank": latent_bank,
        "latent_source_ids": latent_source_ids,
        "latent_bank_size": len(latent_bank),
        "architecture": "wave_autoencoder_v1",
    }, output / "sound_reconstruction.pt")
    (output / "sound_info.json").write_text(json.dumps({"sources": source_details, "clips": len(clips), "sample_rate": args.sample_rate, "clip_seconds": args.clip_seconds, "purpose": "waveform reconstruction baseline"}, indent=2))
    progress(event="done", preview_path=str(preview_path), output_dir=str(output))


if __name__ == "__main__":
    main()
