"""Generate short audio by blending learned autoencoder latents."""
import argparse
import json
import math
import time
import wave
from pathlib import Path

import numpy as np
import torch

from audio_train import WaveAutoencoder


def progress(**values):
    print("AUDIO_PROGRESS_JSON:" + json.dumps({"ts": time.time(), **values}), flush=True)


def write_wav(path, samples, rate):
    samples = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak > 0.98:
        samples = samples * (0.98 / peak)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def blend_latents(bank, source_ids, count, variation, generator):
    available = bank.shape[0]
    count = max(1, min(int(count), available))
    unique_sources = torch.unique(source_ids)
    source_order = unique_sources[torch.randperm(len(unique_sources), generator=generator)]
    selected = []
    for source_id in source_order[:count]:
        candidates = torch.nonzero(source_ids == source_id, as_tuple=False).flatten()
        pick = candidates[torch.randint(len(candidates), (1,), generator=generator)].item()
        selected.append(pick)
    if len(selected) < count:
        remaining = [index for index in torch.randperm(available, generator=generator).tolist() if index not in selected]
        selected.extend(remaining[:count - len(selected)])
    indices = torch.tensor(selected, dtype=torch.long)
    weights = torch.rand(count, generator=generator)
    weights = weights / weights.sum().clamp_min(1e-8)
    latent = (bank[indices].float() * weights[:, None, None]).sum(dim=0, keepdim=True)
    if variation > 0:
        scale = bank.float().std().clamp_min(1e-5)
        noise = torch.randn(latent.shape, generator=generator) * scale * variation
        latent = latent + noise
    return latent, indices.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--blend_count", type=int, default=3)
    parser.add_argument("--variation", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / "sound_reconstruction.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(f"No trained sound model found in {model_dir}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    bank = checkpoint.get("latent_bank")
    if bank is None or len(bank) == 0:
        raise RuntimeError("This sound model predates Generation Mode. Retrain it once to create learned generation latents.")
    source_ids = checkpoint.get("latent_source_ids")
    if source_ids is None or len(source_ids) != len(bank):
        source_ids = torch.zeros(len(bank), dtype=torch.long)

    sample_rate = int(checkpoint["sample_rate"])
    clip_seconds = float(checkpoint["clip_seconds"])
    clip_samples = max(1, round(sample_rate * clip_seconds))
    duration = max(0.1, min(60.0, float(args.duration)))
    target_samples = max(1, round(sample_rate * duration))
    seed = int(args.seed) if args.seed else int(time.time_ns() % 2_147_483_647)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaveAutoencoder().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    crossfade = min(max(1, int(sample_rate * 0.03)), clip_samples // 4)
    step = max(1, clip_samples - crossfade)
    chunk_count = max(1, math.ceil(max(1, target_samples - crossfade) / step))
    output = np.zeros(step * (chunk_count - 1) + clip_samples, dtype=np.float32)
    weight = np.zeros_like(output)
    chosen = []
    with torch.no_grad():
        for chunk_index in range(chunk_count):
            latent, indices = blend_latents(bank, source_ids, args.blend_count, max(0.0, min(0.5, args.variation)), generator)
            decoded = model.decoder(latent.to(device))[0, 0, :clip_samples].cpu().numpy()
            start = chunk_index * step
            envelope = np.ones(clip_samples, dtype=np.float32)
            if chunk_index > 0:
                envelope[:crossfade] = np.linspace(0, 1, crossfade, dtype=np.float32)
            if chunk_index < chunk_count - 1:
                envelope[-crossfade:] = np.minimum(envelope[-crossfade:], np.linspace(1, 0, crossfade, dtype=np.float32))
            output[start:start + clip_samples] += decoded * envelope
            weight[start:start + clip_samples] += envelope
            chosen.append(indices)
            progress(event="generation_step", step=chunk_index + 1, total_steps=chunk_count)
    output = output[:target_samples] / np.maximum(weight[:target_samples], 1e-6)
    generations = model_dir / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    path = generations / f"generated_seed_{seed}_{int(time.time())}.wav"
    write_wav(path, output, sample_rate)
    progress(event="generated", preview_path=str(path), seed=seed, duration=duration, blend_indices=chosen)


if __name__ == "__main__":
    main()
