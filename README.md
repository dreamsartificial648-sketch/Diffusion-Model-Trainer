# Unconditional Diffusion Styler

Generate images with a single button—no text prompts. The model learns from your consistent image dataset and produces new samples in that style.

## Requirements

- Python 3.8+
- NVIDIA GPU with 6GB+ VRAM (GTX 1660 SUPER supported)
- CUDA (for GPU training and inference)

## Setup

1. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure Accelerate for training:

   ```bash
   accelerate config
   ```

   Use the default settings, or choose your GPU when prompted.

## Dataset

Place your training images in `data/train/`:

```
data/
  train/
    image1.png
    image2.jpg
    ...
```

- Use 20–100+ images for best results
- Images should be consistent in style/subject
- Supported formats: PNG, JPG, JPEG

## Training

Run training with 6GB VRAM–friendly settings:

```bash
accelerate launch train.py --train_data_dir data --output_dir output/model --resolution 64 --train_batch_size 1 --mixed_precision fp16
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_data_dir` | — | Path to folder containing `train/` subfolder with images |
| `--output_dir` | `output/model` | Where to save the trained model |
| `--resolution` | `64` | Image size (64 for 6GB VRAM; 128 if memory allows) |
| `--train_batch_size` | `1` | Batch size per device |
| `--gradient_accumulation_steps` | `4` | Simulates larger batch |
| `--mixed_precision` | `fp16` | Reduces VRAM usage |
| `--num_epochs` | `100` | Training epochs |

Training may take 1–4 hours depending on dataset size and hardware.

## Run the App

```bash
python app.py
```

- **Generate** – Creates a new image (no text prompt)
- **Seed** – Optional. Enter a number for reproducible results
- The app uses your trained model from `output/model` if it exists; otherwise it falls back to a pre-trained butterflies model for testing

## Project Structure

```
StabbleDiffusionStyler/
├── data/
│   └── train/           # Your training images
├── output/
│   └── model/          # Trained model (created after training)
├── train.py             # Training script
├── app.py               # Desktop GUI
├── requirements.txt
└── README.md
```

## Troubleshooting

## Sound reconstruction test

The **Sound** tab is a deliberately small, short-sound baseline. Pick one or
more WAV or MP3 files and it will split them into overlapping clips of your
chosen length (0.1-10 seconds),
clips, train a compact waveform reconstruction model, and save a playable
`reconstruction_preview.wav`.

- Start with one WAV or MP3 and the default 1-second / 8 kHz settings.
- A single file is useful for this test: the goal is to intentionally overfit
  and confirm the saved preview is recognizable rather than static.
- MP3s are decoded into temporary WAV files during the run and the temporary
  copies are removed automatically. Your original MP3s are never changed.
- Reconstruction Preview verifies audio loading, clipping, GPU learning, and
  WAV decoding. Generation Mode then blends learned clip representations; it
  is aimed at short textures rather than full song composition.

After training, **Generate New Sound** blends the learned representations of
multiple training clips and decodes them into a new WAV. You can choose the
output duration, number of clips blended per section, variation, and seed.
Older sound models must be retrained once so they contain the learned latent
bank used by Generation Mode.

**Out of memory during training**

- Use `--resolution 32`
- Ensure `--train_batch_size 1` and `--mixed_precision fp16`

**Out of memory during inference**

- The app enables attention slicing automatically
- Use a smaller pre-trained model or your 64x64 trained model

**No trained model yet**

- The app will use `anton-l/ddpm-butterflies-128` so you can test the UI immediately
- Train on your own images, then the app will automatically use `output/model`
