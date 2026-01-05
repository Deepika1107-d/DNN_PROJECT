# Diffusion-Based Next-Frame Prediction for Visual Stories

Purpose
- Generate sharper, context-aware next frames by conditioning a diffusion model on the previous K frames.

Method overview
- Dataset: StoryReasoning (image sequences).
- Baseline: CNN frame encoder + GRU over context + CNN decoder.
- Diffusion: conditional UNet-lite denoiser with a CNN+GRU context encoder and sinusoidal time embeddings.
- Metrics: L1, MSE, PSNR, and temporal consistency (TC2) against the last context frame.

Repository layout
- `src/dataloader.py`: indexing and temporal image dataset with preprocessing.
- `src/model_baseline_cnn.py`: GRU baseline + metrics helpers.
- `src/context_encoder.py`: CNN+GRU context encoder.
- `src/diffusion_model_wrapper.py`: diffusion schedule, UNet-lite, and conditional model.
- `src/model_diffusion.py`: DDPM/DDIM/refine sampling helpers.
- `src/train_diffusion.py`: training for baseline or diffusion.
- `src/eval_diffusion.py`: evaluation utilities for baseline and diffusion sampling.

Quickstart
```bash
cd "Deepika - Project 2"

# Baseline training
python src/train_diffusion.py --mode baseline --epochs 2

# Diffusion training
python src/train_diffusion.py --mode diffusion --epochs 3 --t_steps 100
```

Notes
- Images are normalized to [-1, 1] at 128x128 by default.
- Diffusion sampling is slower; use small subsets for quick evaluation.
<img width="698" height="387" alt="Screenshot 2026-01-05 at 9 14 11 PM" src="https://github.com/user-attachments/assets/6c534642-216b-47ca-804c-505324c01dc1" />
<img width="697" height="390" alt="Screenshot 2026-01-05 at 9 24 42 PM" src="https://github.com/user-attachments/assets/5235ff78-2693-4f29-a117-3ddcdcdaa920" />
<img width="540" height="369" alt="Screenshot 2026-01-05 at 9 24 54 PM" src="https://github.com/user-attachments/assets/d97eef7b-b22d-4409-adc0-32c0f1b69300" />
<img width="536" height="392" alt="Screenshot 2026-01-05 at 9 25 07 PM" src="https://github.com/user-attachments/assets/11af0bfd-ae7d-4fa5-a321-bcedb8cc0a86" />
<img width="693" height="390" alt="Screenshot 2026-01-05 at 9 25 19 PM" src="https://github.com/user-attachments/assets/4635671e-087c-4b93-8064-b69ef624cd3d" />
