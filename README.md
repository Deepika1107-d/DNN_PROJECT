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

* The graph shows the distribution of training samples across different **frame counts**, with most samples clustered around **5–7** and **16–18 frames**.
* This indicates an **imbalanced distribution**, where certain frame counts are much more frequent than others.

<img width="698" height="387" alt="Screenshot 2026-01-05 at 9 14 11 PM" src="https://github.com/user-attachments/assets/6c534642-216b-47ca-804c-505324c01dc1" />

* The graph shows the distribution of **frame counts in the test set**, with most samples concentrated around **5–6** and **16–18 frames**.
* Similar to training data, the test set is **imbalanced**, but with overall lower sample counts.

<img width="697" height="390" alt="Screenshot 2026-01-05 at 9 24 42 PM" src="https://github.com/user-attachments/assets/5235ff78-2693-4f29-a117-3ddcdcdaa920" />

* The boxplot shows that **train and test sets have similar frame_count distributions**, with a median around **13 frames**.
* Both sets exhibit a comparable spread and range (≈5 to 22), indicating **consistent data distribution** between training and testing.

<img width="540" height="369" alt="Screenshot 2026-01-05 at 9 24 54 PM" src="https://github.com/user-attachments/assets/d97eef7b-b22d-4409-adc0-32c0f1b69300" />

* This graph displays the dimensions of sampled image frames, revealing that every image has a constant **height of 240 pixels**.

* In contrast, the **width varies** significantly across the samples, ranging from approximately 390 to 580 pixels.

Would you like to explore why the widths vary while the height stays fixed (e.g., aspect ratio preservation)?
<img width="536" height="392" alt="Screenshot 2026-01-05 at 9 25 07 PM" src="https://github.com/user-attachments/assets/11af0bfd-ae7d-4fa5-a321-bcedb8cc0a86" />
<img width="693" height="390" alt="Screenshot 2026-01-05 at 9 25 19 PM" src="https://github.com/user-attachments/assets/4635671e-087c-4b93-8064-b69ef624cd3d" />
<img width="695" height="391" alt="Screenshot 2026-01-05 at 9 26 12 PM" src="https://github.com/user-attachments/assets/679227cb-bdb7-4db0-8cea-bb7d8852e8df" />
<img width="694" height="390" alt="Screenshot 2026-01-05 at 9 26 24 PM" src="https://github.com/user-attachments/assets/8c38fd0b-e8e6-45d1-a1aa-8f85dade1558" />
<img width="685" height="388" alt="Screenshot 2026-01-05 at 9 26 44 PM" src="https://github.com/user-attachments/assets/382c64cf-61f4-4dc6-b59c-34aab9df2cf8" />
<img width="546" height="387" alt="Screenshot 2026-01-05 at 9 26 59 PM" src="https://github.com/user-attachments/assets/690e06ff-db42-4757-bd83-42a2b8b40be6" />
<img width="706" height="388" alt="Screenshot 2026-01-05 at 9 27 14 PM" src="https://github.com/user-attachments/assets/8e75615a-9a08-4f8c-a22a-38799347aced" />
