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

<img width="536" height="392" alt="Screenshot 2026-01-05 at 9 25 07 PM" src="https://github.com/user-attachments/assets/11af0bfd-ae7d-4fa5-a321-bcedb8cc0a86" />

* This histogram reveals a bimodal distribution, where image widths fall into two distinct groups rather than a random spread.

* Most images are clustered either in the 425–450 pixel range or in a larger group between 560–580 pixels.

<img width="693" height="390" alt="Screenshot 2026-01-05 at 9 25 19 PM" src="https://github.com/user-attachments/assets/4635671e-087c-4b93-8064-b69ef624cd3d" />

* This graph confirms that there is zero variation in image height, as every single image in the sample measures exactly 240 pixels tall.

* The single, isolated bar reaching a count of 500 demonstrates perfect uniformity in the vertical dimension, in sharp contrast to the variable widths seen earlier.

<img width="695" height="391" alt="Screenshot 2026-01-05 at 9 26 12 PM" src="https://github.com/user-attachments/assets/679227cb-bdb7-4db0-8cea-bb7d8852e8df" />

* This histogram represents the frame-to-frame pixel change, showing that most consecutive images have a relatively low mean absolute difference of roughly 0.1 to 0.2.

* The data follows a right-skewed distribution, indicating that while most frames are visually similar, there are occasional instances of significant changes or transitions.

<img width="694" height="390" alt="Screenshot 2026-01-05 at 9 26 24 PM" src="https://github.com/user-attachments/assets/8c38fd0b-e8e6-45d1-a1aa-8f85dade1558" />


* This histogram shows the average pixel change per story, illustrating how much movement or visual variety typically occurs within an entire sequence.

* The data is distributed fairly evenly between 0.1 and 0.25, suggesting that most stories maintain a consistent, moderate level of temporal activity.
  
<img width="685" height="388" alt="Screenshot 2026-01-05 at 9 26 44 PM" src="https://github.com/user-attachments/assets/382c64cf-61f4-4dc6-b59c-34aab9df2cf8" />


* This scatter plot correlates the length of a "story" (measured in frames) with its average temporal change, revealing that shorter stories exhibit a wider variance in visual activity levels.

* As the frame count increases, the average change tends to stabilize within a narrower range, suggesting that longer sequences converge toward a more consistent, moderate level of movement.
  
<img width="546" height="387" alt="Screenshot 2026-01-05 at 9 26 59 PM" src="https://github.com/user-attachments/assets/690e06ff-db42-4757-bd83-42a2b8b40be6" />


* This line graph tracks the average visual change over time, showing how much frames differ from one another as a "story" progresses through its first 20 transitions.

* The trend reveals that visual activity peaks early and fluctuates before significantly dropping off by the 20th transition, indicating that sequences often become more static or uniform toward the end.

<img width="706" height="388" alt="Screenshot 2026-01-05 at 9 27 14 PM" src="https://github.com/user-attachments/assets/8e75615a-9a08-4f8c-a22a-38799347aced" />
