# MeanFlow-MNIST (Optimized)

🚀 An optimized PyTorch implementation of **MeanFlow generative modeling** applied to **MNIST digits**.  
This project combines flow-matching training objectives with modern architectural and optimization improvements to achieve **stable training, high-quality samples, and efficient inference**.

---

## 🔑 Key Features

- **Flow Matching Training** ([Lipman et al., 2023](https://arxiv.org/abs/2210.02747)):  
  Learns a neural ODE that directly matches the true data velocity field.
- **Fourier Time Embeddings**:  
  Encodes scalar time `t` into high-dimensional features with sinusoidal frequencies for stronger conditioning.
- **UNet Backbone with Residual Blocks**:  
  Lightweight but expressive architecture tailored for MNIST (28×28), with skip connections and residual refinement.
- **Average Velocity Loss**:  
  Adds stability by matching the velocity at `t=0` to a Monte Carlo average over intermediate times.
- **Heun’s Method Sampler**:  
  A second-order ODE solver for better numerical stability and sharper generated samples.
- **Exponential Moving Average (EMA) Weights**:  
  Ensures smoother convergence and higher-quality samples at inference.
- **Cosine Learning Rate Scheduler + Checkpointing**:  
  Efficient training with automatic restarts and state saving (including EMA state).

---

## 📂 Project Structure

├── meanflow_mnist_optimized.py # Main training & sampling script
├── samples/ # Generated MNIST samples during training
├── checkpoints/ # Saved checkpoints (model + optimizer + EMA)
├── data/ # MNIST dataset (downloaded automatically)
└── README.md # Project documentation


---

## ⚙️ Requirements

- Python 3.9+
- PyTorch ≥ 2.0
- torchvision
- CUDA GPU recommended (for speed)

Install dependencies:

```bash
pip install torch torchvision
```
---

## 🏋️ Training

The training loop follows the **MeanFlow** paradigm, combining **flow matching** and **average velocity** constraints.

### Training Objective
The loss function has two parts:

1. **Flow Matching Loss**  
   Ensures the model learns the correct instantaneous velocity field:  
   \[
   \mathcal{L}_{FM} = \| v_\theta(z_t, t) - (x - z_t) \|^2
   \]

2. **Average Velocity Loss**  
   Matches the velocity at \( t = 0 \) to the average velocity across random time points:  
   \[
   \mathcal{L}_{AVG} = \| v_\theta(z_0, 0) - \bar{v} \|^2
   \]

Total training loss:  
\[
\mathcal{L} = \mathcal{L}_{FM} + \lambda \cdot \mathcal{L}_{AVG}
\]

---

### Training Setup
- **Dataset**: MNIST (handwritten digits, 28×28 grayscale).  
- **Optimizer**: AdamW with weight decay.  
- **Learning Rate**: Cosine annealing schedule.  
- **EMA (Exponential Moving Average)**: Maintains a smoothed copy of weights for more stable sampling.  
- **Checkpointing**: Saves both model and EMA weights every few steps.  

---

### Training Loop Outline
1. Sample a minibatch \( x \) from MNIST.  
2. Sample Gaussian noise \( z_0 \).  
3. Interpolate:  
   \[
   z_t = (1 - t) z_0 + t x
   \]
4. Compute **flow matching** and **average velocity** losses.  
5. Backpropagate, clip gradients, and update weights.  
6. Periodically:  
   - Save checkpoints.  
   - Generate sample digits for visual monitoring.  

---

### Running Training
```bash
python meanflow_mnist_optimized.py
```

---

## 📊 Model Overview

### Fourier Time Embedding
Encodes time *t* into a feature vector:

\[
\phi(t) = \text{MLP}\big[ \sin(\omega t), \cos(\omega t) \big]
\]

with logarithmically spaced frequencies.

---

### UNet + Residual Blocks
- **Downsampling path**: convolutional feature extraction.  
- **Bottleneck**: residual refinement with time conditioning.  
- **Upsampling path**: skip connections and reconstruction.  

---

### Loss Function
The training objective is a combination:

\[
\mathcal{L} = 
\underbrace{\| v_\theta(z_t, t) - (x - z_t) \|^2}_{\text{Flow Matching}}
+ \lambda \,
\underbrace{\| v_\theta(z_0, 0) - \bar{v} \|^2}_{\text{Average Velocity}}
\]

## 📈 Results

The model generates MNIST-like handwritten digits using the MeanFlow approach.  
After sufficient training (≈50k steps), digits become increasingly clear and structured.

### Qualitative Samples
Generated images show progressive improvement in digit quality as training advances:

- **Early training (~10k steps):** blurry, noisy shapes without clear digit structure.  
- **Mid training (~30k steps):** digits start forming but remain fuzzy.  
- **Later training (~50k steps):** sharp digits resembling MNIST samples, though with occasional artifacts.  

<p align="center">
  <img src="samples/mnist_digits.png" alt="Generated MNIST Digits" width="400">
</p>

### Key Observations
- Increasing the number of sampling steps (e.g., 50 → 100) improves sharpness.  
- EMA (Exponential Moving Average) weights further stabilize outputs.  
- Model capacity (base channels) directly influences clarity vs. memory usage.  

---
## 📚 References & Future Work

### References
This work builds on the foundations of **flow matching** and **MeanFlow** models:

- Lipman, Yaron, et al. *"Flow Matching for Generative Modeling."* (NeurIPS 2023)  
- Albergo, Michael S., et al. *"Building normalizing flows with stochastic interpolants."* (ICML 2023)  
- Official MeanFlow GitHub: [haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow)  

The training code and architecture here are adapted for the **MNIST** dataset, with practical modifications:
- Fourier time embeddings for richer temporal encoding.  
- UNet backbone with residual connections for improved capacity.  
- Heun’s method for more stable ODE integration.  
- Combined **flow matching** + **average-velocity** losses.  

---

### 🚀 Future Work
There are several directions to push this project further:

1. **Higher-Resolution Datasets**  
   Extend beyond MNIST (28×28) to datasets like **CIFAR-10** (32×32) or **CelebA** (64×64+).  
   This would test scalability and generalization.

2. **Stronger Architectures**  
   - Replace UNetSmall with a full UNet backbone or Transformer blocks.  
   - Explore attention mechanisms for structured generation.  

3. **Advanced Samplers**  
   - Investigate adaptive ODE solvers (e.g., RK45) instead of fixed Heun/Euler steps.  
   - Incorporate variance reduction for faster sampling.  

4. **Hybrid Training Objectives**  
   - Blend MeanFlow with diffusion-style noise prediction.  
   - Explore score distillation loss variants.  

5. **Evaluation Metrics**  
   - Add FID (Fréchet Inception Distance) or IS (Inception Score) to benchmark results.  
   - Compare with diffusion baselines on MNIST and CIFAR.  

---

📌 *This project currently demonstrates the viability of MeanFlow on MNIST. Future expansions aim to bridge the gap between academic research and practical large-scale generative modeling.*  
