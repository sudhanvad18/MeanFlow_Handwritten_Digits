# Implementation notes

## Data and model

Training uses MNIST's training split, normalized from `[0, 1]` to `[-1, 1]`. Labels are discarded: this is unconditional generation. The network receives a noisy image and an embedding of scalar time. Its U-Net-style backbone downsamples once from 28 to 14 pixels, uses residual convolutional blocks, and concatenates an early feature map during upsampling.

## Relationship to Flow Matching and MeanFlow

The current experiment constructs a straight interpolation:

```text
z_t = (1 - t) z_0 + t x
```

It regresses the network output against `x - z_t` and adds a penalty comparing the output at `(z_0, 0)` to the average of four network predictions at random interpolated times. Gradients propagate through both sides of that average-velocity penalty. Its weight is `LAMBDA_AVG = 2.0`.

This local target differs from the derivative `x - z_0` of the displayed straight interpolation. The network also accepts only one time value; the published MeanFlow formulation models an average velocity over an interval with two endpoints and a corresponding training identity. This implementation should therefore be evaluated as a separate exploratory objective, not a faithful MeanFlow or standard straight-path Flow Matching reproduction. See [Flow Matching](https://arxiv.org/abs/2210.02747) and [MeanFlow](https://arxiv.org/abs/2505.13447).

Sampling starts from Gaussian noise and integrates the learned field from time zero to one with Heun's predictor-corrector update. With `S` steps, sampling performs `2S` network evaluations. It does not implement one-step generation.

## Configuration

| Setting | Current default |
| --- | --- |
| Training batch size | 128 |
| Learning rate | `1e-4` |
| Maximum step count | 100,000 |
| U-Net base channels | 128 |
| Time embedding dimension | 128 |
| Average-velocity samples | 4 |
| Training-time sampling steps | 100 |
| Generation script sampling steps | 200 |
| Training sample/checkpoint interval | 1,000 steps |
| Optimizer | AdamW, weight decay `1e-6` |
| Gradient clipping | Norm 1.0, model plus time embedding |
| Learning-rate schedule | Cosine decay |
| Device | CUDA if available; otherwise CPU |

Training parameters are Python constants and constructor arguments, not command-line flags. The generation script reads `N_SAMPLES`, `SAMPLE_STEPS`, and `OUTPUT_FILE` from environment variables. Use positive counts. The source does not seed the random generators.

## Checkpoints and reproducibility

`checkpoints/meanflow_mnist.pth` stores model weights, time-embedding weights, optimizer state, optional EMA weights, and a step counter. The inspected checkpoint has step 120,000 and no EMA weights. It loads into the current model architecture, but its historical training configuration is not recorded and its step count exceeds the current training limit.

The current training path sets `ema_model = None` and never performs EMA updates. The loader can use an EMA state produced elsewhere, but EMA improvement should not be claimed for the supplied result. Restarting training restores the step counter and optimizer; it does not restore the scheduler state or random-number states, so a resumed run is not an exact continuation.

The committed sample names include intervals and steps that differ from the current constants. Treat them as historical artifacts rather than proof that the current configuration produced those exact outputs. The curated README images are byte-for-byte copies of `samples/sample_000000.png` and `samples/sample_120000.png` from the inspected checkout.

## Remaining evaluation work

- Check the objective against the intended modeling formulation before further training.
- Record the full configuration, seed, software versions, and hardware for future runs.
- Establish a baseline, then compare quality and compute using the same protocol.
- Save scheduler and random-number state if exact resume is a requirement.

No numerical quality, speed, or efficiency claim is established by the current qualitative grids.
