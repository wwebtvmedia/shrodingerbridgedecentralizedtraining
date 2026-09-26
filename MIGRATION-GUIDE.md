# Migration Guide: Evolution of the Swarm System

> [!NOTE]
> This guide details the transition from the legacy `js-pytorch` (MLP-Mixer) architecture to the current **TensorFlow.js** (96x96 CNN) system.

---

## 🚦 Migration Status

- [x] **Phase 1**: Dependency update (TFJS)
- [x] **Phase 2**: CNN Architecture implementation
- [x] **Phase 3**: LoRA Integration
- [x] **Phase 4**: 96x96 Resolution Support
- [x] **Phase 5**: WebGPU Optimization & Multi-Phase SB Alignment

---

## 1. Environment Variables Migration

Legacy hardcoded URLs have been replaced with `.env` based configuration. Ensure your `.env` file is populated based on `.env.template`.

---

## 2. AI Engine Migration: js-pytorch to TensorFlow.js (96x96 CNN & WebGPU)

### What Changed

The system has evolved from an MLP-Mixer architecture to a state-of-the-art, high-resolution CNN trainer aligned with `enhancedoptimaltransport`.

- **Engine**: Switched to **TensorFlow.js WebGPU** hardware acceleration (with WASM fallback) for maximum performance and cross-platform shader compute.
- **Architecture**: Moved from MLP-Mixer to **CNN Residual** architectures with **Axial Attention** and Group Normalization.
- **Resolution**: Upgraded from 32x32 to **96x96** images (27,648 features).
- **Training**: Multi-phase Schrödinger Bridge optimization (VAE, Drift with CFG dropout, Joint fine-tuning, variance floor, cosine LR decay).
- **Inference**: High-fidelity Schrödinger Bridge sampling with **Heun**, **RK4**, and **Euler** ODE solvers + Langevin refinement.

### 📝 Quick Checklist for Developers

- [ ] Run `npm install`
- [ ] Clear Local IndexedDB (Models are incompatible)
- [ ] Use a WebGPU-enabled browser (Chrome, Edge, Firefox Nightly, Safari 18+)
- [ ] Update `.env` with new consolidation server URLs

### 📊 Comparative Analysis

| Feature          | Legacy (js-pytorch) | Current (TensorFlow.js / WebGPU) |
| :--------------- | :------------------ | :------------------------------- |
| **Architecture** | MLP-Mixer           | **CNN Residual + Attention**     |
| **Acceleration** | CPU / WebGL 1.0     | **WebGPU (Hardware Accelerated)**|
| **Image Size**   | 32x32               | **96x96**                        |
| **Latent Space** | 64-dim (Flat)       | **12x12x8 (4D Tensor)**          |
| **Resolution**   | 1,024 pixels        | **9,216 pixels**                 |
| **Optimization** | Adam                | **AdamW + Cosine LR Schedule**   |
| **Adaptation**   | None                | **LoRA (Rank: 8, Alpha: 16)**    |
| **ODE Solvers**  | Simple Step         | **Heun / RK4 / Langevin**        |

---

## 3. Training Paradigm: Three-Phase Evolution (Aligned with enhancedoptimaltransport)

The system follows a synchronized Three-Phase training schedule:

1.  **Phase 1: VAE Optimization**: Focuses on learning the latent manifold, reconstruction (L1 + SSIM + Edge + TV), KL annealing, and channel diversity.
2.  **Phase 2: Drift Learning & Anchor Tuning**: Freezes the VAE decoder; trains the U-Net drift network with time-weighted Huber loss & CFG dropout, while regularizing the VAE encoder with consistency, diversity, and global variance floor against `vae_ref`.
3.  **Phase 3: Joint Refinement**: Co-optimizes Drift network and VAE with balanced reconstruction and consistency loss scales.

## 4. Efficiency with LoRA (Low-Rank Adaptation)

To minimize the bandwidth required for P2P synchronization, we have implemented LoRA:

- **Freezing Base Weights**: The massive 96x96 CNN weights are frozen during swarm sync.
- **Trainable Adapters**: Only the low-rank matrices (A and B) are updated and shared.
- **Bandwidth Reduction**: Synchronization is now **~10x faster** compared to full model weight sharing.

---

## 5. Adaptive Logic & SARSA Optimization

The system now features an **Adaptive Co-Training (ACT)** layer powered by **SARSA (State-Action-Reward-State-Action)** reinforcement learning.

- **Dynamic Task Selection**: The swarm now automatically decides whether to focus on **VAE** (manifold), **Drift** (trajectory), or **Both** based on which action provides the highest "Reward" (Loss Improvement per Millisecond).
- **Hardware Profile Adaptation**: Weaker nodes (like Raspberry Pi 3) will naturally learn to prioritize VAE training if the Joint training consumes too much time or causes instability.
- **Trajectory Advantage Estimation**: Using the **TrajectoryAdvantageEstimator**, the system now identifies high-impact layer updates, allowing for selective gossip and reduced bandwidth usage.

### How it works:

1.  **Observe**: Current phase and loss magnitude.
2.  **Act**: Select a training task (VAE, Drift, or Both).
3.  **Reward**: Measure `Delta_Loss / Training_Time`.
4.  **Update**: Adjust the Q-table to favor efficient training paths in the future.

---

## ⚠️ Troubleshooting AI Initialization

### Issue: "ValueError: The first layer in a Sequential model must get an inputShape"

- **Cause**: Incorrect layer initialization.
- **Solution**: Ensure `inputShape` is provided to the first layer (e.g., `inputShape: [null, null, channels]`).

### Issue: Out of Memory (OOM)

- **Cause**: 96x96 CNNs require significant VRAM.
- **Solution**: Reduce `BATCH_SIZE` in `src/config.js` or close other browser tabs.

### Issue: "Cannot compute gradient: DepthToSpace"

- **Cause**: Lack of gradient support in some TFJS backends.
- **Solution**: We now use `UpSampling2d + Conv2d` for improved compatibility.
