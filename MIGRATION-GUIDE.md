# Migration Guide: Evolution of the Swarm System

## 1. Environment Variables Migration

_See the previous section for details on migrating from hardcoded URLs to `.env` based configuration._

---

## 2. AI Engine Migration: js-pytorch to TensorFlow.js (96x96 CNN)

### What Changed

The system has evolved from an MLP-Mixer architecture to a state-of-the-art, high-resolution CNN trainer.

- **Engine**: Switched from js-pytorch to **TensorFlow.js** (TFJS) for better performance and GPU utilization.
- **Architecture**: Moved from MLP-Mixer to **CNN Residual** architectures with **Axial Attention**.
- **Resolution**: Upgraded from 32x32 to **96x96** images (27,648 features).
- **Training**: Improved **U-Net** based drift networks for superior generative quality.
- **Inference**: High-fidelity Schrödinger Bridge sampling with iterative Euler updates.

### Migration Steps for Developers

#### Step 1: Update Dependencies

Ensure you have the latest TensorFlow.js packages:

```bash
npm install @tensorflow/tfjs
```

#### Step 2: Clear Local Database

Since the model architecture and resolution (96x96) have changed significantly, old checkpoints in IndexedDB/LocalStorage are **incompatible**.

1. Open the Training Dashboard.
2. Click **"Clear DB"** or **"Cleanup"** in the Local Database panel.
3. Refresh the page to initialize the new CNN weights.

#### Step 3: Enable Hardware Acceleration

For 96x96 resolution, hardware acceleration is **mandatory**:

- **Chrome**: `chrome://settings/system` -> "Use graphics acceleration when available".
- **WebGL/WebGPU**: TFJS will automatically detect the best available backend.

### New Model Specifications

| Feature      | Old (js-pytorch) | New (TensorFlow.js)           |
| ------------ | ---------------- | ----------------------------- |
| Architecture | MLP-Mixer        | **CNN Residual + Attention**  |
| Image Size   | 32x32            | **96x96**                     |
| Latent Space | 64-dim (Flat)    | **12x12x8 (4D Tensor)**       |
| Resolution   | 1,024 pixels     | **9,216 pixels**              |
| Optimization | Adam             | **Adam (LR: 2e-4)**           |
| Adaptation   | N/A              | **LoRA (Rank: 8, Alpha: 16)** |

## 3. Training Paradigm: Three-Phase Evolution

The system now follows a structured Three-Phase training schedule to ensure stable convergence:

1.  **Phase 1: VAE Optimization**: Focuses on learning the latent manifold and reconstruction.
2.  **Phase 2: Drift Learning**: Freezes the VAE and trains the U-Net drift network to model the Schrödinger Bridge.
3.  **Phase 3: Joint Refinement**: Co-optimizes both networks with adaptive loss weighting for maximum fidelity.

## 4. Efficiency with LoRA (Low-Rank Adaptation)

To minimize the bandwidth required for P2P synchronization, we have implemented LoRA:

- **Freezing Base Weights**: The massive 96x96 CNN weights are frozen during swarm sync.
- **Trainable Adapters**: Only the low-rank matrices (A and B) are updated and shared between peers.
- **Bandwidth Reduction**: Synchronization is now ~10x faster compared to sharing full model weights.

## 5. Architecture Benefits

### Why CNN Residual + Axial Attention?

While MLP-Mixer provided a workaround for missing convolutions, TensorFlow.js supports full convolutional pipelines:

1.  **Inductive Bias**: Convolutions are naturally suited for 96x96 image data.
2.  **Residual Learning**: Skip connections allow for much deeper and more stable networks.
3.  **Axial Attention**: Spatial Split Attention enables long-range dependencies without the $O(N^2)$ cost of full self-attention.
4.  **U-Net Drift**: The U-Net structure in the drift network provides precise multi-scale control over noise transformation.

## Troubleshooting AI Initialization

### Issue: "ValueError: The first layer in a Sequential model must get an inputShape"

**Cause**: Incorrect layer initialization in TFJS Sequential models.
**Solution**: Ensure `inputShape` is provided to the first layer (e.g., `inputShape: [null, null, channels]`).

### Issue: Out of Memory (OOM) on RPi or Low-End Devices

**Cause**: 96x96 CNNs require significant VRAM/RAM.
**Solution**: Reduce `BATCH_SIZE` in `src/config.js` or ensure only one browser tab is active.

### Issue: "Cannot compute gradient: gradient function not found for DepthToSpace"

**Cause**: Some TFJS operations lack gradient support on certain backends.
**Solution**: The system now uses `UpSampling2d + Conv2d` instead of `DepthToSpace` for improved compatibility.
