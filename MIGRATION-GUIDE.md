# Migration Guide: Evolution of the Swarm System

## 1. Environment Variables Migration
*See the previous section for details on migrating from hardcoded URLs to `.env` based configuration.*

---

## 2. AI Engine Migration: WebTorch to js-pytorch

### What Changed
The system has evolved from a simulated training environment to a real, hardware-accelerated generative AI trainer.

- **Engine**: Switched from WebTorch (simulated) to **js-pytorch 0.7.2** (real).
- **Architecture**: Moved from simple MLPs to **MLP-Mixer** spatial architectures.
- **Resolution**: Optimized for **32x32** images (3072 features) to fit browser memory.
- **Training**: Implemented **real backpropagation** and Adam optimization.
- **Inference**: Implemented **real Schrödinger Bridge SDE sampling** (Reverse Drift).

### Migration Steps for Developers

#### Step 1: Update Dependencies
Ensure you have the latest `js-pytorch` package:
```bash
npm install js-pytorch@0.7.2
```

#### Step 2: Clear Local Database
Since the model architecture and input dimensions have changed, old checkpoints in IndexedDB will be incompatible.
1. Open the Training Dashboard.
2. Click **"Clear DB"** in the Local Database panel.
3. Refresh the page to initialize the new MLP-Mixer weights.

#### Step 3: Enable Hardware Acceleration
For optimal performance, ensure WebGL is enabled in your browser:
- **Chrome**: `chrome://settings/system` -> "Use graphics acceleration when available".
- **Firefox**: `about:config` -> `webgl.force-enabled = true`.

### New Model Specifications

| Feature | Old (Simulated) | New (Real) |
|---------|-----------------|------------|
| Architecture | Flat MLP | **MLP-Mixer (Patch-based)** |
| Image Size | 64x64 (Simulated) | **32x32 (Real)** |
| Latent Space | 8-dim | **64-dim** |
| Patch Size | N/A | **4x4 pixels** |
| Optimization | None | **Adam (LR: 0.0002)** |

## 3. Architecture Benefits

### Why MLP-Mixer?
`js-pytorch` does not currently support `Conv2d` layers reliably across all backends. To maintain high image quality, we implemented the **MLP-Mixer** architecture:
1.  **Spatial Awareness**: By dividing images into patches, the model learns local features.
2.  **Mixing Layers**: Token-mixing allows spatial communication without convolutions.
3.  **Stability**: Provides CNN-like performance using only the stable `Linear` primitives of the library.

## Troubleshooting AI Initialization

### Issue: "TypeError: nn.Conv2d is not a constructor"
**Cause**: Using an outdated model definition with convolutions not supported by the current runtime.
**Solution**: Ensure `src/torchjs/models.js` is using the `MixerBlock` implementation.

### Issue: Out of Memory (OOM) in Node.js Tests
**Cause**: Training on large batches or high resolutions in the Node.js heap.
**Solution**: The system is now default to 32x32. If running tests manually, use `node --max-old-space-size=4096 test-torchjs.js`.

### Issue: "Attempting to reshape into invalid shape"
**Cause**: Dimension mismatch in the data pipeline.
**Solution**: Verify that your input data is flattened to 3072 features (for 32x32 RGB). The `EnhancedSwarmTrainer` handles this automatically for new imports.
