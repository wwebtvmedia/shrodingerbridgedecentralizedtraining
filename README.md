# Swarm Schrödinger Bridge Training System

## 📖 Abstract

A decentralized, self-organizing swarm training system for Schrödinger Bridge models using **TensorFlow.js** and peer-to-peer networking. This system implements a novel evolutionary optimization approach where multiple browser clients collaboratively train generative models without any central coordinator, using gossip protocols for model synchronization and adaptive phase management. The architecture is now aligned with the latest **CNN-based** PyTorch models (96x96 resolution).

## 🧮 Mathematical Foundations

### Schrödinger Bridge Formulation

The Schrödinger Bridge problem seeks the most likely stochastic process connecting two probability distributions \( p_0 \) and \( p_1 \) over a time interval \([0, T]\). Given:

- **Source distribution**: \( p_0(x) \) (e.g., Gaussian noise)
- **Target distribution**: \( p_1(x) \) (e.g., data distribution)
- **Reference process**: \( dX_t = f(X_t, t)dt + \sigma(t)dW_t \)

The Schrödinger Bridge finds the optimal drift \( u^\*(x, t) \) that minimizes:

\[
\mathbb{E}\left[\int_0^T \frac{1}{2} \|u(X_t, t)\|^2 dt\right]
\]

subject to \( X_0 \sim p_0 \) and \( X_T \sim p_1 \).

### Three-Phase Training Architecture

#### Phase 1: Variational Autoencoder (VAE) Training
**Objective**: Learn robust latent representations and reconstruction capabilities using a **CNN Residual** architecture at 96x96 resolution.

#### Phase 2: Drift Network Training
**Objective**: Learn the optimal drift function \( u(x, t) \) using a **U-Net** based trajectory modeling, keeping the VAE manifold fixed.

#### Phase 3: Joint Training (Both)
**Objective**: Combined loss with adaptive weighting for end-to-end refinement and high-fidelity detail generation.

### Low-Rank Adaptation (LoRA)
To enable efficient decentralized training, the system incorporates LoRA for both Dense and Conv2D layers:
- **Rank-8 Adapters**: Significantly reduces trainable parameters.
- **Fast Synchronization**: Only LoRA weights are exchanged during peer updates, minimizing network overhead.
- **Base Model Preservation**: Allows for fine-tuning large pre-trained weights within the browser environment.

## 🏗️ System Architecture

### CNN Residual Architecture (96x96)

To achieve high-quality generation, the system utilizes a modern convolutional architecture:

1.  **Residual Blocks**: Deep feature extraction with skip connections for stable gradient flow.
2.  **Spatial Split Attention**: Axial attention mechanism for long-range spatial dependencies.
3.  **Subpixel Upsampling**: High-fidelity image reconstruction (96x96) using learnable upsampling.
4.  **Label Conditioning**: FiLM-based modulation for class-conditional generation (10 real classes + 1 NULL class).

This provides the same inductive biases as state-of-the-art PyTorch models, now optimized for browser execution via **TensorFlow.js**.

### Core Components

#### 1. **SwarmTrainer** (`src/core/trainer.js`)
- Manages local training loop with evolutionary optimization.
- Handles 96x96 image data and 4D tensor ([B, H, W, C]) pipelines.

#### 2. **ModelManager** (`src/core/models.js`)
- Manages **TensorFlow.js** models and state.
- Supports 96x96 image input with 12x12x8 latent space (8 channels).

#### 3. **TorchJSTrainer** (`src/torchjs/integration.js`)
- Provides hardware-accelerated training using WebGL/WebGPU via **TensorFlow.js**.
- Maps P2P swarm logic to real CNN operations and backpropagation.

#### 4. **InferenceEngine** (`src/utils/inference.js`)
- Implements real Schrödinger Bridge sampling (Reverse SDE).
- Uses iterative drift updates to transform noise into coherent 96x96 images.

## 🚀 Implementation Details

### TensorFlow.js Integration

The system uses **TensorFlow.js** for browser-based execution:

```javascript
// CNN-based VAE (simplified)
export class LabelConditionedVAE {
  constructor() {
    this.encBlocks = [
      new ResidualBlock(16, 32, 2),
      new LabelConditionedBlock(32, 32),
      new SpatialSplitAttention(64, 4)
    ];
  }

  forward(x, labels) {
    // Real backprop-capable forward pass
  }
}
```

## 🛠️ Technical Specifications

### System Requirements

- **Browser**: Chrome 80+, Firefox 75+, Safari 14+, Edge 80+
- **Hardware Acceleration**: WebGL or WebGPU enabled for TensorFlow.js speed.
- **Memory**: Minimum 4GB RAM (recommended for 96x96 models).
- **Storage**: IndexedDB for training data persistence.

### Dependencies

```json
{
  "@tensorflow/tfjs": "Hardware-accelerated deep learning in the browser",
  "simple-peer": "WebRTC wrapper for P2P",
  "chart.js": "Real-time visualization",
  "express": "Consolidation server"
}
```

## 🚀 Getting Started

### Installation

```bash
cd prototype
npm install
```

### Development

```bash
npm run dev
```

Open http://localhost:3000 in multiple browser windows to start real swarm training.

---

**Note**: This system implements **real model training and inference** with a modern CNN architecture. It performs actual gradient descent and SDE-based sampling at 96x96 resolution, mirroring the latest PyTorch implementations.

_Last updated: April 2026_
