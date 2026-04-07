# Swarm Schrödinger Bridge Training System

## 📖 Abstract

A decentralized, self-organizing swarm training system for Schrödinger Bridge models using **js-pytorch** and peer-to-peer networking. This system implements a novel evolutionary optimization approach where multiple browser clients collaboratively train generative models without any central coordinator, using gossip protocols for model synchronization and adaptive phase management.

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

**Objective**: Learn latent representations and reconstruction capabilities using **MLP-Mixer** architecture.

#### Phase 2: Drift Network Training

**Objective**: Learn the optimal drift function \( u(x, t) \) using **MLP-Mixer** based trajectory modeling.

#### Phase 3: Joint Training (Both)

**Objective**: Combined loss with adaptive weighting.

## 🏗️ System Architecture

### MLP-Mixer Architecture

To achieve high-quality generation without convolutional layers (unsupported in some web environments), the system utilizes an **MLP-Mixer** architecture:

1.  **Patchification**: Images (32x32) are divided into 64 patches (4x4 pixels).
2.  **Token Mixing**: Allows different spatial locations to communicate using Linear layers and Transpositions.
3.  **Channel Mixing**: Allows different features within the same location to interact.

This provides the inductive bias of CNNs (spatial locality and weight sharing) using only the primitive operations supported by `js-pytorch`.

### Core Components

#### 1. **SwarmTrainer** (`src/core/trainer.js`)
- Manages local training loop with evolutionary optimization.
- Fetches real training data from IndexedDB batches.

#### 2. **ModelManager** (`src/core/models.js`)
- Manages **js-pytorch** models and state.
- Supports 32x32 image input with 64-dimensional latent space.

#### 3. **TorchJSTrainer** (`src/torchjs/integration.js`)
- Provides hardware-accelerated training using WebGL via **js-pytorch**.
- Maps P2P swarm logic to real tensor operations and backpropagation.

#### 4. **InferenceEngine** (`src/utils/inference.js`)
- Implements real Schrödinger Bridge sampling (Reverse SDE).
- Uses iterative drift updates to transform noise into coherent images.

## 🚀 Implementation Details

### js-pytorch Integration

The system uses **js-pytorch 0.7.2** for browser-based execution:

```javascript
// MLP-Mixer based Model (simplified)
class LabelConditionedVAE extends torch.nn.Module {
  constructor() {
    this.enc_mixer = new MixerBlock(64, 64, 32, 128);
    this.dec_mixer = new MixerBlock(64, 64, 32, 128);
  }

  forward(x, labels) {
    // Real backprop-capable forward pass
  }
}
```

## 🛠️ Technical Specifications

### System Requirements

- **Browser**: Chrome 80+, Firefox 75+, Safari 14+, Edge 80+
- **Hardware Acceleration**: WebGL enabled for js-pytorch speed.
- **Memory**: Minimum 2GB RAM.
- **Storage**: IndexedDB for real training data persistence.

### Dependencies

```json
{
  "js-pytorch": "Browser-based PyTorch runtime (WebGL)",
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

**Note**: This system implements **real model training and inference**. Unlike previous versions, it performs actual gradient descent and SDE-based sampling using **js-pytorch** hardware acceleration.

_Last updated: April 2026_
