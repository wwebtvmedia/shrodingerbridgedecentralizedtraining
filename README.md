# Swarm Schrödinger Bridge Training System

[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![Platform: Web/Node.js](https://img.shields.io/badge/Platform-Web%2FNode.js-orange.svg)]()

A decentralized, self-organizing swarm training system for Schrödinger Bridge models using **TensorFlow.js** and peer-to-peer networking. This system implements a novel evolutionary optimization approach where multiple browser clients collaboratively train generative models without any central coordinator.

---

## 📑 Table of Contents
- [📖 Abstract](#-abstract)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🧮 Mathematical Foundations](#-mathematical-foundations)
- [📂 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [🛠️ Technical Specifications](#️-technical-specifications)
- [📝 Reviewer Quick Start](#-reviewer-quick-start)

---

## 📖 Abstract

This project implements a distributed training environment for high-resolution (96x96) generative models. By leveraging **Gossip Protocols** and **LoRA (Low-Rank Adaptation)**, it allows browser-based clients to participate in a global training swarm with minimal bandwidth requirements. The architecture is aligned with state-of-the-art CNN-based PyTorch models, optimized for browser execution via TensorFlow.js.

> [!IMPORTANT]
> This system performs **real gradient descent and SDE-based sampling** directly in the browser. It is not a simulation; it is a functional decentralized training platform.

## ✨ Key Features

- **Decentralized Swarm**: Collaborative training without a central server.
- **96x96 High-Res CNN**: Advanced architecture with Residual Blocks and Axial Attention.
- **SARSA Adaptive Logic**: Reinforcement learning for dynamic task selection (VAE vs Drift).
- **Schrödinger Bridge SDE**: Generative modeling based on optimal transport.
- **LoRA Optimization**: Efficient synchronization by only sharing low-rank weight adapters.
- **Three-Phase Training**: Structured evolution from VAE to Drift to Joint optimization.
- **Hardware Accelerated**: Native WebGL/WebGPU support via TensorFlow.js.

## 🏗️ System Architecture

### CNN Residual Architecture (96x96)
1.  **Residual Blocks**: Deep feature extraction with stable gradient flow.
2.  **Spatial Split Attention**: Axial attention for long-range dependencies.
3.  **Subpixel Upsampling**: High-fidelity reconstruction.
4.  **Label Conditioning**: FiLM-based modulation for class-conditional generation.

### Core Components
| Component | File | Description |
| :--- | :--- | :--- |
| **SwarmTrainer** | `src/core/trainer.js` | Manages local training loop & evolutionary optimization. |
| **ModelManager** | `src/core/models.js` | TF.js model management (12x12x8 latent space). |
| **TorchJSTrainer** | `src/torchjs/integration.js` | Hardware-accelerated training pipeline. |
| **InferenceEngine** | `src/utils/inference.js` | SDE-based sampling (Reverse SDE). |

## 🧮 Mathematical Foundations

### Schrödinger Bridge Formulation
The Schrödinger Bridge finds the optimal drift \( u^\*(x, t) \) that connects noise to data distribution while minimizing energy:
\[ \mathbb{E}\left[\int_0^T \frac{1}{2} \|u(X_t, t)\|^2 dt\right] \]

### Low-Rank Adaptation (LoRA)
- **Rank-8 Adapters**: Minimizes trainable parameters.
- **Fast Sync**: Only LoRA weights are exchanged, reducing network overhead by ~10x.

## 📂 Project Structure

```text
/
├── public/                 # Static assets & vendor libs (torch.min.js)
├── src/
│   ├── core/               # Core training & phase logic
│   ├── network/            # P2P (PeerJS) & Tunnel logic
│   ├── storage/            # IndexedDB database management
│   ├── torchjs/            # LoRA & TF.js integration layers
│   ├── ui/                 # Dashboard & visualization manager
│   └── utils/              # Inference, sanitizers, & exports
├── server/                 # Consolidation/Signaling server
├── tests/                  # Test suites
└── scripts/                # Deployment & Cloudflare utilities
```

## 🚀 Getting Started

### 1. Installation
```bash
npm install
```

### 2. Run Development Environment
```bash
npm run dev
```
Open `http://localhost:3000` in multiple tabs to witness the swarm in action.

### 3. Build for Production
```bash
npm run build
```

## 🛠️ Technical Specifications

- **Browser**: Modern browser with WebGL/WebGPU support.
- **Memory**: 4GB+ RAM recommended for 96x96 resolution.
- **Backend**: TensorFlow.js (Hardware Accelerated).
- **Communication**: WebRTC (Simple-Peer).

## 📝 Reviewer Quick Start

To effectively review this implementation:
1.  **Check `src/core/sarsa-optimizer.js`**: Understand the reinforcement learning logic for task selection.
2.  **Check `src/core/phase.js`**: Understand the three-phase training state machine.
3.  **Check `src/torchjs/lora.js`**: Review the low-rank adaptation implementation for TF.js layers.
4.  **Check `src/utils/inference.js`**: Examine the Euler-Maruyama integration for SDE sampling.
5.  **Run `node tests/sarsa.test.js`**: Validate the SARSA Q-learning logic.

---
_Last updated: April 19, 2026_
