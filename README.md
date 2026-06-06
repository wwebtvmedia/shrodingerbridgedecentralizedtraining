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

| Component            | File                         | Description                                              |
| :------------------- | :--------------------------- | :------------------------------------------------------- |
| **SwarmTrainer**     | `src/core/trainer.js`        | Manages local training loop & evolutionary optimization. |
| **ModelManager**     | `src/core/models.js`         | TF.js model management (12x12x8 latent space).           |
| **TorchJSTrainer**   | `src/torchjs/integration.js` | Hardware-accelerated training pipeline.                  |
| **InferenceEngine**  | `src/utils/inference.js`     | SDE-based sampling (Reverse SDE).                        |
| **CloudflareTunnel** | `src/network/tunnel.js`      | WebSocket transport + peer directory (per-peer).         |
| **Signaling Server** | `server/index.js`            | Rendezvous directory, identity issuer, message relay.    |

### Peer Discovery Protocol

Peers find each other through the server, which acts as a **rendezvous
directory** (the standard signaling pattern for browser P2P). The handshake:

```text
client                              server
  │ ── WS connect (token subproto) ──▶│  authenticate (constant-time)
  │ ◀── identity {peerId, sig} ───────│  mint random id, HMAC-sign with secret
  │ ── register {peerId, sig, meta} ─▶│  verify signature, add to directory
  │ ◀── PEER_CONNECTED (per peer) ────│  send joiner the current roster …
  │                                   │  … and announce joiner to the others
  │ ◀── PEER_DISCONNECTED ────────────│  on any peer leaving (close/idle reap)
```

- **Host-issued identity.** The client does **not** choose its own id. The
  server mints a cryptographically-random `peerId` and signs it (HMAC-SHA256
  over the server secret), so every directory entry is a value minted and
  signed by the host. A registration whose signature doesn't verify is rejected.
- **Anti-spoof relay.** The server stamps the authoritative sender id onto every
  relayed `PEER_MESSAGE`/`BROADCAST`, so a peer cannot impersonate another.
- **Per-peer directory.** Each client keeps its own live directory of known
  peers (`tunnel.getPeers()` / `getPeerCount()`), updated from the
  `PEER_CONNECTED`/`PEER_DISCONNECTED` stream. The server's shared roster is
  also exposed read-only at `GET /api/peers`.

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
│   ├── network/            # Peer directory & WebSocket tunnel logic
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

### 2. Configure (required for the server)

The consolidation server authenticates every protected endpoint with a token.
Copy the env template and set a strong secret:

```bash
cp .env.template .env
# Generate a token and put it in .env as SECRET_TOKEN=...
openssl rand -hex 32
```

| Variable          | Purpose                                                                                                                                                           |
| :---------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SECRET_TOKEN`    | Shared auth token for all protected HTTP/WS endpoints. If unset, the server generates a random ephemeral token and prints it to the log (never a shared default). |
| `ALLOWED_ORIGINS` | Comma-separated CORS allowlist (defaults to `localhost`).                                                                                                         |
| `MAX_CONN_PER_IP` | Max concurrent WebSocket connections per IP (default 20).                                                                                                         |
| `PORT`            | Server port (default 3001).                                                                                                                                       |

The **browser client** must present the same token. It is no longer hardcoded —
provide it at runtime, e.g. in the browser console:

```js
localStorage.setItem("swarmAuthToken", "<your SECRET_TOKEN>");
```

or by injecting `window.SWARM_AUTH_TOKEN` from your deployment/build step.

### 3. Run Development Environment

```bash
# Front-end only
npm run dev            # Vite dev server (http://localhost:3000)

# Consolidation server (reads SECRET_TOKEN from the environment)
SECRET_TOKEN=$(openssl rand -hex 32) npm run server

# Both together
npm run dev:full
```

Open the app in multiple tabs to witness the swarm in action.

### Testing across multiple devices (phones, other networks)

To see peers from separate devices (e.g. two phones on different networks),
they must all reach the **same** server instance and agree on identity:

1.  Open the **`/enhanced`** page on every device (it connects back to whatever
    origin served it). Do **not** use `localhost` URLs on a phone — `localhost`
    is the phone itself. Use the server's LAN IP (`http://<ip>:3001`) or a
    public tunnel URL (`https://…`, which the client upgrades to `wss://`).
2.  Run the server with a **stable** `SECRET_TOKEN` (otherwise it mints a random
    one per run that devices can't know).
3.  On each device set the matching token before connecting:
    `localStorage.setItem('swarmAuthToken','<token>')`, then reload.
4.  Hit **Connect** on each. Each device is assigned a host-signed peer id and
    appears in the others' peer count. Sanity-check from any device by visiting
    `…/api/peers` — it lists everyone currently connected.

If `/api/peers` shows `0`, the sockets aren't connecting — almost always a token
mismatch or a `localhost`/`ws://`-over-`https` (mixed content) issue.

### 4. Build for Production

```bash
npm run build
```

## 🛠️ Technical Specifications

- **Browser**: Modern browser with WebGL/WebGPU support.
- **Memory**: 4GB+ RAM recommended for 96x96 resolution.
- **Backend**: TensorFlow.js (Hardware Accelerated).
- **Communication**: WebSocket signaling/relay with a host-issued, signed peer
  directory (WebRTC direct data channels are the documented next step).

## 🔒 Security & Hardening

The server and client validate and sanitize all untrusted input:

- **Auth**: required `SECRET_TOKEN` (no shared default), constant-time compare,
  token sent via WebSocket subprotocol (not the URL).
- **Identity**: peer ids are minted (random) and HMAC-signed by the host, not
  chosen by clients; relayed messages are stamped with the verified sender id.
- **CORS**: restricted to an `ALLOWED_ORIGINS` allowlist.
- **Input validation**: strict message schemas, finite-number checks, payload
  size caps, and prototype-pollution-safe JSON parsing/sanitization.
- **DoS guards**: submitted-model size cap, model-file pruning, idle-connection
  reaper, and per-IP connection limits.
- **Output safety**: all server/peer-supplied fields are escaped before being
  rendered in the DOM (no `innerHTML` injection).
- **Checkpoints**: Python tooling loads with `weights_only=True` to avoid pickle
  code execution from untrusted `.pt` files.

> [!CAUTION]
> `start-swarm.sh` stashes local changes before pulling. The old destructive
> reset is opt-in via `FORCE_RESET=1` and preserves `data/`, `models/`,
> `checkpoints/`, and `.env`.

## 📝 Reviewer Quick Start

To effectively review this implementation:

1.  **Check `src/core/sarsa-optimizer.js`**: Understand the reinforcement learning logic for task selection.
2.  **Check `src/core/phase.js`**: Understand the three-phase training state machine.
3.  **Check `src/torchjs/lora.js`**: Review the low-rank adaptation implementation for TF.js layers.
4.  **Check `src/utils/inference.js`**: Examine the Euler-Maruyama integration for SDE sampling.
5.  **Run `node tests/sarsa.test.js`**: Validate the SARSA Q-learning logic.
6.  **Run `node tests/peer-presence.test.mjs`**: Validate the peer discovery
    protocol (host-signed identity, directory sharing, spoof rejection).

---

_Last updated: June 6, 2026_
