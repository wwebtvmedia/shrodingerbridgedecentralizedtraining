// Integration module for torch-js with existing swarm system

import { CONFIG } from "../config.js";
import { LabelConditionedVAE, LabelConditionedDrift } from "./models.js";

export class TorchJSTrainer {
  constructor() {
    this.vae = null;
    this.drift = null;
    this.optimizers = null;
    this.phase = 1;
    this.epoch = 0;
    this.isInitialized = false;

    // Check if torch is available
    this.torchAvailable = false;
    this.checkTorchAvailability();
  }

  async checkTorchAvailability() {
    try {
      // Try to import js-pytorch
      const torchModule = await import("js-pytorch");
      this.torch = torchModule;
      this.torchAvailable = true;
      console.log("✅ js-pytorch available");
    } catch (error) {
      console.warn("⚠️ js-pytorch not available, using mock mode");
      this.torchAvailable = false;
    }
  }

  async initialize() {
    if (this.isInitialized) return;

    console.log("🧠 Initializing TorchJS Trainer...");

    try {
      // Initialize models
      this.vae = new LabelConditionedVAE();
      this.drift = new LabelConditionedDrift();

      // Set training mode
      this.vae.training = true;
      this.drift.training = true;

      // Create optimizers
      this.optimizers = {
        vae: {
          parameters: this.getParameters(this.vae),
          lr: CONFIG.LR,
          weight_decay: CONFIG.WEIGHT_DECAY,
        },
        drift: {
          parameters: this.getParameters(this.drift),
          lr: CONFIG.LR * CONFIG.DRIFT_LR_MULTIPLIER,
          weight_decay: CONFIG.WEIGHT_DECAY,
        },
      };

      this.isInitialized = true;
      console.log("✅ TorchJS Trainer initialized");
    } catch (error) {
      console.error("❌ TorchJS Trainer initialization failed:", error);
      throw error;
    }
  }

  getParameters(model) {
    // Extract trainable parameters from model
    const params = [];
    const queue = [model];

    while (queue.length > 0) {
      const obj = queue.shift();
      if (obj && typeof obj === "object") {
        for (const key in obj) {
          const value = obj[key];
          if (value && value.requires_grad !== undefined) {
            params.push(value);
          } else if (value && typeof value === "object") {
            queue.push(value);
          }
        }
      }
    }

    return params;
  }

  setPhase(phase) {
    console.log(`🔄 Setting training phase to ${phase}`);
    this.phase = phase;

    // Adjust model training modes based on phase
    if (phase === 1) {
      // VAE only
      this.vae.training = true;
      this.drift.training = false;
    } else if (phase === 2) {
      // Drift only, freeze VAE decoder
      this.vae.training = true;
      this.drift.training = true;
      // Note: In full implementation, would freeze decoder parameters
    } else if (phase === 3) {
      // Both trainable
      this.vae.training = true;
      this.drift.training = true;
    }
  }

  async trainStep(batch, labels) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    // Simulate training step based on phase
    let loss = 0;
    let metrics = {};

    if (this.phase === 1) {
      // VAE training
      const [recon, mu, logvar] = this.vae.forward(batch, labels);

      // Compute losses
      const recon_loss = this.computeReconstructionLoss(batch, recon);
      const kl_loss = this.computeKLLoss(mu, logvar);

      loss = recon_loss + kl_loss * CONFIG.KL_WEIGHT;
      metrics = {
        recon_loss,
        kl_loss,
        total_loss: loss,
        phase: "vae",
      };
    } else if (this.phase === 2 || this.phase === 3) {
      // Drift training
      // Get latent representation
      const [mu, logvar] = this.vae.encode(batch, labels);

      // Sample time
      const t = Math.random();

      // Sample z0 and z1
      const z0 = this.sampleNoise(mu.shape);
      const z1 = mu; // Simplified

      // Interpolate
      const zt = z0 * (1 - t) + z1 * t;

      // Predict drift
      const pred_drift = this.drift.forward(zt, t, labels);
      const target_drift = z1 - z0;

      // Compute drift loss
      const drift_loss = this.computeDriftLoss(pred_drift, target_drift);

      loss = drift_loss * CONFIG.DRIFT_WEIGHT;
      metrics = {
        drift_loss,
        total_loss: loss,
        phase: "drift",
      };

      if (this.phase === 3) {
        // Also compute reconstruction loss
        const recon = this.vae.decode(mu, labels);
        const recon_loss = this.computeReconstructionLoss(batch, recon);
        loss += recon_loss * CONFIG.RECON_WEIGHT * CONFIG.PHASE3_RECON_SCALE;
        metrics.recon_loss = recon_loss;
      }
    }

    // Simulate parameter update
    this.simulateParameterUpdate(loss);

    // Update epoch
    this.epoch++;

    return { loss, metrics };
  }

  computeReconstructionLoss(real, recon) {
    // Mean Absolute Error
    if (this.torchAvailable && this.torch) {
      return real.sub(recon).abs().mean().data[0];
    }
    // Mock implementation
    return 0.1 + Math.random() * 0.1;
  }

  computeKLLoss(mu, logvar) {
    // KL divergence
    if (this.torchAvailable && this.torch) {
      const kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp());
      return kl.mean().data[0];
    }
    // Mock implementation
    return 0.05 + Math.random() * 0.05;
  }

  computeDriftLoss(pred, target) {
    // Huber loss
    if (this.torchAvailable && this.torch) {
      const diff = pred.sub(target);
      const absDiff = diff.abs();
      const loss = absDiff.where(absDiff.lt(1), diff.pow(2).mul(0.5).add(0.5));
      return loss.mean().data[0];
    }
    // Mock implementation
    return 0.2 + Math.random() * 0.1;
  }

  sampleNoise(shape) {
    // Sample Gaussian noise
    if (this.torchAvailable && this.torch) {
      return this.torch.randn(shape);
    }
    // Mock implementation
    return Array.from(
      { length: shape.reduce((a, b) => a * b, 1) },
      () => (Math.random() - 0.5) * 2,
    ).reduce((arr, val, i) => {
      const idx = i % shape[0];
      if (!arr[idx]) arr[idx] = [];
      arr[idx].push(val);
      return arr;
    }, []);
  }

  simulateParameterUpdate(loss) {
    // Simulate gradient descent update
    if (this.torchAvailable && this.torch) {
      // Real backprop would go here
      return;
    }

    // Mock parameter update
    const optimizer =
      this.phase === 1 ? this.optimizers.vae : this.optimizers.drift;
    const lr = optimizer.lr;

    // Simulate weight update
    for (const param of optimizer.parameters) {
      if (param.data && param.grad) {
        // Mock gradient
        const grad = Math.random() - 0.5;
        param.data = param.data - lr * grad;

        // Weight decay
        if (optimizer.weight_decay > 0) {
          param.data = param.data * (1 - optimizer.weight_decay);
        }
      }
    }
  }

  async generateSamples(labels, count = 4) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    console.log(`🎨 Generating ${count} samples for labels: ${labels}`);

    // Generate samples using the VAE decoder
    const samples = [];
    for (let i = 0; i < count; i++) {
      // Sample from latent space
      const z = this.sampleNoise([
        1,
        CONFIG.LATENT_CHANNELS,
        CONFIG.LATENT_H,
        CONFIG.LATENT_W,
      ]);

      // Decode
      const sample = this.vae.decode(z, [labels[i % labels.length]]);
      samples.push(sample);
    }

    return samples;
  }

  getModelState() {
    return {
      epoch: this.epoch,
      phase: this.phase,
      vae_initialized: this.vae !== null,
      drift_initialized: this.drift !== null,
      torch_available: this.torchAvailable,
    };
  }

  async saveCheckpoint() {
    const checkpoint = {
      epoch: this.epoch,
      phase: this.phase,
      vae_state: this.vae ? this.vae.stateDict() : null,
      drift_state: this.drift ? this.drift.stateDict() : null,
      config: CONFIG,
    };

    console.log(`💾 Saved checkpoint at epoch ${this.epoch}`);
    return checkpoint;
  }

  async loadCheckpoint(checkpoint) {
    this.epoch = checkpoint.epoch || 0;
    this.phase = checkpoint.phase || 1;

    if (checkpoint.vae_state && this.vae) {
      this.vae.loadStateDict(checkpoint.vae_state);
    }

    if (checkpoint.drift_state && this.drift) {
      this.drift.loadStateDict(checkpoint.drift_state);
    }

    console.log(`📂 Loaded checkpoint from epoch ${this.epoch}`);
  }
}

// Create singleton instance
export const torchJSTrainer = new TorchJSTrainer();

// Export for use in existing system
export default torchJSTrainer;
