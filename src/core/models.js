import { tfjsTrainer } from "../torchjs/integration.js";
import { CONFIG } from "../config.js";

class ModelManager {
  constructor() {
    this.model = null;
    this.optimizer = null;
    this.isInitialized = false;

    // Use configuration from config.js
    this.config = CONFIG;

    // Model state
    this.state = {
      parameters: null,
      hash: null,
      version: "3.0.0", // Updated for tfjs integration
      tfjs_initialized: false,
    };
  }

  async initialize() {
    if (this.isInitialized) return;

    console.log("🧠 Initializing model manager with TensorFlow.js...");

    try {
      // Initialize tfjs trainer
      await tfjsTrainer.initialize();

      // Update model structure to use tfjs
      this.model = {
        vae: tfjsTrainer.vae,
        drift: tfjsTrainer.drift,
      };

      this.optimizer = tfjsTrainer.optimizers;

      this.isInitialized = true;
      this.state.tfjs_initialized = true;
      console.log("✅ Model manager initialized with TensorFlow.js");
    } catch (error) {
      console.error("❌ Model initialization failed:", error);
      // Fall back to prototype model
      console.log("🔄 Falling back to prototype model...");
      await this.createPrototypeModel();
      this.isInitialized = true;
    }
  }

  async createPrototypeModel() {
    // For prototype, create a simple model structure
    // In production, this would load the actual Schrödinger Bridge models

    this.model = {
      vae: {
        encoder: this.createEncoder(),
        decoder: this.createDecoder(),
      },
      drift: this.createDriftNetwork(),
    };

    // Create optimizer
    this.optimizer = {
      vae: { lr: this.config.learningRate },
      drift: { lr: this.config.learningRate * 2 },
    };

    // Generate initial hash
    await this.updateModelHash();
  }

  createEncoder() {
    // Simple encoder for prototype
    return {
      layers: [
        { type: "conv", in: 3, out: 16, kernel: 3 },
        { type: "conv", in: 16, out: 32, kernel: 3, stride: 2 },
        { type: "conv", in: 32, out: 64, kernel: 3, stride: 2 },
        { type: "linear", in: 64 * 8 * 8, out: this.config.latentChannels * 2 },
      ],
      activation: "silu",
    };
  }

  createDecoder() {
    // Simple decoder for prototype
    return {
      layers: [
        { type: "linear", in: this.config.latentChannels, out: 64 * 8 * 8 },
        { type: "conv_transpose", in: 64, out: 32, kernel: 3, stride: 2 },
        { type: "conv_transpose", in: 32, out: 16, kernel: 3, stride: 2 },
        { type: "conv", in: 16, out: 3, kernel: 3 },
      ],
      activation: "silu",
    };
  }

  createDriftNetwork() {
    // Simple drift network for prototype
    return {
      layers: [
        {
          type: "conv",
          in: this.config.latentChannels + this.config.labelEmbDim,
          out: 64,
          kernel: 3,
        },
        { type: "conv", in: 64, out: 128, kernel: 3 },
        { type: "conv", in: 128, out: 64, kernel: 3 },
        { type: "conv", in: 64, out: this.config.latentChannels, kernel: 3 },
      ],
      timeEmbedding: true,
      labelConditioning: true,
    };
  }

  async trainStep(batch, labels, phase) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    console.log(`Training step - Phase: ${phase}, Batch size: ${batch.length}`);

    // Use tfjs trainer if available
    if (this.state.tfjs_initialized && tfjsTrainer.isInitialized) {
      // Convert phase string to number for tfjs
      let phaseNum;
      switch (phase) {
        case "vae":
          phaseNum = 1;
          break;
        case "drift":
          phaseNum = 2;
          break;
        case "both":
          phaseNum = 3;
          break;
        default:
          phaseNum = 1;
      }

      // Set phase in tfjs trainer
      tfjsTrainer.setPhase(phaseNum);

      // Train step with tfjs
      const result = await tfjsTrainer.trainStep(batch, labels);

      // Update model hash
      await this.updateModelHash();

      return {
        loss: result.loss,
        metrics: {
          ...result.metrics,
          phase: phase,
          tfjs: true,
        },
      };
    } else {
      // Fall back to simulation
      console.log("⚠️ FAKE TRAINING: Using simulated training because TensorFlow.js is not available");
      // Also log to UI if available
      if (typeof window !== 'undefined' && window.enhancedApp && window.enhancedApp.ui && window.enhancedApp.ui.log) {
        window.enhancedApp.ui.log("⚠️ FAKE TRAINING: Using simulated training (TensorFlow.js not available)");
      }
      const loss = this.simulateLoss(batch, labels, phase);
      await this.simulateParameterUpdate(loss);
      await this.updateModelHash();

      return {
        loss,
        metrics: {
          reconstruction_loss: loss * 0.7,
          kl_loss: loss * 0.2,
          diversity_loss: loss * 0.1,
          phase: phase,
          torchjs: false,
          simulated: true,
        },
      };
    }
  }

  simulateLoss(batch, labels, phase) {
    // Simulate different losses based on phase
    const baseLoss = 0.5 + Math.random() * 0.3;

    switch (phase) {
      case "vae":
        return baseLoss * 0.8; // VAE typically has lower loss
      case "drift":
        return baseLoss * 1.2; // Drift training can be harder
      case "both":
        return baseLoss;
      default:
        return baseLoss;
    }
  }

  async simulateParameterUpdate(loss) {
    // Simulate parameter updates
    // In real implementation, this would use WebTorch autograd

    // Update state to reflect "training"
    if (!this.state.parameters) {
      this.state.parameters = {
        vae: this.generateRandomParameters(1000),
        drift: this.generateRandomParameters(500),
      };
    } else {
      // Simulate small random changes
      this.state.parameters.vae = this.addNoise(
        this.state.parameters.vae,
        0.01,
      );
      this.state.parameters.drift = this.addNoise(
        this.state.parameters.drift,
        0.02,
      );
    }
  }

  generateRandomParameters(count) {
    const params = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      params[i] = Math.random() * 2 - 1; // Uniform [-1, 1]
    }
    return params;
  }

  addNoise(parameters, scale) {
    const noisy = new Float32Array(parameters.length);
    for (let i = 0; i < parameters.length; i++) {
      noisy[i] = parameters[i] + (Math.random() * 2 - 1) * scale;
    }
    return noisy;
  }

  async updateModelHash() {
    // Generate a simple hash from model parameters
    const params = this.state.parameters;

    if (!params) {
      this.state.hash = "initial";
      return;
    }

    // Combine parameter arrays
    const allParams = [...(params.vae || []), ...(params.drift || [])];

    // Simple hash: average of first 100 parameters
    const sample = allParams.slice(0, Math.min(100, allParams.length));
    const sum = sample.reduce((a, b) => a + b, 0);
    const avg = sample.length > 0 ? sum / sample.length : 0;

    this.state.hash = `model_${avg.toFixed(6).replace(".", "")}`;
  }

  async getModelHash() {
    if (!this.state.hash) {
      await this.updateModelHash();
    }
    return this.state.hash;
  }

  async getState() {
    return {
      parameters: this.state.parameters,
      hash: this.state.hash,
      version: this.state.version,
      config: this.config,
      timestamp: Date.now(),
    };
  }

  async setState(state) {
    if (!state) return;

    this.state.parameters = state.parameters;
    this.state.hash = state.hash;
    this.state.version = state.version || "1.0.0";

    if (state.config) {
      this.config = { ...this.config, ...state.config };
    }

    console.log(`📥 Model state loaded (hash: ${state.hash})`);
  }

  async loadModel(modelData) {
    try {
      console.log("📥 Loading model from external source...");

      // Extract state from model data
      const state = modelData.modelData || modelData;

      // Validate state
      if (!state || !state.hash) {
        throw new Error("Invalid model data");
      }

      // Load state
      await this.setState(state);

      console.log(`✅ Model loaded: ${state.hash}`);
      return true;
    } catch (error) {
      console.error("❌ Failed to load model:", error);
      return false;
    }
  }

  /**
   * Stub for loading from a PyTorch checkpoint file (e.g., latest.pt).
   * In a real implementation, this would parse the .pt file (using a server-side
   * Python script or WebTorch compatibility layer) and load the weights.
   * The checkpoint can be inspected using inspect_checkpoint.py.
   */
  async loadFromPyTorchCheckpoint(checkpointPath = "latest.pt") {
    console.log(
      `📥 Attempting to load PyTorch checkpoint from ${checkpointPath}...`,
    );
    console.log(
      "⚠️  This is a stub. Real implementation would require WebTorch compatibility.",
    );

    // For prototype, we could load the web-friendly JSON version
    try {
      const response = await fetch("/models/checkpoint_web.json");
      if (!response.ok) throw new Error("Failed to fetch checkpoint");
      const webCheckpoint = await response.json();

      // Simulate loading metadata
      console.log(
        `📊 Checkpoint metadata: epoch ${webCheckpoint.metadata.epoch}, phase ${webCheckpoint.metadata.phase}`,
      );
      console.log(
        "📝 Note: Actual model weights are not loaded in this prototype.",
      );

      // Update config if present
      if (webCheckpoint.config) {
        this.config = { ...this.config, ...webCheckpoint.config };
      }

      return {
        success: true,
        epoch: webCheckpoint.metadata.epoch,
        phase: webCheckpoint.metadata.phase,
        message: "Checkpoint metadata loaded (weights not implemented)",
      };
    } catch (error) {
      console.error("❌ Failed to load checkpoint:", error);
      return {
        success: false,
        error: error.message,
      };
    }
  }

  async generateSample(label = 0) {
    if (!this.isInitialized) {
      throw new Error("Model not initialized");
    }

    // Simulate sample generation
    // In real implementation, this would use the decoder

    return {
      label,
      timestamp: Date.now(),
      quality: 0.7 + Math.random() * 0.3,
    };
  }

  async encodeImage(imageData) {
    // Simulate encoding
    const latent = new Float32Array(this.config.latentChannels);
    for (let i = 0; i < latent.length; i++) {
      latent[i] = Math.random() * 2 - 1;
    }
    return latent;
  }

  async decodeLatent(latent, label = 0) {
    // Simulate decoding
    // Returns simulated image data
    return {
      width: 64,
      height: 64,
      channels: 3,
      data: new Uint8Array(64 * 64 * 3).map(() => Math.random() * 255),
    };
  }

  async computeDrift(latent, time, label = 0) {
    // Simulate drift computation
    const drift = new Float32Array(latent.length);
    for (let i = 0; i < drift.length; i++) {
      drift[i] = (Math.random() * 2 - 1) * 0.1;
    }
    return drift;
  }

  getModelSize() {
    if (!this.state.parameters) return 0;

    let size = 0;
    if (this.state.parameters.vae) {
      size += this.state.parameters.vae.length * 4; // Float32 = 4 bytes
    }
    if (this.state.parameters.drift) {
      size += this.state.parameters.drift.length * 4;
    }

    return size; // bytes
  }

  async exportModel(format = "json") {
    const state = await this.getState();

    switch (format) {
      case "json":
        return JSON.stringify(state, null, 2);

      case "binary":
        // Simple binary format for prototype
        const buffer = new ArrayBuffer(this.getModelSize());
        const view = new DataView(buffer);

        // In real implementation, would pack parameters
        return buffer;

      default:
        throw new Error(`Unsupported format: ${format}`);
    }
  }

  async importModel(data, format = "json") {
    try {
      let state;

      switch (format) {
        case "json":
          state = typeof data === "string" ? JSON.parse(data) : data;
          break;

        case "binary":
          // Parse binary format
          state = this.parseBinaryModel(data);
          break;

        default:
          throw new Error(`Unsupported format: ${format}`);
      }

      await this.setState(state);
      return true;
    } catch (error) {
      console.error("Import failed:", error);
      return false;
    }
  }

  parseBinaryModel(buffer) {
    // Simple parser for prototype
    return {
      parameters: {
        vae: new Float32Array(1000).map(() => Math.random() * 2 - 1),
        drift: new Float32Array(500).map(() => Math.random() * 2 - 1),
      },
      hash: "imported",
      version: "1.0.0",
    };
  }

  reset() {
    this.model = null;
    this.optimizer = null;
    this.isInitialized = false;
    this.state = {
      parameters: null,
      hash: null,
      version: "1.0.0",
    };

    console.log("🔄 Model manager reset");
  }
}

export { ModelManager };
