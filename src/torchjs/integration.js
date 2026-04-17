// Universal TensorFlow.js Hardware Accelerator Integration
// This implementation provides a bridge between the swarm system and the real CNN-based models

// Check for seedrandom and wait if not yet available (for browser environments)
function checkSeedRandom() {
  if (
    typeof window !== "undefined" &&
    typeof window.seedrandom === "undefined"
  ) {
    // If we're at the top-level load, just log warning
    // We'll also check again during async initialize()
    console.warn(
      "⚠️ window.seedrandom not found at module load. If TensorFlow.js fails with 't.alea is not a function', ensure it is loaded before application logic.",
    );
  }
}
checkSeedRandom();

import * as tf from "@tensorflow/tfjs";
import { EnhancedLabelTrainer } from "./training.js";

export class TorchJSTrainer {
  constructor() {
    this.trainer = null;
    this.isInitialized = false;
    this.device = "webgl";
    this.status = "initializing";

    // Start initialization
    this.initialize().catch((err) => {
      console.error("❌ TFJS Initialization Error:", err);
      this.status = "failed";
    });
  }

  async initialize() {
    if (this.isInitialized) return;

    console.log("🔍 Initializing TensorFlow.js hardware acceleration...");

    try {
      // Seedrandom should already be available from synchronous polyfill at top
      // But double-check for Node.js environment
      if (
        typeof window === "undefined" &&
        typeof globalThis.seedrandom === "undefined"
      ) {
        // In Node.js, we might need to handle this differently
        // For now, just log a warning
        console.log(
          "⚠️  Running in Node.js environment, seedrandom may not be available",
        );
      }

      // Try to use WebGPU if available, fallback to WebGL
      await tf.ready();

      this.device = tf.getBackend();

      this.trainer = new EnhancedLabelTrainer(this.device);
      this.isInitialized = true;
      this.status = "ready";

      this.updateUIWithDevice("ready", this.device);
      console.log(
        `🚀 Swarm AI Engine: Using [TensorFlow.js / ${this.device.toUpperCase()}] (Hardware Accelerated Training)`,
      );
      console.log(
        `🧠 Architecture: CNN-based (Mirroring enhancedoptimaltransport)`,
      );
    } catch (error) {
      console.error("❌ TensorFlow.js initialization failed:", error);
      this.device = "cpu";
      this.status = "failed";
      throw error;
    }
  }

  updateUIWithDevice(status, device) {
    if (typeof window !== "undefined") {
      if (window.enhancedApp && window.enhancedApp.ui) {
        window.enhancedApp.ui.updateHardwareInfo(status, device);
      }
    }
  }

  setPhase(phase) {
    if (this.trainer) {
      this.trainer.setPhase(phase);
    }
  }

  async trainStep(batch, labels, textBytes = null) {
    if (!this.isInitialized) await this.initialize();

    const processedBatch = Array.isArray(batch) ? batch : [batch];
    const processedLabels = Array.isArray(labels) ? labels : [labels];

    return await this.trainer.trainStep(
      processedBatch,
      processedLabels,
      textBytes,
    );
  }

  async generateSamples(labels, count = 4, textBytes = null) {
    if (!this.isInitialized) await this.initialize();
    return await this.trainer.generateSamples(labels, count, textBytes);
  }

  async getHashSample() {
    if (!this.isInitialized) await this.initialize();
    return this.trainer.getHashSample();
  }

  async saveCheckpoint() {
    if (!this.isInitialized) await this.initialize();
    return this.trainer.getCheckpoint();
  }

  async loadCheckpoint(checkpoint) {
    if (!this.isInitialized) await this.initialize();
    this.trainer.loadCheckpoint(checkpoint);
  }

  getModelState() {
    if (!this.trainer) return { epoch: 0, phase: 1, device: this.device };
    return {
      epoch: this.trainer.epoch,
      phase: this.trainer.phase,
      device: this.device,
    };
  }

  dispose() {
    if (this.trainer) {
      this.trainer.dispose();
      this.trainer = null;
    }
    this.isInitialized = false;
  }
}

export const tfjsTrainer = new TorchJSTrainer();
export default tfjsTrainer;
