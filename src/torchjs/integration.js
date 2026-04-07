// Universal js-pytorch Hardware Accelerator Integration
// This implementation provides a bridge between the swarm system and js-pytorch (WebTorch)

import { EnhancedLabelTrainer } from "./training.js";

/**
 * Robust torch resolution
 */
async function resolveTorch() {
  if (typeof window !== 'undefined') {
    if (window.torch) return window.torch;
    return new Promise((resolve, reject) => {
      let attempts = 0;
      const interval = setInterval(() => {
        if (window.torch) {
          clearInterval(interval);
          resolve(window.torch);
        }
        if (attempts++ > 100) {
          clearInterval(interval);
          reject(new Error("Timeout waiting for torch"));
        }
      }, 100);
    });
  }
  try {
    const JSTorch = await import('js-pytorch');
    const t = JSTorch.torch || (JSTorch.default && JSTorch.default.torch) || JSTorch;
    if (!t || !t.nn) throw new Error("Invalid js-pytorch module");
    return t;
  } catch (e) {
    if (typeof globalThis !== 'undefined' && globalThis.torch) {
      return globalThis.torch;
    }
    throw e;
  }
}

const torch = await resolveTorch();

export class TorchJSTrainer {
  constructor() {
    this.trainer = null;
    this.isInitialized = false;
    this.device = 'gpu'; 
    this.status = 'initializing';

    // Start initialization
    this.initialize().catch(err => {
      console.error("❌ TorchJS Initialization Error:", err);
      this.status = 'failed';
    });
  }

  async initialize() {
    if (this.isInitialized) return;

    console.log("🔍 Initializing js-pytorch hardware acceleration (GPU)...");
    
    try {
      this.device = 'gpu'; 
      
      this.trainer = new EnhancedLabelTrainer(this.device);
      this.isInitialized = true;
      this.status = 'ready';
      
      this.updateUIWithDevice("ready", this.device);
      console.log(`🚀 Swarm AI Engine: Using [JS-PYTORCH / ${this.device.toUpperCase()}] (Hardware Accelerated Training)`);
    } catch (error) {
      console.error("❌ js-pytorch initialization failed:", error);
      this.device = 'cpu';
      this.status = 'failed';
      throw error;
    }
  }

  updateUIWithDevice(status, device) {
    if (typeof window !== 'undefined') {
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

  async trainStep(batch, labels) {
    if (!this.isInitialized) await this.initialize();
    
    const processedBatch = Array.isArray(batch) ? batch : [batch];
    const processedLabels = Array.isArray(labels) ? labels : [labels];
    
    return await this.trainer.trainStep(processedBatch, processedLabels);
  }

  async generateSamples(labels, count = 4) {
    if (!this.isInitialized) await this.initialize();
    return await this.trainer.generateSamples(labels, count);
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
      device: this.device
    };
  }
}

export const tfjsTrainer = new TorchJSTrainer();
export default tfjsTrainer;
