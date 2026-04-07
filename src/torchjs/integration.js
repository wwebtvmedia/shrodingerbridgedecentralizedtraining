// Universal js-pytorch Hardware Accelerator Integration
// This implementation provides a bridge between the swarm system and js-pytorch (WebTorch)

import { torch } from 'js-pytorch';
import { EnhancedLabelTrainer } from "./training.js";

export class TorchJSTrainer {
  constructor() {
    this.trainer = null;
    this.isInitialized = false;
    this.device = 'cpu'; // Default
    this.status = 'initializing';

    // Start initialization
    this.initialize().catch(err => {
      console.error("❌ TorchJS Initialization Error:", err);
      this.status = 'failed';
    });
  }

  async initialize() {
    if (this.isInitialized) return;

    console.log("🔍 Initializing js-pytorch hardware acceleration...");
    
    try {
      // js-pytorch automatically tries to use GPU if available via WebGL
      // In some versions we might need to explicitly set it, but 0.7.2 is usually auto
      this.device = 'webgl'; 
      
      this.trainer = new EnhancedLabelTrainer();
      this.isInitialized = true;
      this.status = 'ready';
      
      this.updateUIWithDevice("ready", this.device);
      console.log(`🚀 Swarm AI Engine: Using [JS-PYTORCH / ${this.device.toUpperCase()}]`);
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
    
    // Convert batch to nested arrays if it's not already
    // js-pytorch expects arrays for tensor creation
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

// Export as tfjsTrainer to maintain compatibility with existing code
// but it's actually using js-pytorch now
export const tfjsTrainer = new TorchJSTrainer();
export default tfjsTrainer;
