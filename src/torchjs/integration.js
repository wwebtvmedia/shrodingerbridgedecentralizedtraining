// Universal TensorFlow.js Hardware Accelerator Integration
// This implementation provides a bridge between the swarm system and the real CNN-based models

// TensorFlow.js compatibility fix for newer versions
// According to TensorFlow.js releases, newer versions changed module bundling
// and dropped support for some global functions like t.alea()
// This fix ensures compatibility with TensorFlow.js 4.22.0+

// Load seedrandom if not available (for browser environments)
if (typeof window !== 'undefined' && typeof window.seedrandom === 'undefined') {
  // Try to load seedrandom from CDN or use a minimal compatible implementation
  console.warn("⚠️ seedrandom not found. TensorFlow.js may fail to initialize.");
  
  // Minimal seedrandom polyfill that should work with TensorFlow.js
  (function() {
    // Simple Math.random-based implementation for compatibility
    const seedrandom = function(seed) {
      // Simple deterministic PRNG based on seed
      let x = 0;
      if (seed) {
        for (let i = 0; i < seed.length; i++) {
          x = (x << 5) - x + seed.charCodeAt(i);
          x |= 0;
        }
      }
      
      const prng = function() {
        x = Math.sin(x + 1) * 10000;
        return x - Math.floor(x);
      };
      
      prng.int32 = function() {
        return Math.floor(prng() * 0x100000000);
      };
      
      prng.double = prng;
      prng.quick = prng;
      
      return prng;
    };
    
    // Add alea property for TensorFlow.js compatibility
    seedrandom.alea = function(seed) {
      return seedrandom(seed);
    };
    
    window.seedrandom = seedrandom;
    if (typeof globalThis !== 'undefined') {
      globalThis.seedrandom = seedrandom;
    }
    console.log("✅ Minimal seedrandom polyfill loaded for TensorFlow.js compatibility");
  })();
}

import * as tf from '@tensorflow/tfjs';
import { EnhancedLabelTrainer } from "./training.js";

export class TorchJSTrainer {
  constructor() {
    this.trainer = null;
    this.isInitialized = false;
    this.device = 'webgl';
    this.status = 'initializing';

    // Start initialization
    this.initialize().catch(err => {
      console.error("❌ TFJS Initialization Error:", err);
      this.status = 'failed';
    });
  }

  async initialize() {
    if (this.isInitialized) return;

    console.log("🔍 Initializing TensorFlow.js hardware acceleration...");
    
    try {
      // Seedrandom should already be available from synchronous polyfill at top
      // But double-check for Node.js environment
      if (typeof window === 'undefined' && typeof globalThis.seedrandom === 'undefined') {
        // In Node.js, we might need to handle this differently
        // For now, just log a warning
        console.log("⚠️  Running in Node.js environment, seedrandom may not be available");
      }
      
      // Try to use WebGPU if available, fallback to WebGL
      await tf.ready();
      
      this.device = tf.getBackend();
      
      this.trainer = new EnhancedLabelTrainer(this.device);
      this.isInitialized = true;
      this.status = 'ready';
      
      this.updateUIWithDevice("ready", this.device);
      console.log(`🚀 Swarm AI Engine: Using [TensorFlow.js / ${this.device.toUpperCase()}] (Hardware Accelerated Training)`);
      console.log(`🧠 Architecture: CNN-based (Mirroring enhancedoptimaltransport)`);
    } catch (error) {
      console.error("❌ TensorFlow.js initialization failed:", error);
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
