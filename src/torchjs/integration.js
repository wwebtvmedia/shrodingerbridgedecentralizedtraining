// Universal TensorFlow.js Hardware Accelerator Integration
// This implementation provides a bridge between the swarm system and the real CNN-based models

// First ensure seedrandom is available before importing TensorFlow.js
if (typeof window !== 'undefined' && typeof window.seedrandom === 'undefined') {
  // Load seedrandom polyfill synchronously since we need it before tf import
  (function() {
    // Define the Alea constructor that TensorFlow.js expects (Dc class in rand_util.ts)
    function Alea(seed) {
      // Store seed for potential state management
      this._seed = seed;
      
      // Initialize internal state based on seed
      let s = 0;
      if (seed) {
        // Simple hash function for seed
        for (let i = 0; i < seed.length; i++) {
          s = (s << 5) - s + seed.charCodeAt(i);
          s |= 0;
        }
      }
      this.s = s || 0;
      this.c = 1;
      
      // The alea method should return a PRNG function
      this.alea = function() {
        const self = this;
        const rng = function() {
          return self.random();
        };
        rng.int32 = function() { return self.int32(); };
        rng.double = function() { return self.double(); };
        rng.quick = function() { return self.quick(); };
        return rng;
      };
    }
    
    Alea.prototype.random = function() {
      // Multiply-with-carry algorithm similar to actual Alea
      const t = 2091639 * this.s + this.c * 2.3283064365386963e-10;
      this.s = t | 0;
      this.c = t - this.s;
      return this.s / 0x100000000;
    };
    
    Alea.prototype.int32 = function() {
      return (this.random() * 0x100000000) | 0;
    };
    
    Alea.prototype.double = function() {
      return this.random();
    };
    
    Alea.prototype.quick = function() {
      return this.random();
    };
    
    // Main seedrandom function
    const seedrandom = function(seed, options) {
      const alea = new Alea(seed);
      return alea.alea();
    };
    
    // The alea property should be the Alea constructor
    seedrandom.alea = Alea;
    
    // Also expose Alea globally for direct access
    seedrandom.Alea = Alea;
    
    window.seedrandom = seedrandom;
    if (typeof globalThis !== 'undefined') {
      globalThis.seedrandom = seedrandom;
    }
    console.log("✅ seedrandom polyfill loaded (synchronous)");
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
