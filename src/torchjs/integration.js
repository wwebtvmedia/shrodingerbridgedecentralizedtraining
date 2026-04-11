// Universal TensorFlow.js Hardware Accelerator Integration
// This implementation provides a bridge between the swarm system and the real CNN-based models

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
      // Workaround for "t.alea is not a function" error
      // Ensure seedrandom library is available before tf.ready()
      if (typeof window !== 'undefined') {
        // In browser environment, ensure seedrandom is loaded
        if (typeof window.seedrandom === 'undefined') {
          console.log("⚠️  seedrandom not found, loading polyfill...");
          // Load seedrandom polyfill if needed
          await this.loadSeedRandomPolyfill();
        }
        // Also ensure globalThis has seedrandom for TensorFlow.js internal requires
        if (typeof globalThis !== 'undefined' && typeof globalThis.seedrandom === 'undefined') {
          globalThis.seedrandom = window.seedrandom;
        }
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

  async loadSeedRandomPolyfill() {
    // This is a workaround for the "t.alea is not a function" error
    // In a real implementation, you might load seedrandom from a CDN
    // For now, we'll just define a minimal polyfill that matches the seedrandom v3 API
    if (typeof window !== 'undefined') {
      // Define Alea generator class that TensorFlow.js expects
      class Alea {
        constructor(seed) {
          this.seed = seed;
          let s = 0;
          if (seed) {
            for (let i = 0; i < seed.length; i++) {
              s = (s << 5) - s + seed.charCodeAt(i);
              s |= 0;
            }
          }
          this.s = s;
          this.c = 1;
        }
        
        random() {
          const t = 2091639 * this.s + this.c * 2.3283064365386963e-10;
          this.s = t | 0;
          this.c = t - this.s;
          return this.s / 0x100000000;
        }
        
        int32() {
          return (this.random() * 0x100000000) | 0;
        }
        
        quick() {
          return this.random();
        }
        
        double() {
          return this.random();
        }
      }
      
      // Main seedrandom function
      const seedrandom = function(seed, options) {
        const generator = new Alea(seed);
        const rng = function() {
          return generator.random();
        };
        
        // Copy generator methods
        rng.int32 = function() { return generator.int32(); };
        rng.quick = function() { return generator.quick(); };
        rng.double = function() { return generator.double(); };
        rng.alea = rng; // Self-reference for compatibility
        
        if (options && options.state) {
          // Handle state restoration if needed
          generator.s = options.state[0] || 0;
          generator.c = options.state[1] || 1;
        }
        
        return rng;
      };
      
      // Add alea method to seedrandom function (as in the real seedrandom library)
      seedrandom.alea = function(seed, options) {
        return seedrandom(seed, options);
      };
      
      window.seedrandom = seedrandom;
      console.log("✅ seedrandom polyfill loaded (with Alea support)");
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
