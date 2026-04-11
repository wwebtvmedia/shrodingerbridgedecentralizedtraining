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
    // For now, we'll just define a minimal polyfill
    if (typeof window !== 'undefined') {
      window.seedrandom = function(seed) {
        // Simple polyfill that returns a pseudo-random function
        let x = 0;
        if (seed) {
          // Simple hash-based seed
          for (let i = 0; i < seed.length; i++) {
            x = (x << 5) - x + seed.charCodeAt(i);
            x |= 0;
          }
        }
        const random = function() {
          x = (x * 9301 + 49297) % 233280;
          return x / 233280;
        };
        random.double = random;
        random.int32 = function() {
          return Math.floor(random() * 0xFFFFFFFF) | 0;
        };
        random.quick = random;
        random.alea = random;
        return random;
      };
      console.log("✅ seedrandom polyfill loaded");
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
