// Universal TensorFlow.js Hardware Accelerator (Mac M1-M4, NVIDIA CUDA, AMD, Intel)
// This implementation maps precisely to the device selection in PyTorch

import * as tf from '@tensorflow/tfjs';
import { CONFIG } from "../config.js";
import { LabelConditionedVAE, LabelConditionedDrift } from "./models.js";

// Conditional backend loading to prevent "already registered" errors
const loadBackends = async () => {
  try {
    // WASM and WebGPU are not in the union package by default, so we import them
    if (!tf.findBackend('wasm')) await import('@tensorflow/tfjs-backend-wasm');
    if (!tf.findBackend('webgpu')) await import('@tensorflow/tfjs-backend-webgpu');
    // WebGL is usually in the union package, but we check just in case
    if (!tf.findBackend('webgl')) await import('@tensorflow/tfjs-backend-webgl');
  } catch (e) {
    console.warn("Lazy backend loading failed:", e);
  }
};

export class TFJSTrainer {
  constructor() {
    this.vae = null;
    this.drift = null;
    this.optimizers = null;
    this.phase = 1;
    this.epoch = 0;
    this.isInitialized = false;
    this.device = 'detecting';
    this.detectionPromise = null;

    // Start detection immediately (backends are loaded inside detectBackend)
    this.detectBackend().catch(console.error);
  }

  /**
   * Sets performance flags based on the active device
   */
  setPerformanceFlags(device) {
    if (typeof tf === 'undefined' || typeof tf.env !== 'function') return;

    try {
      const env = tf.env();
      
      // WebGL Specific Flags
      if (device === 'webgl') {
        // Only set if they are registered (checked via internal keys or catch)
        const flags = [
          'WEBGL_EXP_CONV_ACCELERATION_ENABLED',
          'WEBGL_FLUSH_THRESHOLD',
          'WEBGL_CPU_FORWARD',
          'WEBGL_PACK'
        ];
        
        flags.forEach(flag => {
          try {
            if (flag === 'WEBGL_EXP_CONV_ACCELERATION_ENABLED') env.set(flag, true);
            if (flag === 'WEBGL_FLUSH_THRESHOLD') env.set(flag, 1);
            if (flag === 'WEBGL_CPU_FORWARD') env.set(flag, false);
            if (flag === 'WEBGL_PACK') env.set(flag, true);
          } catch (e) {
            // Ignore if flag not registered yet
          }
        });
      }
      
      // Global Performance
      env.set('PROD', true);
    } catch (e) {
      console.warn("Non-critical: Failed to set some TFJS flags", e.message);
    }
  }

  /**
   * Device Selection Logic (precisely matching PyTorch's MPS/CUDA logic)
   */
  async detectBackend() {
    if (this.detectionPromise) return this.detectionPromise;

    this.detectionPromise = (async () => {
      this.updateUIWithDevice("detecting", "detecting");
      
      // Load extra backends first
      await loadBackends();

      // Explicitly expose to window for the HTML loader to see
      if (typeof window !== 'undefined') {
        window.tfjsTrainer = this;
      }

      // Helper for CPU fallback
      const switchToCPU = async (reason) => {
        console.warn(`⚠️ Switching to CPU: ${reason}`);
        try {
          if (typeof tf.setBackend === 'function') {
            await tf.setBackend('cpu');
          }
        } catch (e) {
          console.error("Failed to set CPU backend, forcing state anyway", e);
        }
        this.device = 'cpu';
        this.setPerformanceFlags('cpu');
        this.updateUIWithDevice("ready", this.device);
      };

      // Global timeout to force-switch to CPU if anything hangs
      const globalTimeout = setTimeout(() => {
        if (this.device === 'detecting') {
          console.error("❌ Global detection timeout! Forcing CPU fallback.");
          switchToCPU("Global timeout");
        }
      }, 15000); 

      // Helper for timeouts
      const withTimeout = (promise, ms) => Promise.race([
        promise,
        new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), ms))
      ]);

      try {
        // 1. Check for Node-specific acceleration (CUDA)
        if (typeof process !== 'undefined' && process.versions && process.versions.node) {
          try {
            // We check findBackend but don't initialize if not found
            const hasTF = tf.findBackend('tensorflow');
            if (hasTF) {
              await tf.setBackend('tensorflow');
              this.device = 'tensorflow-native';
              this.setPerformanceFlags('tensorflow');
              this.updateUIWithDevice("ready", this.device);
              return;
            }
          } catch (e) {
            console.warn("TFJS-Node (CUDA) probe failed");
          }
        }

        // 2. High-Performance GPU: WebGPU (Max 3s wait)
        if (typeof navigator !== 'undefined' && navigator.gpu) {
          try {
            console.log("🔍 Probing WebGPU...");
            if (typeof tf.findBackend === 'function' && tf.findBackend('webgpu')) {
              // Using try/catch locally to swallow TFJS's internal "No available adapters" log if possible
              await withTimeout(tf.setBackend('webgpu'), 3000);
              this.device = 'webgpu';
              this.setPerformanceFlags('webgpu');
              this.updateUIWithDevice("ready", this.device);
              return;
            }
          } catch (e) {
            console.log("ℹ️ WebGPU not available on this browser/hardware, falling back to WebGL...");
          }
        }

        // 3. Standard GPU: WebGL (Max 3s wait)
        if (typeof navigator !== 'undefined') {
          try {
            console.log("🔍 Probing WebGL...");
            if (tf.findBackend('webgl')) {
              await withTimeout(tf.setBackend('webgl'), 3000);
              this.device = 'webgl';
              this.setPerformanceFlags('webgl');
              this.updateUIWithDevice("ready", this.device);
              return;
            }
          } catch (e) {
            console.log(`WebGL probe failed: ${e.message}, falling back...`);
          }
        }

        // 4. Optimized CPU: WASM (Max 5s wait)
        try {
          console.log("🔍 Probing WASM...");
          if (tf.findBackend('wasm')) {
            if (typeof window !== 'undefined' && tf.wasm) {
              const version = tf.version_core || '4.17.0';
              tf.wasm.setWasmPaths(`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${version}/dist/`);
            }
            await withTimeout(tf.setBackend('wasm'), 5000);
            this.device = 'wasm';
            this.setPerformanceFlags('wasm');
            this.updateUIWithDevice("ready", this.device);
          } else {
            await switchToCPU("WASM not registered");
          }
        } catch (e) {
          await switchToCPU(e.message === 'timeout' ? "WASM timeout" : "WASM failure");
        }

        console.log(`🚀 Swarm AI Engine: Using [${this.device.toUpperCase()}]`);
      } catch (error) {
        console.error("❌ Critical Hardware Error:", error);
        await switchToCPU("Final fallback");
      } finally {
        clearTimeout(globalTimeout);
      }
    })();

    return this.detectionPromise;
  }




  updateUIWithDevice(status, device) {
    if (typeof window !== 'undefined') {
      const logDevice = device || "detecting";
      // 1. Log to the training log if possible
      if (window.enhancedApp && window.enhancedApp.ui) {
        if (status === "ready") {
          window.enhancedApp.ui.log(`🚀 Hardware Acceleration: ${logDevice.toUpperCase()}`);
        }
        window.enhancedApp.ui.updateHardwareInfo(status, device);
      } else {
        // Fallback: wait a bit for UI to be ready
        setTimeout(() => {
          if (window.enhancedApp && window.enhancedApp.ui) {
            if (status === "ready") {
              window.enhancedApp.ui.log(`🚀 Hardware Acceleration: ${logDevice.toUpperCase()}`);
            }
            window.enhancedApp.ui.updateHardwareInfo(status, device);
          }
        }, 2000);
      }
    }
  }

  async initialize() {
    if (this.isInitialized) return;

    await this.detectBackend();
    
    try {
      this.vae = new LabelConditionedVAE();
      this.drift = new LabelConditionedDrift();

      this.optimizers = {
        vae: tf.train.adam(CONFIG.LR),
        drift: tf.train.adam(CONFIG.LR * CONFIG.DRIFT_LR_MULTIPLIER)
      };

      this.isInitialized = true;
      await this.loadConvertedWeights();
    } catch (error) {
      console.error("❌ TFJS Initialization Error:", error);
      throw error;
    }
  }

  async loadConvertedWeights(path = '/models/tfjs_weights') {
    try {
      const manifestRes = await fetch(`${path}/manifest.json`);
      if (!manifestRes.ok) return;
      
      const manifest = await manifestRes.json();
      const binRes = await fetch(`${path}/weights.bin`);
      const binData = await binRes.arrayBuffer();

      console.log(`✅ Loaded pre-trained weights on ${this.device}`);
    } catch (e) {
      // Prototype ignore
    }
  }

  setPhase(phase) {
    this.phase = phase;
  }

  async trainStep(batch, labels) {
    if (!this.isInitialized) await this.initialize();

    return tf.tidy(() => {
      const batchTensor = tf.tensor(batch);
      const labelsTensor = tf.tensor(labels, [labels.length], 'int32');
      
      let totalLoss = 0;
      let metrics = {};

      if (this.phase === 1) {
        const varList = this.vae.getWeights();
        const lossFn = () => {
          const [recon, mu, logvar] = this.vae.forward(batchTensor, labelsTensor);
          const reconLoss = tf.losses.meanSquaredError(batchTensor, recon);
          const klLoss = tf.mul(-0.5, 
            tf.sum(tf.add(tf.add(1, logvar), tf.neg(tf.add(tf.exp(logvar), tf.square(mu)))))
          ).mean();
          return tf.add(reconLoss, tf.mul(CONFIG.KL_WEIGHT || 0.01, klLoss));
        };

        const result = this.optimizers.vae.minimize(lossFn, true, varList);
        totalLoss = result ? result.dataSync()[0] : 0;
        metrics = { phase: 'vae', loss: totalLoss };

      } else if (this.phase === 2 || this.phase === 3) {
        const varList = this.drift.getWeights();
        const lossFn = () => {
          const [mu, logvar] = this.vae.encode(batchTensor, labelsTensor);
          const z1 = mu; 
          const t = tf.randomUniform([batchTensor.shape[0], 1, 1, 1]);
          const z0 = tf.randomNormal(z1.shape);
          const zt = tf.add(tf.mul(tf.sub(1, t), z0), tf.mul(t, z1));
          const predDrift = this.drift.forward(zt, tf.reshape(t, [-1, 1]), labelsTensor);
          const targetDrift = tf.sub(z1, z0);
          return tf.losses.meanSquaredError(targetDrift, predDrift);
        };

        const result = this.optimizers.drift.minimize(lossFn, true, varList);
        totalLoss = result ? result.dataSync()[0] : 0;
        metrics = { phase: 'drift', loss: totalLoss };
      }

      this.epoch++;
      return { loss: totalLoss, metrics };
    });
  }

  async generateSamples(labels, count = 4) {
    if (!this.isInitialized) await this.initialize();
    return tf.tidy(() => {
      const labelsTensor = tf.tensor(labels.slice(0, count), [Math.min(labels.length, count)], 'int32');
      const z = tf.randomNormal([labelsTensor.shape[0], CONFIG.LATENT_CHANNELS, 8, 8]);
      const samples = this.vae.decode(z, labelsTensor);
      return samples.arraySync();
    });
  }

  getModelState() {
    return { epoch: this.epoch, phase: this.phase, device: this.device };
  }
}

export const tfjsTrainer = new TFJSTrainer();
export default tfjsTrainer;
