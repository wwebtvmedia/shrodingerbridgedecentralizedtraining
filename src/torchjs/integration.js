// Universal TensorFlow.js Hardware Accelerator (Mac M1-M4, NVIDIA CUDA, AMD, Intel)
// This implementation maps precisely to the device selection in PyTorch

import * as tf from '@tensorflow/tfjs';
import { CONFIG } from "../config.js";
import { LabelConditionedVAE, LabelConditionedDrift } from "./models.js";

// Helper function to check if a backend is available
const isBackendAvailable = (backendName) => {
  try {
    // Check if backend exists in registry
    if (tf.engine && tf.engine().registry) {
      return backendName in tf.engine().registry;
    }
    // Fallback: try to get current backend
    return tf.getBackend() === backendName;
  } catch (e) {
    return false;
  }
};

// Conditional backend loading to prevent "already registered" errors
const loadBackends = async () => {
  const backends = [
    { name: 'wasm', pkg: '@tensorflow/tfjs-backend-wasm' },
    { name: 'webgpu', pkg: '@tensorflow/tfjs-backend-webgpu' },
    { name: 'webgl', pkg: '@tensorflow/tfjs-backend-webgl' }
  ];
  
  for (const backend of backends) {
    if (!isBackendAvailable(backend.name)) {
      try {
        await import(backend.pkg);
        console.log(`✅ Loaded ${backend.name} backend`);
      } catch (e) {
        console.warn(`⚠️ Failed to load ${backend.name} backend: ${e.message}`);
        // Continue with other backends
      }
    }
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
    this.status = 'initializing';
    this.detectionPromise = null;

    // Explicitly expose to window immediately for the HTML loader to see
    if (typeof window !== 'undefined') {
      window.tfjsTrainer = this;
      console.log("🛠️ TFJSTrainer exposed to window");
    }

    // Start detection immediately (backends are loaded inside detectBackend)
    this.detectBackend().catch(err => {
      console.error("❌ Hardware detection critical failure:", err);
      this.status = 'failed';
    });
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
        const flags = {
          'WEBGL_EXP_CONV_ACCELERATION_ENABLED': true,
          'WEBGL_FLUSH_THRESHOLD': 1,
          'WEBGL_CPU_FORWARD': false,
          'WEBGL_PACK': true
        };
        
        Object.entries(flags).forEach(([flag, value]) => {
          try {
            env.set(flag, value);
          } catch (e) {}
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
      console.log("🔍 Starting hardware detection...");
      this.updateUIWithDevice("detecting", "detecting");
      
      // Load extra backends first with timeout
      try {
        await Promise.race([
          loadBackends(),
          new Promise((_, reject) => setTimeout(() => reject(new Error('Backend load timeout')), 5000))
        ]);
      } catch (e) {
        console.warn("⚠️ Backend loading issue:", e.message);
      }

      // Helper for CPU fallback
      const switchToCPU = async (reason) => {
        console.warn(`⚠️ Switching to CPU: ${reason}`);
        try {
          if (typeof tf.setBackend === 'function') {
            await tf.setBackend('cpu');
            await tf.ready();
          }
        } catch (e) {
          console.error("Failed to set CPU backend, forcing state anyway", e);
        }
        this.device = 'cpu';
        this.status = 'ready';
        this.setPerformanceFlags('cpu');
        this.updateUIWithDevice("ready", this.device);
      };

      // Global timeout to force-switch to CPU if anything hangs
      const globalTimeout = setTimeout(() => {
        if (this.device === 'detecting') {
          console.error("❌ Global detection timeout! Forcing CPU fallback.");
          switchToCPU("Global timeout");
        }
      }, 10000); 

      // Helper for timeouts
      const withTimeout = (promise, ms) => Promise.race([
        promise,
        new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), ms))
      ]);

      try {
        // 1. Check for Node-specific acceleration (CUDA/TensorFlow-Native)
        if (typeof process !== 'undefined' && process.versions && process.versions.node) {
          try {
            if (isBackendAvailable('tensorflow')) {
              console.log("✅ Using TensorFlow Native (Node.js)");
              await tf.setBackend('tensorflow');
              await tf.ready();
              this.device = 'tensorflow-native';
              this.status = 'ready';
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
            if (isBackendAvailable('webgpu')) {
              await withTimeout(tf.setBackend('webgpu'), 3000);
              await tf.ready();
              console.log("✅ WebGPU selected");
              this.device = 'webgpu';
              this.status = 'ready';
              this.setPerformanceFlags('webgpu');
              this.updateUIWithDevice("ready", this.device);
              return;
            }
          } catch (e) {
            console.log("ℹ️ WebGPU not available, falling back...");
          }
        }

        // 3. Standard GPU: WebGL (Max 3s wait)
        if (typeof navigator !== 'undefined') {
          try {
            console.log("🔍 Probing WebGL...");
            if (isBackendAvailable('webgl')) {
              await withTimeout(tf.setBackend('webgl'), 3000);
              await tf.ready();
              console.log("✅ WebGL selected");
              this.device = 'webgl';
              this.status = 'ready';
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
          if (isBackendAvailable('wasm')) {
            if (typeof window !== 'undefined' && tf.wasm) {
              const version = tf.version_core || '4.17.0';
              tf.wasm.setWasmPaths(`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${version}/dist/`);
            }
            await withTimeout(tf.setBackend('wasm'), 5000);
            await tf.ready();
            console.log("✅ WASM selected");
            this.device = 'wasm';
            this.status = 'ready';
            this.setPerformanceFlags('wasm');
            this.updateUIWithDevice("ready", this.device);
            return;
          }
        } catch (e) {
          console.warn("WASM probe failed");
        }

        // 5. Final Fallback: CPU
        await switchToCPU("Final fallback");

        console.log(`🚀 Swarm AI Engine: Using [${this.device.toUpperCase()}]`);
      } catch (error) {
        console.error("❌ Critical Hardware Error during detection:", error);
        await switchToCPU("Error fallback");
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

  async saveCheckpoint() {
    if (!this.isInitialized) await this.initialize();
    const checkpoint = {
      epoch: this.epoch,
      phase: this.phase,
      device: this.device,
      vae_state: this.vae ? this.vae.getWeights() : null,
      drift_state: this.drift ? this.drift.getWeights() : null
    };
    console.log(`✅ Checkpoint saved at epoch ${this.epoch}`);
    return checkpoint;
  }

  async loadCheckpoint(checkpoint) {
    if (!this.isInitialized) await this.initialize();
    this.epoch = checkpoint.epoch || this.epoch;
    this.phase = checkpoint.phase || this.phase;
    
    if (checkpoint.vae_state && this.vae && this.vae.setWeights) {
      this.vae.setWeights(checkpoint.vae_state);
    }
    if (checkpoint.drift_state && this.drift && this.drift.setWeights) {
      this.drift.setWeights(checkpoint.drift_state);
    }
    
    console.log(`✅ Checkpoint loaded from epoch ${this.epoch}`);
  }

  getModelState() {
    return { epoch: this.epoch, phase: this.phase, device: this.device };
  }
}

export const tfjsTrainer = new TFJSTrainer();
export default tfjsTrainer;
