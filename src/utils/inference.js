// Enhanced Schrödinger Bridge Inference Engine
// Ported to TensorFlow.js to match CNN architecture (96x96)

import * as tf from "@tensorflow/tfjs";
import { CONFIG } from "../config.js";
import {
  LabelConditionedVAE,
  LabelConditionedDrift,
} from "../torchjs/models.js";

export class InferenceEngine {
  constructor() {
    this.isInitialized = false;

    // Inference configuration
    this.config = {
      steps: 50,
      temperature: 0.7,
      cfgScale: CONFIG.CFG_SCALE || 3.0,
      method: "euler",
      seed: null,
    };

    // Inference state
    this.currentInference = null;
    this.inferenceHistory = [];

    // Models
    this.vae = new LabelConditionedVAE();
    this.drift = new LabelConditionedDrift();
  }

  async initialize() {
    if (this.isInitialized) return;

    console.log("🔮 Initializing Inference Engine (TensorFlow.js)...");
    await tf.ready();

    // Load models from checkpoint if available
    await this.loadModelsFromCheckpoint();

    this.isInitialized = true;
    console.log("✅ Inference Engine initialized");
  }

  async loadModelsFromCheckpoint() {
    // Simplified loader - in real app would use tf.loadLayersModel or similar
    console.log("Using default initialized weights (Real CNN architecture)");
  }

  async generateSamples(options = {}) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const config = { ...this.config, ...options };
    const sampleCount = config.sampleCount || 4;

    console.log(
      `🎨 Generating ${sampleCount} samples with real SB model (CNN)...`,
    );

    const samples = [];
    for (let i = 0; i < sampleCount; i++) {
      const sample = await this.generateSampleWithSB(config, i);
      samples.push(sample);
    }

    this.inferenceHistory.push({
      timestamp: Date.now(),
      sampleCount,
      config,
    });

    return {
      samples,
      inference: {
        duration: 0,
        id: `inf_${Date.now()}`,
      },
    };
  }

  async generateSampleWithSB(config, index) {
    return tf.tidy(() => {
      const steps = config.steps || 50;
      const label =
        config.label !== undefined
          ? config.label
          : Math.floor(Math.random() * 10);

      // 1. Initial Latent (Noise) - [1, 12, 12, 8]
      const latentShape = [
        1,
        CONFIG.LATENT_H,
        CONFIG.LATENT_W,
        CONFIG.LATENT_CHANNELS,
      ];
      let zt = tf.randomNormal(latentShape);

      const labelsTensor = tf.tensor([label], [1], "int32");

      // 2. Iterative Drift updates (Forward Bridge Generation: Noise -> Data)
      const dt = 1.0 / steps;
      for (let step = 0; step < steps; step++) {
        const t = step / steps;
        const tTensor = tf.tensor([[t]]);

        // Compute drift
        const predDrift = this.drift.forward(zt, tTensor, labelsTensor);

        // Update zt (Euler step)
        zt = tf.add(zt, tf.mul(predDrift, dt));

        // Add temperature-scaled noise
        if (config.temperature > 0 && step < steps - 1) {
          const noise = tf
            .randomNormal(latentShape)
            .mul(config.temperature * Math.sqrt(dt));
          zt = tf.add(zt, noise);
        }
      }

      // 3. Final Decode
      const decoded = this.vae.decode(zt, labelsTensor);

      // 4. Convert to Image (Canvas)
      const pixels = decoded.squeeze().arraySync();
      const image = this.arrayToDataURL(pixels);

      return {
        id: `sample_${Date.now()}_${index}`,
        image,
        metadata: { label, steps, temperature: config.temperature },
      };
    });
  }

  arrayToDataURL(pixels) {
    const imgSize = CONFIG.IMG_SIZE || 96;
    const canvas = document.createElement("canvas");
    canvas.width = imgSize;
    canvas.height = imgSize;
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(imgSize, imgSize);

    // pixels is [H, W, C]
    for (let y = 0; y < imgSize; y++) {
      for (let x = 0; x < imgSize; x++) {
        const i = (y * imgSize + x) * 4;
        const p = pixels[y][x];

        const r = Math.floor(((p[0] || 0) + 1) * 127.5);
        const g = Math.floor(((p[1] || 0) + 1) * 127.5);
        const b = Math.floor(((p[2] || 0) + 1) * 127.5);

        imgData.data[i] = Math.max(0, Math.min(255, r));
        imgData.data[i + 1] = Math.max(0, Math.min(255, g));
        imgData.data[i + 2] = Math.max(0, Math.min(255, b));
        imgData.data[i + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
    return canvas.toDataURL();
  }

  getInferenceStats() {
    if (this.inferenceHistory.length === 0) return null;
    return {
      totalInferences: this.inferenceHistory.length,
      totalSamples: this.inferenceHistory.reduce(
        (sum, h) => sum + h.sampleCount,
        0,
      ),
    };
  }
}
