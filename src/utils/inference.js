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
    try {
      const response = await fetch("/models/checkpoint_web.json");
      if (response.ok) {
        const checkpoint = await response.json();
        console.log(
          `📊 Loaded checkpoint metadata: Epoch ${checkpoint.metadata.epoch}`,
        );

        // Update inference config based on checkpoint
        if (checkpoint.config) {
          this.config.imgSize = checkpoint.config.IMG_SIZE || 96;
        }

        return checkpoint;
      } else {
        console.warn(
          `⚠️ Checkpoint file not found (status ${response.status}), using untrained models.`,
        );
      }
    } catch (error) {
      console.warn(
        "⚠️ Could not load checkpoint_web.json, using default weights.",
        error,
      );
    }
    return null;
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
    const steps = config.steps || 50;
    const numClasses = CONFIG.NUM_CLASSES || 11;
    const nullClass = numClasses - 1; // NULL class (last index) for CFG.

    // Resolve and validate the label. An out-of-range label would index the
    // embedding's Gather out of bounds and produce garbage/NaN.
    let label;
    if (config.label !== undefined && config.label !== null) {
      const l = Number(config.label);
      label =
        Number.isInteger(l) && l >= 0 && l < nullClass
          ? l
          : Math.floor(Math.random() * nullClass);
    } else {
      label = Math.floor(Math.random() * nullClass);
    }

    // Classifier-free guidance scale (0/1 disables guidance).
    const cfgScale = Number.isFinite(config.cfgScale) ? config.cfgScale : 1.0;
    const useCFG = cfgScale > 1.0;
    // Diffusion coefficient for the noise term: g(t)·sqrt(dt)·N(0,1). We use the
    // OU sigma as the base schedule and let `temperature` scale it, rather than
    // replacing the diffusion coefficient with the temperature outright.
    const gBase = CONFIG.OU_SIGMA || Math.SQRT2;

    const latentShape = [
      1,
      CONFIG.LATENT_H,
      CONFIG.LATENT_W,
      CONFIG.LATENT_CHANNELS,
    ];

    // 1. Initial Latent (Noise) - [1, 12, 12, 8]
    let zt = tf.randomNormal(latentShape);
    const labelsTensor = tf.tensor([label], [1], "int32");
    const nullTensor = tf.tensor([nullClass], [1], "int32");

    // 2. Iterative Drift updates (Forward Bridge Generation: Noise -> Data)
    const dt = 1.0 / steps;
    for (let step = 0; step < steps; step++) {
      const t = step / steps;

      const nextZt = tf.tidy(() => {
        const tTensor = tf.tensor([[t]]);
        // Conditional drift, optionally blended with the unconditional drift
        // via classifier-free guidance: u = u_uncond + s·(u_cond − u_uncond).
        const condDrift = this.drift.forward(zt, tTensor, labelsTensor);
        let predDrift = condDrift;
        if (useCFG) {
          const uncondDrift = this.drift.forward(zt, tTensor, nullTensor);
          predDrift = tf.add(
            uncondDrift,
            tf.mul(tf.sub(condDrift, uncondDrift), cfgScale),
          );
        }
        // Update zt (Euler–Maruyama step)
        let res = tf.add(zt, tf.mul(predDrift, dt));

        // Diffusion noise: g(t)·sqrt(dt)·N(0,1), scaled by temperature.
        if (config.temperature > 0 && step < steps - 1) {
          const noiseScale = config.temperature * gBase * Math.sqrt(dt);
          const noise = tf.randomNormal(latentShape).mul(noiseScale);
          res = tf.add(res, noise);
        }
        return res;
      });

      zt.dispose();
      zt = nextZt;
    }

    // 3. Final Decode
    const decoded = tf.tidy(() => this.vae.decode(zt, labelsTensor));

    // 4. Convert to Image (Canvas). squeeze() allocates a tensor that must be
    // disposed explicitly (it escapes the tidy above).
    const squeezed = decoded.squeeze();
    const pixels = await squeezed.array();
    const image = this.arrayToDataURL(pixels);

    // Cleanup
    tf.dispose([zt, labelsTensor, nullTensor, decoded, squeezed]);

    return {
      id: `sample_${Date.now()}_${index}`,
      image,
      metadata: { label, steps, temperature: config.temperature, cfgScale },
    };
  }

  dispose() {
    if (this.vae) this.vae.dispose();
    if (this.drift) this.drift.dispose();
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

  async generateWithLabel(label, options = {}) {
    console.log(`🔢 Generating label-conditioned samples with label ${label}`);
    return this.generateSamples({
      ...options,
      label: label,
    });
  }

  async generateWithPrompt(prompt, options = {}) {
    console.log(
      `📝 Generating text-conditioned samples with prompt: "${prompt}"`,
    );
    // For now, treat as unconditional since text conditioning not implemented
    // TODO: integrate neural tokenizer for text conditioning
    return this.generateSamples({
      ...options,
      label: undefined,
    });
  }

  async generateUnconditional(options = {}) {
    console.log(`🎲 Generating unconditional samples`);
    return this.generateSamples({
      ...options,
      label: undefined,
    });
  }
}
