// Enhanced Schrödinger Bridge Inference Engine
// Ported to TensorFlow.js to match CNN architecture (96x96)

import * as tf from "@tensorflow/tfjs";
import { CONFIG } from "../config.js";
import {
  LabelConditionedVAE,
  LabelConditionedDrift,
} from "../torchjs/models.js";
import { globalSwarmKnowledge } from "../network/swarm-knowledge-bridge.js";

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
    const cfgScale = Number.isFinite(config.cfgScale) ? config.cfgScale : (CONFIG.CFG_SCALE || 6.5);
    const method = config.method || "heun";
    const langevinSteps = config.langevinSteps !== undefined ? config.langevinSteps : (CONFIG.DEFAULT_LANGEVIN_STEPS || 0);

    const latentShape = [
      1,
      CONFIG.LATENT_H || 12,
      CONFIG.LATENT_W || 12,
      CONFIG.LATENT_CHANNELS || 8,
    ];

    // 1. Initial Latent (Noise from prior scale)
    let zt = tf.mul(
      tf.randomNormal(latentShape),
      CONFIG.CST_COEF_GAUSSIAN_PRIO || 0.8,
    );
    const labelsTensor = tf.tensor([label], [1], "int32");
    const nullTensor = tf.tensor([nullClass], [1], "int32");

    const evalDrift = (zIn, tIn) => {
      const condDrift = this.drift.forward(zIn, tIn, labelsTensor);
      if (cfgScale > 1.0) {
        const uncondDrift = this.drift.forward(zIn, tIn, nullTensor);
        return tf.add(
          uncondDrift,
          tf.mul(tf.sub(condDrift, uncondDrift), cfgScale),
        );
      }
      return condDrift;
    };

    // 2. Iterative Drift updates (ODE Solver)
    const dt = 1.0 / steps;
    for (let step = 0; step < steps; step++) {
      const tVal = step * dt;

      const nextZt = tf.tidy(() => {
        const tCur = tf.tensor([[tVal]]);
        let zOut;

        if (method === "euler") {
          const k1 = evalDrift(zt, tCur);
          zOut = tf.add(zt, tf.mul(k1, dt));
        } else if (method === "rk4") {
          const k1 = evalDrift(zt, tCur);
          const tHalf = tf.tensor([[tVal + 0.5 * dt]]);
          const zHalf1 = tf.add(zt, tf.mul(k1, 0.5 * dt));
          const k2 = evalDrift(zHalf1, tHalf);
          const zHalf2 = tf.add(zt, tf.mul(k2, 0.5 * dt));
          const k3 = evalDrift(zHalf2, tHalf);
          const tNext = tf.tensor([[tVal + dt]]);
          const zNext = tf.add(zt, tf.mul(k3, dt));
          const k4 = evalDrift(zNext, tNext);

          const rkSum = tf.add(
            tf.add(k1, tf.mul(2.0, k2)),
            tf.add(tf.mul(2.0, k3), k4),
          );
          zOut = tf.add(zt, tf.mul(rkSum, dt / 6.0));
        } else {
          // Heun
          const k1 = evalDrift(zt, tCur);
          const tNext = tf.tensor([[tVal + dt]]);
          const zPred = tf.add(zt, tf.mul(k1, dt));
          const k2 = evalDrift(zPred, tNext);
          zOut = tf.add(zt, tf.mul(tf.add(k1, k2), dt / 2.0));
        }

        const clampLimit = CONFIG.ODE_CLAMP_MAX || 10.0;
        return tf.clipByValue(zOut, -clampLimit, clampLimit);
      });

      zt.dispose();
      zt = nextZt;
    }

    // 2b. Optional Langevin refinement at t=1
    if (langevinSteps > 0) {
      const stepSize = CONFIG.LANGEVIN_STEP_SIZE || 0.01;
      const scoreScale = CONFIG.LANGEVIN_SCORE_SCALE || 1.2;
      const tOne = tf.tensor([[1.0]]);

      for (let s = 0; s < langevinSteps; s++) {
        const nextZ = tf.tidy(() => {
          const driftScore = evalDrift(zt, tOne);
          const noise = tf.randomNormal(zt.shape);
          const stepDelta = tf.mul(driftScore, stepSize * scoreScale);
          const noiseDelta = tf.mul(noise, Math.sqrt(2 * stepSize));
          return tf.add(tf.add(zt, stepDelta), noiseDelta);
        });
        zt.dispose();
        zt = nextZ;
      }
      tOne.dispose();
    }

    // 3. Final Decode
    const decoded = tf.tidy(() => this.vae.decode(zt, labelsTensor));

    // 4. Convert to Image (Canvas).
    const squeezed = decoded.squeeze();
    const pixels = await squeezed.array();
    const image = this.arrayToDataURL(pixels);

    // Cleanup
    tf.dispose([zt, labelsTensor, nullTensor, decoded, squeezed]);

    const promptText = config.prompt || `Class ${label} Schrödinger Bridge Generative Sample`;
    const ospEnvelope = globalSwarmKnowledge.createKnowledgeEnvelope(
      "image/schrodinger-bridge",
      {
        image,
        metadata: { label, steps, method, cfgScale, prompt: promptText },
      },
      {
        prompt: promptText,
        label,
        method,
        cfgScale,
        modelHash: "sb_webgpu_v4",
        citedChunks: config.citedChunks || [
          `Schrödinger Bridge generative trajectory for class ${label}`,
          `ODE solver ${method} with CFG scale ${cfgScale}`,
        ],
      },
    );

    return {
      id: `sample_${Date.now()}_${index}`,
      image,
      metadata: { label, steps, method, cfgScale, prompt: promptText },
      ospEnvelope,
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
