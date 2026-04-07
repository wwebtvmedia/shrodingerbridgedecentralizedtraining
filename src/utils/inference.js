import { torch } from 'js-pytorch';
import { CONFIG } from "../config.js";
import { LabelConditionedVAE, LabelConditionedDrift } from "../torchjs/models.js";

export class InferenceEngine {
  constructor() {
    this.isInitialized = false;

    // Inference configuration
    this.config = {
      steps: 50,
      temperature: 0.7,
      cfgScale: 1.0,
      method: "euler",
      seed: null,
    };

    // Inference state
    this.currentInference = null;
    this.inferenceHistory = [];

    // Models
    this.vae = new LabelConditionedVAE();
    this.drift = new LabelConditionedDrift();
    this.torch = torch;
  }

  async initialize() {
    if (this.isInitialized) return;

    console.log("🔮 Initializing Inference Engine (js-pytorch)...");

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
        if (checkpoint.vae_params) {
          const params = this.vae.parameters();
          checkpoint.vae_params.forEach((data, i) => {
            if (params[i]) params[i]._data = data;
          });
        }
        if (checkpoint.drift_params) {
          const params = this.drift.parameters();
          checkpoint.drift_params.forEach((data, i) => {
            if (params[i]) params[i]._data = data;
          });
        }
        console.log(`📂 Loaded weights from checkpoint`);
      }
    } catch (error) {
      console.log("Using default initialized weights");
    }
  }

  async generateSamples(options = {}) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const config = { ...this.config, ...options };
    const sampleCount = config.sampleCount || 4;

    console.log(`🎨 Generating ${sampleCount} samples with real model...`);

    const samples = [];
    for (let i = 0; i < sampleCount; i++) {
      const sample = await this.generateSampleWithSB(config, i);
      samples.push(sample);
    }

    this.inferenceHistory.push({
      timestamp: Date.now(),
      sampleCount,
      config
    });

    return { 
      samples, 
      inference: { 
        duration: 0,
        id: `inf_${Date.now()}`
      } 
    };
  }

  async generateSampleWithSB(config, index) {
    const steps = config.steps || 50;
    const label = config.label !== undefined ? config.label : Math.floor(Math.random() * 10);
    
    return torch.tidy(() => {
      // 1. Initial Latent (Noise) - Flattened 64
      const latentShape = [1, 64];
      let zt = torch.randn(latentShape);
      
      const labelsTensor = torch.tensor([[label]]);

      // 2. Iterative Drift updates (Reverse SB)
      for (let step = 0; step < steps; step++) {
        const t = (steps - step) / steps;
        const tTensor = torch.tensor([[t]]);
        
        // Compute drift
        const predDrift = this.drift.forward(zt, tTensor, labelsTensor);
        
        // Update zt (Euler step)
        const dt = 1.0 / steps;
        zt = torch.add(zt, torch.mul(predDrift, dt));
        
        // Add temperature-scaled noise
        if (config.temperature > 0 && step < steps - 1) {
          const noise = torch.mul(torch.randn(latentShape), config.temperature * Math.sqrt(dt));
          zt = torch.add(zt, noise);
        }
      }

      // 3. Final Decode
      const decoded = this.vae.decode(zt, labelsTensor);
      
      // 4. Convert to Image (Canvas)
      const image = this.tensorToDataURL(decoded);

      return {
        id: `sample_${Date.now()}_${index}`,
        image,
        metadata: { label, steps, temperature: config.temperature }
      };
    });
  }

  tensorToDataURL(tensor) {
    const data = tensor._data[0];
    const width = 32;
    const height = 32;

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(width, height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const pixelIdx = y * width + x;
        const r = Math.floor((data[pixelIdx] || 0) * 255); // Use 0-1 range from ReLU
        const g = Math.floor((data[pixelIdx + 1024] || 0) * 255);
        const b = Math.floor((data[pixelIdx + 2048] || 0) * 255);
        
        imgData.data[idx] = Math.max(0, Math.min(255, r));
        imgData.data[idx + 1] = Math.max(0, Math.min(255, g));
        imgData.data[idx + 2] = Math.max(0, Math.min(255, b));
        imgData.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
    return canvas.toDataURL();
  }

  getInferenceStats() {
    if (this.inferenceHistory.length === 0) return null;
    return {
      totalInferences: this.inferenceHistory.length,
      totalSamples: this.inferenceHistory.reduce((sum, h) => sum + h.sampleCount, 0)
    };
  }
}
