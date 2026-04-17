import { tfjsTrainer } from "../torchjs/integration.js";
import { CONFIG } from "../config.js";

export class ModelManager {
  constructor() {
    this.model = null;
    this.optimizer = null;
    this.isInitialized = false;

    // Use configuration from config.js
    this.config = CONFIG;

    // Model state
    this.state = {
      parameters: null,
      hash: null,
      version: "4.0.0",
      torchjs_initialized: false,
    };
  }

  async initialize() {
    if (this.isInitialized) return;

    console.log("🧠 Initializing model manager with js-pytorch...");

    try {
      // Initialize trainer (which now uses js-pytorch)
      await tfjsTrainer.initialize();

      // Access models from the trainer
      if (tfjsTrainer.trainer) {
        this.model = {
          vae: tfjsTrainer.trainer.vae,
          drift: tfjsTrainer.trainer.drift,
        };
        this.isInitialized = true;
        this.state.torchjs_initialized = true;

        // Initial hash
        await this.updateModelHash();

        console.log("✅ Model manager initialized with js-pytorch");
      } else {
        throw new Error("Trainer not initialized");
      }
    } catch (error) {
      console.error("❌ Model initialization failed:", error);
      this.isInitialized = false;
    }
  }

  async trainStep(batch, labels, textBytes = null, phase = 1) {
    if (!this.isInitialized) await this.initialize();

    // Support legacy calls where textBytes might be the phase
    let actualTextBytes = textBytes;
    let actualPhase = phase;

    if (typeof textBytes === "string" || typeof textBytes === "number") {
      actualPhase = textBytes;
      actualTextBytes = null;
    }

    try {
      // Ensure phase is numeric for TorchJS
      let phaseNum = 1;
      if (actualPhase === "vae" || actualPhase === 1) phaseNum = 1;
      else if (actualPhase === "drift" || actualPhase === 2) phaseNum = 2;
      else if (actualPhase === "both" || actualPhase === 3) phaseNum = 3;

      tfjsTrainer.setPhase(phaseNum);

      // Pre-process labels if they are objects (common in this codebase)
      const numericLabels = Array.isArray(labels)
        ? labels.map((l) => (typeof l === "object" ? l.class : l))
        : labels;

      // Validate textBytes format (should be array of arrays if provided)
      let processedTextBytes = actualTextBytes;
      if (
        actualTextBytes &&
        Array.isArray(actualTextBytes) &&
        !Array.isArray(actualTextBytes[0])
      ) {
        // If it's a flat array, wrap it as a single sample batch or assume it's invalid
        console.warn(
          "⚠️  textBytes provided as flat array, wrapping in batch of 1",
        );
        processedTextBytes = [actualTextBytes];
      }

      const result = await tfjsTrainer.trainStep(
        batch,
        numericLabels,
        processedTextBytes,
      );

      // Do NOT update hash every step, it's too expensive (downloads all weights)
      // Only set a dirty flag if needed, or let periodic tasks handle it

      return {
        loss: result.loss,
        metrics: result.metrics,
        hash: this.state.hash,
        usingTorchJS: true,
      };
    } catch (error) {
      console.error("❌ Training step failed:", error);
      throw error;
    }
  }

  async updateModelHash() {
    try {
      // Get real weights for hash
      const sample_weights = await tfjsTrainer.getHashSample();
      // Use small part of weights for stable but unique hash
      const sample = Array.isArray(sample_weights) 
        ? sample_weights.flat().slice(0, 10) 
        : [Math.random()];

      const sum = sample.reduce(
        (a, b) => a + (typeof b === "number" ? b : 0),
        0,
      );
      const avg = sample.length > 0 ? sum / sample.length : 0;

      this.state.hash = `model_${Math.abs(avg).toFixed(8).replace(".", "")}_${Date.now().toString().slice(-4)}`;
    } catch (e) {
      this.state.hash = `model_fallback_${Date.now()}`;
    }
  }

  async getModelHash() {
    if (!this.state.hash) {
      await this.updateModelHash();
    }
    return this.state.hash;
  }

  async getState() {
    // CRITICAL: Get real weights from TorchJS for sharing
    const checkpoint = await tfjsTrainer.saveCheckpoint();

    return {
      parameters: checkpoint, // This contains vae_params and drift_params
      hash: this.state.hash || `gen_${Date.now()}`,
      version: this.state.version,
      config: this.config,
      timestamp: Date.now(),
    };
  }

  async setState(state) {
    if (!state) return;

    try {
      // Load real weights into TorchJS
      // Handle various nesting levels of parameters
      const params = state.parameters?.parameters || state.parameters || state;

      if (params && (params.vae_params || params.drift_params)) {
        await tfjsTrainer.loadCheckpoint(params);
      } else {
        console.warn(
          "⚠️ No valid parameters found in state to load into TorchJS",
        );
      }

      this.state.hash = state.hash || state.modelHash;
      this.state.version = state.version || "4.0.0";

      if (state.config) {
        this.config = { ...this.config, ...state.config };
      }

      console.log(`📥 Model state loaded (hash: ${this.state.hash})`);
    } catch (error) {
      console.error("❌ Failed to set model state:", error);
      throw error; // Rethrow to be caught by loadModel
    }
  }

  async loadModel(modelData) {
    try {
      console.log("📥 Loading model from external source...");

      if (!modelData) {
        throw new Error("Model data is null or undefined");
      }

      // Extract state from model data
      // Handle both direct state and wrapped model data from neighbors
      const state = modelData.parameters || modelData.modelData || modelData;

      // Validate state
      if (
        !state ||
        (!state.hash &&
          !state.modelHash &&
          !state.parameters &&
          !state.vae_params)
      ) {
        console.error("Invalid model data received:", modelData);
        throw new Error("Invalid model data");
      }

      // Load state
      await this.setState(state);

      console.log(`✅ Model loaded: ${this.state.hash}`);
      return true;
    } catch (error) {
      console.error("❌ Failed to load model:", error);
      return false;
    }
  }

  async loadFromPyTorchCheckpoint(checkpointPath = "latest.pt") {
    console.log(`📥 PyTorch .pt load not supported in browser. Use JSON sync.`);
    return false;
  }

  async exportToONNX() {
    console.log("📤 ONNX export not supported in this version.");
    return null;
  }

  dispose() {
    if (tfjsTrainer && tfjsTrainer.dispose) {
      tfjsTrainer.dispose();
    }
  }
}

export const modelManager = new ModelManager();
export default modelManager;
