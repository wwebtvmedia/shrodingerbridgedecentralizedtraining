import { v4 as uuidv4 } from "uuid";
import { ModelManager } from "./models.js";
import { CONFIG } from "../config.js";

class SwarmTrainer {
  constructor(network, phaseManager) {
    this.id = uuidv4();
    this.network = network;
    this.phaseManager = phaseManager;
    this.modelManager = new ModelManager();

    // Training state
    this.currentEpoch = 0;
    this.currentPhase = "auto";
    this.isTraining = false;
    this.explorationRate = 0.3;

    // Metrics
    this.lossHistory = [];
    this.metricsHistory = [];
    this.modelsEvaluated = 0;
    this.syncCount = 0;

    // Callbacks
    this.onEpochComplete = null;
    this.onModelShared = null;
    this.onModelAdopted = null;

    // Training interval
    this.trainingInterval = null;
  }

  async start() {
    if (this.isTraining) return;

    console.log("🚀 Starting swarm trainer...");
    this.isTraining = true;

    // Initialize model
    await this.modelManager.initialize();

    // Start training loop
    this.trainingLoop();

    // Start gossip for results
    this.startGossip();
  }

  async stop() {
    if (!this.isTraining) return;

    console.log("🛑 Stopping swarm trainer...");
    this.isTraining = false;

    // Clear intervals
    if (this.trainingInterval) {
      clearInterval(this.trainingInterval);
      this.trainingInterval = null;
    }

    // Save checkpoint
    await this.saveCheckpoint();
  }

  async trainingLoop() {
    while (this.isTraining) {
      // Determine phase
      if (this.currentPhase === "auto") {
        this.currentPhase = this.phaseManager.determinePhase(this.currentEpoch);
      }

      // Train one epoch
      const { loss, metrics } = await this.trainEpoch();

      // Update history
      this.lossHistory.push({ epoch: this.currentEpoch, loss });
      this.metricsHistory.push({ epoch: this.currentEpoch, ...metrics });

      // Notify UI
      if (this.onEpochComplete) {
        this.onEpochComplete(this.currentEpoch, loss, metrics);
      }

      // Share results with swarm (gossip)
      await this.shareResults(loss, metrics);

      // Check for model synchronization
      await this.checkForSynchronization();

      // Increment epoch
      this.currentEpoch++;

      // Save checkpoint periodically
      if (this.currentEpoch % 10 === 0) {
        await this.saveCheckpoint();
      }

      // Generate samples periodically
      if (this.currentEpoch % 20 === 0) {
        await this.generateAndShareSamples();
      }

      // Small delay to prevent blocking
      await this.sleep(100);
    }
  }

  async trainEpoch() {
    // Generate dummy training data for this epoch
    // In a real application, this would come from a dataset loader
    const batchSize = CONFIG.BATCH_SIZE || 2;
    const dummyBatch = [];
    const imgSize = CONFIG.IMG_SIZE || 96;

    for (let i = 0; i < batchSize; i++) {
      // Generate flat array of size IMG_SIZE * IMG_SIZE * 3
      const pixels = new Array(imgSize * imgSize * 3)
        .fill(0)
        .map(() => Math.random() * 2 - 1);
      dummyBatch.push(pixels);
    }
    const dummyLabels = new Array(batchSize)
      .fill(0)
      .map(() => Math.floor(Math.random() * CONFIG.NUM_CLASSES));

    // Train using model manager
    const result = await this.modelManager.trainStep(
      dummyBatch,
      dummyLabels,
      this.currentPhase === "auto"
        ? this.phaseManager.determinePhase(this.currentEpoch)
        : this.currentPhase,
    );

    const metrics = {
      diversity: result.metrics.kl_loss
        ? 1 / (1 + result.metrics.kl_loss)
        : 0.5,
      reconstruction: result.metrics.recon_loss || result.loss,
      kl_divergence: result.metrics.kl_loss || 0,
      ...result.metrics,
    };

    return { loss: result.loss, metrics };
  }

  async shareResults(loss, metrics) {
    const result = {
      type: "TRAINING_RESULT",
      peerId: this.id,
      epoch: this.currentEpoch,
      phase: this.currentPhase,
      loss,
      metrics,
      timestamp: Date.now(),
      modelHash: await this.modelManager.getModelHash(),
    };

    // Share via gossip protocol
    await this.network.gossip(result);

    if (this.onModelShared) {
      this.onModelShared(result);
    }
  }

  async checkForSynchronization() {
    // Get best model from network
    const bestModel = await this.network.getBestModel();

    if (!bestModel) return;

    // Decide whether to synchronize
    const shouldSync = this.shouldSynchronize(bestModel);

    if (shouldSync) {
      await this.synchronizeTo(bestModel);
    }
  }

  shouldSynchronize(bestModel) {
    // Exploration vs exploitation
    if (Math.random() < this.explorationRate) {
      // Exploration: sometimes sync randomly
      return Math.random() < 0.3;
    }

    // Exploitation: sync only if significantly better
    const myLoss =
      this.lossHistory.length > 0
        ? this.lossHistory[this.lossHistory.length - 1].loss
        : 1.0;

    const improvement = (myLoss - bestModel.loss) / myLoss;
    return improvement > 0.15; // 15% improvement threshold
  }

  async synchronizeTo(bestModel) {
    console.log(`🔄 Synchronizing to model from ${bestModel.peerId}`);

    // Request model from peer
    const modelData = await this.network.requestModel(
      bestModel.peerId,
      bestModel.modelHash,
    );

    if (!modelData) {
      console.warn("Failed to get model from peer");
      return;
    }

    // Load the model
    await this.modelManager.loadModel(modelData);

    // Synchronize to the best model's epoch
    this.currentEpoch = bestModel.epoch;

    // Update metrics
    this.syncCount++;
    this.modelsEvaluated++;

    // Notify
    if (this.onModelAdopted) {
      this.onModelAdopted(bestModel.peerId, this.currentEpoch);
    }

    console.log(`✅ Synchronized to epoch ${this.currentEpoch}`);
  }

  evaluateIncomingModel(modelData) {
    this.modelsEvaluated++;

    // Simple evaluation: compare loss
    const myLoss =
      this.lossHistory.length > 0
        ? this.lossHistory[this.lossHistory.length - 1].loss
        : Infinity;

    if (modelData.loss < myLoss * 0.9) {
      // 10% better, consider adopting
      if (Math.random() < 0.5) {
        // 50% chance to adopt
        this.synchronizeTo(modelData).catch(console.error);
      }
    }
  }

  async generateSamples(count = 4) {
    console.log(
      `🎨 Generating ${count} real samples using hardware acceleration...`,
    );

    try {
      // Use the underlying trainer for sample generation
      const { tfjsTrainer } = await import("../torchjs/integration.js");
      const samples = await tfjsTrainer.generateSamples(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        count,
      );

      // Convert flattened arrays back to data URLs for the UI
      return samples.map((pixels) => this.arrayToDataURL(pixels));
    } catch (error) {
      console.error("Sample generation failed:", error);
      return [];
    }
  }

  arrayToDataURL(pixels) {
    const imgSize = CONFIG.IMG_SIZE || 96;
    const canvas = document.createElement("canvas");
    canvas.width = imgSize;
    canvas.height = imgSize;
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(imgSize, imgSize);

    // pixels is [H, W, C] array from tfjs arraySync()
    for (let y = 0; y < imgSize; y++) {
      for (let x = 0; x < imgSize; x++) {
        const i = (y * imgSize + x) * 4;
        const p = pixels[y][x]; // [R, G, B]

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

  async generateAndShareSamples() {
    const samples = await this.generateSamples(2);

    // Share samples with network
    await this.network.shareSamples({
      peerId: this.id,
      epoch: this.currentEpoch,
      samples,
      timestamp: Date.now(),
    });
  }

  async saveCheckpoint() {
    const checkpoint = {
      epoch: this.currentEpoch,
      phase: this.currentPhase,
      lossHistory: this.lossHistory,
      metricsHistory: this.metricsHistory,
      modelState: await this.modelManager.getState(),
      timestamp: Date.now(),
    };

    // Save to IndexedDB
    await this.saveToStorage("checkpoint", checkpoint);

    console.log(`💾 Checkpoint saved at epoch ${this.currentEpoch}`);
  }

  async loadCheckpoint() {
    const checkpoint = await this.loadFromStorage("checkpoint");

    if (checkpoint) {
      this.currentEpoch = checkpoint.epoch;
      this.currentPhase = checkpoint.phase;
      this.lossHistory = checkpoint.lossHistory || [];
      this.metricsHistory = checkpoint.metricsHistory || [];

      if (checkpoint.modelState) {
        await this.modelManager.setState(checkpoint.modelState);
      }

      console.log(`📂 Checkpoint loaded from epoch ${this.currentEpoch}`);
      return true;
    }

    return false;
  }

  startGossip() {
    // Start periodic gossip
    setInterval(async () => {
      if (this.isTraining && this.network.peers.size > 0) {
        const latestResult =
          this.lossHistory.length > 0
            ? this.lossHistory[this.lossHistory.length - 1]
            : null;

        if (latestResult) {
          await this.shareResults(
            latestResult.loss,
            this.metricsHistory[this.metricsHistory.length - 1] || {},
          );
        }
      }
    }, 5000); // Every 5 seconds
  }

  setPhase(phase) {
    if (["vae", "drift", "both", "auto"].includes(phase)) {
      this.currentPhase = phase;
    }
  }

  setExplorationRate(rate) {
    this.explorationRate = Math.max(0, Math.min(1, rate));
  }

  // Utility methods
  async saveToStorage(key, value) {
    try {
      localStorage.setItem(`swarm_${key}`, JSON.stringify(value));
    } catch (error) {
      console.warn("Failed to save to storage:", error);
    }
  }

  async loadFromStorage(key) {
    try {
      const data = localStorage.getItem(`swarm_${key}`);
      return data ? JSON.parse(data) : null;
    } catch (error) {
      console.warn("Failed to load from storage:", error);
      return null;
    }
  }

  sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

export { SwarmTrainer };
