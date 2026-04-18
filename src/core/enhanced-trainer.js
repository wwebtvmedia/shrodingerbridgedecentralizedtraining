import { LocalDatabase } from "../storage/database.js";
import { CloudflareTunnel } from "../network/tunnel.js";
import { PhaseManager } from "./phase.js";
import { ModelManager } from "./models.js";
import { CONFIG } from "../config.js";
import { Validator } from "../utils/validator.js";

class EnhancedSwarmTrainer {
  constructor(config = {}) {
    this.config = {
      useDatabase: true,
      useTunnel: true,
      tunnelConfig: {},
      syncInterval: 5, // Epochs between sync checks
      explorationRate: 0.3,
      maxNeighbors: 50,
      batchSize: CONFIG.BATCH_SIZE || 16,
      numClasses: CONFIG.NUM_CLASSES || 10,
      ...config,
    };

    // Core components
    this.database = this.config.useDatabase ? new LocalDatabase() : null;
    this.tunnel = this.config.useTunnel
      ? new CloudflareTunnel(this.config.tunnelConfig)
      : null;
    this.phaseManager = new PhaseManager();
    this.modelManager = new ModelManager();

    // Training state
    this.id = this.generateId();
    this.currentEpoch = 0;
    this.currentPhase = "auto";
    this.isTraining = false;
    this.trainingLoop = null;
    this.isPerformingTasks = false;

    // Neighbor management
    this.neighbors = new Map(); // Active neighbors
    this.neighborStats = new Map(); // Statistics per neighbor
    this.pendingModelRequests = new Map(); // hash -> { resolve, reject, timeout }

    // Metrics
    this.metrics = {
      epochsTrained: 0,
      modelsShared: 0,
      modelsReceived: 0,
      syncEvents: 0,
      databaseOperations: 0,
    };

    // Callbacks
    this.callbacks = {
      onEpochComplete: null,
      onNeighborUpdate: null,
      onModelShared: null,
      onModelReceived: null,
      onSyncEvent: null,
      onResearchResult: null,
      onDatabaseUpdate: null,
    };
  }

  generateId() {
    return "trainer_" + Math.random().toString(36).substring(2, 15);
  }

  async initialize() {
    console.log("🚀 Initializing Enhanced Swarm Trainer...");

    // Initialize database
    if (this.database) {
      await this.database.init();
      console.log("✅ Database initialized");

      // Load previous state if available
      await this.loadState();
    }

    // Initialize tunnel
    if (this.tunnel) {
      await this.tunnel.connect();

      // Setup tunnel event listeners
      this.setupTunnelListeners();

      console.log("✅ Cloudflare Tunnel initialized");
    }

    // Initialize model manager
    await this.modelManager.initialize();

    console.log("✅ Enhanced trainer initialized");
    return true;
  }

  async loadState() {
    if (!this.database) return;

    try {
      // Load latest checkpoint
      const checkpoint = await this.database.getLatestCheckpoint();
      if (checkpoint) {
        this.currentEpoch = checkpoint.epoch || 0;
        this.currentPhase = checkpoint.phase || "auto";

        if (checkpoint.modelState) {
          await this.modelManager.setState(checkpoint.modelState);
        }

        console.log(`📂 Loaded state from epoch ${this.currentEpoch}`);
      }

      // Load neighbors from database
      const savedNeighbors = await this.database.getAllNeighbors();
      for (const neighbor of savedNeighbors) {
        this.neighbors.set(neighbor.peerId, neighbor);
      }

      console.log(`📂 Loaded ${savedNeighbors.length} neighbors from database`);
    } catch (error) {
      console.error("Failed to load state:", error);
    }
  }

  setupTunnelListeners() {
    if (!this.tunnel) return;

    this.tunnel.on("peer:connected", (data) => {
      this.handleNeighborConnected(data.peerId, data.metadata);
    });

    this.tunnel.on("peer:disconnected", (data) => {
      this.handleNeighborDisconnected(data.peerId);
    });

    this.tunnel.on("peer:message", (data) => {
      this.handleNeighborMessage(data.from, data.data);
    });

    this.tunnel.on("broadcast:received", (data) => {
      this.handleBroadcastMessage(data.from, data.data);
    });

    this.tunnel.on("peer:research_result", (data) => {
      this.handleResearchResponse(data.from, data.data);
    });

    this.tunnel.on("connected", () => {
      console.log("✅ Tunnel connected");
      this.broadcastPresence();
    });

    this.tunnel.on("model:initial", (modelInfo) => {
      console.log(`🏆 Server pushed best model from epoch ${modelInfo.epoch}`);
    });

    this.tunnel.on("disconnected", () => {
      console.log("⚠️ Tunnel disconnected");
    });

    this.tunnel.on("error", (data) => {
      console.error("Tunnel error:", data.error);
    });
  }

  async startTraining() {
    if (this.isTraining) return;

    console.log("🧠 Starting enhanced swarm training...");
    this.isTraining = true;

    // Start training loop
    this.runTrainingLoop().catch(console.error);

    // Start periodic tasks
    this.startPeriodicTasks();

    console.log("✅ Training started");
  }

  async runTrainingLoop() {
    while (this.isTraining) {
      try {
        await this.trainStep();
        // Small delay between epochs to prevent blocking the main thread
        await new Promise((resolve) => setTimeout(resolve, 100));
      } catch (error) {
        console.error("Error in training loop:", error);
        // Pause on error
        await new Promise((resolve) => setTimeout(resolve, 5000));
      }
    }
  }

  async stopTraining() {
    if (!this.isTraining) return;

    console.log("🛑 Stopping training...");
    this.isTraining = false;

    // Final sync to server before closing
    if (this.tunnel) {
      const stats = this.getMetrics();
      try {
        await this.tunnel.sendFinalSync(stats, this.getNeighbors());
      } catch (e) {
        console.warn("⚠️ Final sync failed:", e.message);
      }
    }

    // Intervals are no longer used for training loop, but we still have periodic tasks
    if (this.periodicTasks) {
      clearInterval(this.periodicTasks);
      this.periodicTasks = null;
    }

    // Save final state
    await this.saveState();
  }

  dispose() {
    this.isTraining = false;
    if (this.periodicTasks) {
      clearInterval(this.periodicTasks);
      this.periodicTasks = null;
    }
    if (this.modelManager && this.modelManager.dispose) {
      this.modelManager.dispose();
    }
  }

  async trainStep() {
    if (!this.isTraining) return;

    // Determine phase
    if (this.currentPhase === "auto") {
      this.currentPhase = this.phaseManager.determinePhase(this.currentEpoch);
    }

    const targetBatchSize = this.config.batchSize || 16;
    let batch = [];
    let labels = [];

    // 1. Fetch data from database
    if (this.database) {
      const trainingData = await this.database.getTrainingData(targetBatchSize);
      if (trainingData && trainingData.length > 0) {
        batch = trainingData.map((item) => {
          return item.data && Array.isArray(item.data)
            ? item.data
            : this.generateDummyData();
        });
        labels = trainingData.map((item) => item.metadata?.label || 0);
      }
    }

    // 2. Padding logic: ALWAYS ensure batch size is exactly targetBatchSize
    while (batch.length < targetBatchSize) {
      batch.push(this.generateDummyData());
      labels.push(Math.floor(Math.random() * (this.config.numClasses || 10)));
    }

    // 3. Train epoch
    const { loss, metrics } = await this.modelManager.trainStep(
      batch,
      labels,
      this.currentPhase,
    );

    // Update epoch
    this.currentEpoch++;
    this.metrics.epochsTrained++;

    // Save result to database
    if (this.database) {
      await this.database.saveResult({
        epoch: this.currentEpoch,
        phase: this.currentPhase,
        loss,
        metrics,
        timestamp: Date.now(),
      });
      this.metrics.databaseOperations++;
    }

    // Share result with neighbors
    await this.shareTrainingResult(loss, metrics);

    // Check for synchronization
    if (this.currentEpoch % this.config.syncInterval === 0) {
      await this.checkForSynchronization();
    }

    // Save checkpoint periodically
    if (this.currentEpoch % 10 === 0) {
      await this.saveCheckpoint();
    }

    // Call callback
    if (this.callbacks.onEpochComplete) {
      this.callbacks.onEpochComplete(this.currentEpoch, loss, metrics);
    }
  }

  generateDummyData() {
    const imgSize = CONFIG.IMG_SIZE || 96;
    const size = 3 * imgSize * imgSize;
    const data = new Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.random() * 2 - 1;
    }
    return data;
  }

  async shareTrainingResult(loss, metrics) {
    const result = {
      type: "TRAINING_RESULT",
      trainerId: this.id,
      epoch: this.currentEpoch,
      phase: this.currentPhase,
      loss,
      metrics,
      modelHash: await this.modelManager.getModelHash(),
      timestamp: Date.now(),
    };

    if (this.database) {
      await this.database.saveNeighbor({
        peerId: this.id,
        ...result,
      });
    }

    if (this.tunnel) {
      await this.tunnel.broadcast(result);
    }

    this.metrics.modelsShared++;

    if (this.callbacks.onModelShared) {
      this.callbacks.onModelShared(result);
    }
  }

  async checkForSynchronization() {
    if (!this.database) return;

    const bestNeighbors = await this.database.getBestNeighbors(5);
    if (bestNeighbors.length === 0) return;

    const myResults = await this.database.getRecentResults(1);
    const myLoss = myResults.length > 0 ? myResults[0].loss : Infinity;

    const bestNeighbor = bestNeighbors[0];
    const shouldSync = this.shouldSyncWithNeighbor(bestNeighbor, myLoss);

    if (shouldSync) {
      await this.synchronizeWithNeighbor(bestNeighbor);
    }
  }

  shouldSyncWithNeighbor(neighbor, myLoss) {
    if (Math.random() < this.config.explorationRate) {
      return Math.random() < 0.3;
    }

    if (neighbor.loss === undefined || myLoss === Infinity) {
      return false;
    }

    const improvement = (myLoss - neighbor.loss) / myLoss;
    return improvement > 0.15;
  }

  async synchronizeWithNeighbor(neighbor) {
    console.log(`🔄 Synchronizing with neighbor ${neighbor.peerId}`);

    const modelData = await this.requestModelFromNeighbor(
      neighbor.peerId,
      neighbor.modelHash,
    );

    if (!modelData) return;

    const success = await this.modelManager.loadModel(modelData);
    if (!success) return;

    // Synchronize to the neighbor's actual epoch
    this.currentEpoch = neighbor.epoch || 0;

    this.metrics.syncEvents++;
    this.metrics.modelsReceived++;

    if (this.database) {
      await this.database.saveResult({
        type: "SYNC_EVENT",
        epoch: this.currentEpoch,
        sourcePeer: neighbor.peerId,
        targetEpoch: randomEpoch,
        timestamp: Date.now(),
      });
    }

    if (this.callbacks.onSyncEvent) {
      this.callbacks.onSyncEvent(neighbor.peerId, randomEpoch);
    }
  }

  async requestModelFromNeighbor(peerId, modelHash) {
    if (!this.tunnel) return null;

    console.log(`📡 Requesting model ${modelHash} from ${peerId}`);

    return new Promise((resolve, reject) => {
      // Set timeout
      const timeout = setTimeout(() => {
        if (this.pendingModelRequests.has(modelHash)) {
          this.pendingModelRequests.delete(modelHash);
          reject(new Error("Model request timed out"));
        }
      }, 30000); // 30 second timeout

      this.pendingModelRequests.set(modelHash, { resolve, reject, timeout });

      this.tunnel
        .sendToPeer(peerId, {
          type: "MODEL_REQUEST",
          modelHash: modelHash,
          timestamp: Date.now(),
        })
        .catch((err) => {
          clearTimeout(timeout);
          this.pendingModelRequests.delete(modelHash);
          reject(err);
        });
    });
  }

  handleNeighborConnected(peerId, metadata) {
    console.log(`👋 Neighbor connected: ${peerId}`);

    this.neighbors.set(peerId, {
      peerId,
      ...metadata,
      connectedAt: Date.now(),
      lastSeen: Date.now(),
    });

    if (this.database) {
      this.database
        .saveNeighbor({
          peerId,
          ...metadata,
          connectedAt: Date.now(),
        })
        .catch(console.error);
    }

    if (this.callbacks.onNeighborUpdate) {
      this.callbacks.onNeighborUpdate("connected", peerId, metadata);
    }
  }

  handleNeighborDisconnected(peerId) {
    console.log(`👋 Neighbor disconnected: ${peerId}`);
    this.neighbors.delete(peerId);
    if (this.callbacks.onNeighborUpdate) {
      this.callbacks.onNeighborUpdate("disconnected", peerId);
    }
  }

  handleNeighborMessage(peerId, message) {
    if (!message || !message.type) return;

    // Strict Whitelisting Validation
    if (!Validator.validate(message.type, message)) {
      console.warn(
        `Trainer: Rejected malformed or unexpected ${message.type} from ${peerId}`,
      );
      return;
    }

    const neighbor = this.neighbors.get(peerId);
    if (neighbor) {
      neighbor.lastSeen = Date.now();
    }

    switch (message.type) {
      case "TRAINING_RESULT":
        this.handleTrainingResultMessage(peerId, message);
        break;
      case "MODEL_REQUEST":
        this.handleModelRequest(peerId, message);
        break;
      case "MODEL_SHARE":
        this.handleModelShare(peerId, message);
        break;
      case "PEER_RESEARCH_REQUEST":
        this.handleResearchRequest(peerId, message);
        break;
      case "PEER_RESEARCH_RESPONSE":
        this.handleResearchResponse(peerId, message);
        break;
      default:
        console.log(`Received message from ${peerId}:`, message.type);
    }
  }

  handleResearchRequest(peerId, message) {
    if (!this.tunnel) return;
    this.tunnel.sendToPeer(peerId, {
      type: "PEER_RESEARCH_RESPONSE",
      status: {
        trainerId: this.id,
        epoch: this.currentEpoch,
        phase: this.currentPhase,
        metrics: {
          loss: 0.2 + Math.random() * 0.1,
          accuracy: 0.8 + Math.random() * 0.1,
        },
        timestamp: Date.now(),
      },
    });
  }

  handleResearchResponse(peerId, message) {
    if (this.callbacks.onResearchResult) {
      this.callbacks.onResearchResult(peerId, message.status);
    }
  }

  handleTrainingResultMessage(peerId, result) {
    if (this.database) {
      this.database
        .saveNeighbor({
          peerId,
          ...result,
        })
        .catch(console.error);
    }

    this.updateNeighborStats(peerId, result);

    if (this.callbacks.onModelReceived) {
      this.callbacks.onModelReceived(peerId, result);
    }
  }

  handleModelRequest(peerId, request) {
    this.sendModelToPeer(peerId, request.modelHash).catch(console.error);
  }
  handleModelShare(peerId, share) {
    // Resolve pending requests if any
    const pending = this.pendingModelRequests.get(share.modelHash);
    if (pending) {
      clearTimeout(pending.timeout);
      this.pendingModelRequests.delete(share.modelHash);
      pending.resolve(share);
    }

    if (this.database) {
      this.database
        .saveModel({
          hash: share.modelHash,
          peerId,
          data: share.modelData,
          timestamp: Date.now(),
        })
        .catch(console.error);
    }
    if (this.callbacks.onModelReceived) {
      this.callbacks.onModelReceived(peerId, share);
    }
  }

  handleBroadcastMessage(peerId, message) {
    if (!message || !message.type) return;

    // Strict Whitelisting Validation
    if (!Validator.validate(message.type, message)) {
      console.warn(
        `Trainer: Rejected malformed or unexpected broadcast ${message.type} from ${peerId}`,
      );
      return;
    }

    console.log(`📢 Broadcast from ${peerId}:`, message.type);
    if (message.type === "TRAINING_RESULT") {
      this.handleTrainingResultMessage(peerId, message);
    } else if (message.type === "PEER_RESEARCH_REQUEST") {
      this.handleResearchRequest(peerId, message);
    }
  }

  async sendModelToPeer(peerId, modelHash) {
    const modelState = await this.modelManager.getState();
    const message = {
      type: "MODEL_SHARE",
      modelHash,
      modelData: modelState,
      timestamp: Date.now(),
    };

    if (this.tunnel) {
      await this.tunnel.sendToPeer(peerId, message);
    }
  }

  updateNeighborStats(peerId, result) {
    if (!this.neighborStats.has(peerId)) {
      this.neighborStats.set(peerId, {
        resultsReceived: 0,
        bestLoss: Infinity,
        lastEpoch: 0,
        lastUpdate: Date.now(),
      });
    }

    const stats = this.neighborStats.get(peerId);
    stats.resultsReceived++;
    stats.lastUpdate = Date.now();
    stats.lastEpoch = result.epoch || 0;

    if (result.loss !== undefined && result.loss < stats.bestLoss) {
      stats.bestLoss = result.loss;
    }
  }

  async broadcastPresence() {
    if (!this.tunnel) return;
    const presence = {
      type: "PRESENCE",
      trainerId: this.id,
      epoch: this.currentEpoch,
      phase: this.currentPhase,
      timestamp: Date.now(),
    };
    await this.tunnel.broadcast(presence);
  }

  startPeriodicTasks() {
    if (this.periodicTasks) return;
    this.periodicTasks = setInterval(() => {
      this.performPeriodicTasks();
    }, 60000);
  }

  async performPeriodicTasks() {
    if (this.isPerformingTasks) return;
    this.isPerformingTasks = true;
    try {
      if (this.database) {
        await this.database.cleanupOldNeighbors();
      }

      if (this.tunnel) {
        const stats = this.getMetrics();
        await this.tunnel.sendStatusUpdate(stats, this.getNeighbors());
      }

      await this.broadcastPresence();
      await this.saveCheckpoint();

      if (this.callbacks.onDatabaseUpdate && this.database) {
        const stats = await this.database.getStatistics();
        this.callbacks.onDatabaseUpdate(stats);
      }
    } finally {
      this.isPerformingTasks = false;
    }
  }

  async researchNeighbors() {
    if (!this.tunnel) return;
    return this.tunnel.researchNeighbors();
  }

  async saveCheckpoint() {
    if (!this.database) return;
    const checkpoint = {
      epoch: this.currentEpoch,
      phase: this.currentPhase,
      modelState: await this.modelManager.getState(),
      timestamp: Date.now(),
    };
    await this.database.saveCheckpoint(checkpoint);
    console.log(`💾 Checkpoint saved at epoch ${this.currentEpoch}`);
  }

  async saveState() {
    await this.saveCheckpoint();
    if (this.database) {
      const exportData = await this.database.exportData();
      localStorage.setItem("swarm_trainer_backup", exportData);
    }
  }

  getNeighbors() {
    return Array.from(this.neighbors.values());
  }

  getNeighborCount() {
    return this.neighbors.size;
  }

  getMetrics() {
    return {
      ...this.metrics,
      currentEpoch: this.currentEpoch,
      currentPhase: this.currentPhase,
      neighborCount: this.neighbors.size,
      isTraining: this.isTraining,
    };
  }

  getDatabaseStats() {
    if (!this.database) return null;
    return this.database.getStatistics();
  }

  setExplorationRate(rate) {
    this.config.explorationRate = Math.max(0, Math.min(1, rate));
  }

  setPhase(phase) {
    if (["vae", "drift", "both", "auto"].includes(phase)) {
      this.currentPhase = phase;
    }
  }

  onEpochComplete(callback) {
    this.callbacks.onEpochComplete = callback;
  }
  onNeighborUpdate(callback) {
    this.callbacks.onNeighborUpdate = callback;
  }
  onModelShared(callback) {
    this.callbacks.onModelShared = callback;
  }
  onModelReceived(callback) {
    this.callbacks.onModelReceived = callback;
  }
  onSyncEvent(callback) {
    this.callbacks.onSyncEvent = callback;
  }
  onResearchResult(callback) {
    this.callbacks.onResearchResult = callback;
  }
  onDatabaseUpdate(callback) {
    this.callbacks.onDatabaseUpdate = callback;
  }
}

export { EnhancedSwarmTrainer };
