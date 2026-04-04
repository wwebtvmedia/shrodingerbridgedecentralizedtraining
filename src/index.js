import { SwarmTrainer } from "./core/trainer.js";
import { PeerNetwork } from "./network/peer.js";
import { UIManager } from "./ui/manager.js";
import { PhaseManager } from "./core/phase.js";

class SwarmTrainingApp {
  constructor() {
    this.ui = new UIManager();
    this.network = new PeerNetwork();
    this.phaseManager = new PhaseManager();
    this.trainer = null;
    this.isTraining = false;

    this.init();
  }

  async init() {
    console.log("🚀 Initializing Swarm Training App...");

    // Initialize UI
    this.ui.init();

    // Setup event listeners
    this.setupEventListeners();

    // Initialize WebTorch
    await this.initWebTorch();

    // Update UI
    this.ui.updateStatus("Ready to connect");
    this.ui.enableButton("connect-btn");

    console.log("✅ App initialized");
  }

  async initWebTorch() {
    try {
      // Check if WebTorch is available
      if (typeof torch === "undefined") {
        throw new Error("WebTorch not loaded. Check CDN connection.");
      }

      // Test WebTorch
      const testTensor = torch.tensor([1, 2, 3]);
      console.log("✅ WebTorch loaded:", testTensor);

      this.ui.log("WebTorch initialized successfully");
    } catch (error) {
      console.error("❌ WebTorch initialization failed:", error);
      this.ui.log(`Error: ${error.message}`);
      this.ui.updateStatus("WebTorch failed to load");
    }
  }

  setupEventListeners() {
    // Connect button
    document
      .getElementById("connect-btn")
      .addEventListener("click", () => this.connectToSwarm());

    // Start training button
    document
      .getElementById("start-btn")
      .addEventListener("click", () => this.startTraining());

    // Stop training button
    document
      .getElementById("stop-btn")
      .addEventListener("click", () => this.stopTraining());

    // Phase buttons
    document.querySelectorAll(".phase-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const phase = e.target.dataset.phase;
        this.setTrainingPhase(phase);
      });
    });

    // Generate samples button
    document
      .getElementById("generate-btn")
      .addEventListener("click", () => this.generateSamples());

    // Exploration slider
    document
      .getElementById("exploration-slider")
      .addEventListener("input", (e) => {
        const value = e.target.value;
        document.getElementById("exploration-value").textContent = `${value}%`;
        if (this.trainer) {
          this.trainer.setExplorationRate(value / 100);
        }
      });
  }

  async connectToSwarm() {
    this.ui.updateStatus("Connecting to swarm...");
    this.ui.log("Attempting to connect to peer network");

    try {
      await this.network.connect();

      // Update UI
      this.ui.updatePeerCount(this.network.peers.size);
      this.ui.updateStatus("Connected to swarm");
      this.ui.log(`Connected to ${this.network.peers.size} peers`);

      // Enable training button
      this.ui.enableButton("start-btn");
      this.ui.disableButton("connect-btn");

      // Listen for peer updates
      this.network.onPeerUpdate = (count) => {
        this.ui.updatePeerCount(count);
      };

      // Listen for incoming models
      this.network.onModelReceived = (modelData) => {
        this.handleIncomingModel(modelData);
      };
    } catch (error) {
      console.error("Connection failed:", error);
      this.ui.log(`Connection failed: ${error.message}`);
      this.ui.updateStatus("Connection failed");
    }
  }

  async startTraining() {
    if (this.isTraining) return;

    this.ui.updateStatus("Starting training...");
    this.ui.log("Initializing trainer");

    try {
      // Create trainer
      this.trainer = new SwarmTrainer(this.network, this.phaseManager);

      // Setup trainer callbacks
      this.trainer.onEpochComplete = (epoch, loss, metrics) => {
        this.ui.updateEpoch(epoch);
        this.ui.updateLoss(loss);
        this.ui.updateMetrics(metrics);
        this.ui.updatePhase(this.trainer.currentPhase);

        // Update charts
        this.ui.updateLossChart(epoch, loss);
        this.ui.updateDiversityChart(epoch, metrics.diversity);

        // Log progress
        this.ui.log(
          `Epoch ${epoch}: loss=${loss.toFixed(4)}, diversity=${metrics.diversity.toFixed(3)}`,
        );
      };

      this.trainer.onModelShared = (modelData) => {
        this.ui.log("Sharing model with swarm");
        this.ui.incrementSyncCount();
      };

      this.trainer.onModelAdopted = (sourcePeer, epoch) => {
        this.ui.log(`Adopted model from ${sourcePeer} at epoch ${epoch}`);
        this.ui.incrementModelsEvaluated();
      };

      // Start training
      this.isTraining = true;
      this.ui.enableButton("stop-btn");
      this.ui.disableButton("start-btn");
      this.ui.updateStatus("Training in progress");

      await this.trainer.start();
    } catch (error) {
      console.error("Training start failed:", error);
      this.ui.log(`Training failed to start: ${error.message}`);
      this.ui.updateStatus("Training failed");
      this.isTraining = false;
    }
  }

  async stopTraining() {
    if (!this.isTraining || !this.trainer) return;

    this.ui.updateStatus("Stopping training...");
    this.ui.log("Stopping trainer");

    try {
      await this.trainer.stop();
      this.isTraining = false;

      this.ui.enableButton("start-btn");
      this.ui.disableButton("stop-btn");
      this.ui.updateStatus("Training stopped");
      this.ui.log("Training stopped successfully");
    } catch (error) {
      console.error("Stop failed:", error);
      this.ui.log(`Stop failed: ${error.message}`);
    }
  }

  setTrainingPhase(phase) {
    if (!this.trainer) return;

    this.trainer.setPhase(phase);
    this.ui.updatePhase(phase);
    this.ui.log(`Phase changed to: ${phase}`);

    // Update button states
    document.querySelectorAll(".phase-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.phase === phase);
    });
  }

  async generateSamples() {
    if (!this.trainer) {
      this.ui.log("No trainer available. Start training first.");
      return;
    }

    this.ui.log("Generating samples...");

    try {
      const samples = await this.trainer.generateSamples(4);
      this.ui.displaySamples(samples);
      this.ui.log("Samples generated successfully");
    } catch (error) {
      console.error("Sample generation failed:", error);
      this.ui.log(`Sample generation failed: ${error.message}`);
    }
  }

  handleIncomingModel(modelData) {
    if (!this.trainer || !this.isTraining) return;

    this.ui.log(`Received model from peer ${modelData.peerId}`);
    this.ui.incrementModelsEvaluated();

    // Let trainer decide whether to adopt
    this.trainer.evaluateIncomingModel(modelData);
  }
}

// Start the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  window.app = new SwarmTrainingApp();
});

export { SwarmTrainingApp };
