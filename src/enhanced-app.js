import { EnhancedSwarmTrainer } from "./core/enhanced-trainer.js";
import { UIManager } from "./ui/manager.js";
import { DataImporter } from "./utils/data-importer.js";
import { InferenceEngine } from "./utils/inference.js";

class EnhancedSwarmApp {
  constructor() {
    this.ui = new UIManager();
    this.trainer = null;
    this.dataImporter = new DataImporter();
    this.inferenceEngine = null;
    this.isInitialized = false;

    // Start initialization
    this.init().catch(err => {
      console.error("Failed to initialize app:", err);
    });
  }

  async init() {
    console.log("🚀 Initializing Enhanced Swarm App...");

    // Initialize UI
    this.ui.init();
    this.ui.updateStatus("Initializing...");

    // Setup event listeners
    this.setupEventListeners();

    // Set initialized state
    this.isInitialized = true;
    
    // Update UI
    this.ui.updateStatus("Ready to connect");
    this.ui.enableButton("connect-btn");

    console.log("✅ Enhanced app initialized");
  }

  setupEventListeners() {
    // Helper to get element and add listener
    const addListener = (id, event, callback) => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener(event, callback);
      } else {
        console.warn(`⚠️ Element #${id} not found when setting up listeners`);
      }
    };

    // Connect button
    addListener("connect-btn", "click", () => this.connect());

    // Research neighbors button
    addListener("research-btn", "click", () => this.researchNeighbors());

    // Start training button
    addListener("start-btn", "click", () => this.startTraining());

    // Stop training button
    addListener("stop-btn", "click", () => this.stopTraining());

    // Phase buttons
    document.querySelectorAll(".phase-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const target = e.currentTarget || e.target;
        const phase = target.dataset.phase;
        this.setTrainingPhase(phase);
      });
    });

    // Generate samples button
    addListener("generate-btn", "click", () => this.generateSamples());

    // Exploration slider
    const slider = document.getElementById("exploration-slider");
    if (slider) {
      slider.addEventListener("input", (e) => {
        const value = e.target.value;
        const display = document.getElementById("exploration-value");
        if (display) display.textContent = `${value}%`;
        if (this.trainer) {
          this.trainer.setExplorationRate(value / 100);
        }
      });
    }

    // Database export button
    addListener("export-btn", "click", () => this.exportDatabase());

    // Database import button
    addListener("import-btn", "click", () => this.importDatabase());

    // Data import button
    addListener("data-import-btn", "click", () => this.importData());

    // Inference button
    addListener("inference-btn", "click", () => this.runInference());

    // Stats button
    addListener("stats-btn", "click", () => this.showDatabaseStats());

    // Refresh button
    addListener("refresh-btn", "click", () => this.showDatabaseStats());

    // Cleanup button
    addListener("cleanup-btn", "click", () => this.cleanupDatabase());
  }

  async connect() {
    // If not initialized, try to wait a bit or initialize now
    if (!this.isInitialized) {
      this.ui.log("⌛ Still initializing... please wait.");
      await this.init();
    }

    // Disable button to prevent multiple clicks
    this.ui.disableButton("connect-btn");
    this.ui.updateStatus("Connecting...");
    this.ui.log("Initializing enhanced swarm trainer");

    try {
      // Create enhanced trainer with database and tunnel
      this.trainer = new EnhancedSwarmTrainer({
        useDatabase: true,
        useTunnel: true,
        tunnelConfig: {
          tunnelUrl: window.location.origin.replace(/^http/, "ws"),
          tunnelId: `trainer_${Date.now()}`,
          authToken: "swarm-prototype-token-2026" // In real app, this would be from config
        },
        explorationRate: 0.3,
        syncInterval: 5,
      });

      // Setup trainer callbacks
      this.trainer.onEpochComplete((epoch, loss, metrics) => {
        this.ui.updateEpoch(epoch);
        this.ui.updateLoss(loss);
        this.ui.updatePhase(this.trainer.currentPhase);

        // Update charts
        this.ui.updateLossChart(epoch, loss);
        this.ui.updateDiversityChart(epoch, metrics.diversity || 0.5);

        // Update metrics
        this.ui.updateMetrics({
          modelsEvaluated: this.trainer.metrics.modelsReceived,
          syncCount: this.trainer.metrics.syncEvents,
        });

        // Log progress
        this.ui.log(
          `Epoch ${epoch}: loss=${loss.toFixed(4)}, phase=${this.trainer.currentPhase}`,
        );
      });

      this.trainer.onNeighborUpdate((event, peerId, metadata) => {
        if (event === "connected") {
          this.ui.log(`👋 Neighbor connected: ${peerId}`);
          this.ui.updatePeerCount(this.trainer.getNeighborCount());
        } else if (event === "disconnected") {
          this.ui.log(`👋 Neighbor disconnected: ${peerId}`);
          this.ui.updatePeerCount(this.trainer.getNeighborCount());
        }
      });

      this.trainer.onModelShared((result) => {
        this.ui.log(
          `📤 Shared model with neighbors (loss: ${result.loss.toFixed(4)})`,
        );
      });

      this.trainer.onModelReceived((peerId, model) => {
        this.ui.log(`📥 Received model from ${peerId}`);
        this.ui.incrementModelsEvaluated();
      });

      this.trainer.onSyncEvent((peerId, epoch) => {
        this.ui.log(`🔄 Synchronized with ${peerId} at epoch ${epoch}`);
        this.ui.incrementSyncCount();
      });

      this.trainer.onResearchResult((peerId, status) => {
        this.ui.log(`📊 Research: Response from ${peerId} (Loss: ${status.metrics?.loss.toFixed(4) || "N/A"})`);
        this.ui.showNotification(`Found neighbor: ${peerId.substring(0, 8)}...`, "info");
      });

      this.trainer.onDatabaseUpdate((stats) => {
        this.ui.log(
          `📊 Database: ${stats.neighbors.count} neighbors, ${stats.results.count} results`,
        );
      });

      // Initialize trainer
      await this.trainer.initialize();

      // Update UI
      this.ui.updateStatus("Connected to swarm");
      this.ui.updatePeerCount(this.trainer.getNeighborCount());
      this.ui.log("✅ Enhanced trainer initialized with database and tunnel");

      // Enable training button
      this.ui.enableButton("start-btn");
      this.ui.enableButton("research-btn");
      this.ui.disableButton("connect-btn");
    } catch (error) {
      console.error("Connection failed:", error);
      this.ui.log(`❌ Connection failed: ${error.message}`);
      this.ui.updateStatus("Connection failed");
    }
  }

  async researchNeighbors() {
    if (!this.trainer) {
      this.ui.log("❌ Trainer not initialized. Connect first.");
      return;
    }

    this.ui.log("🔍 Starting neighbor research...");
    this.ui.updateStatus("Researching neighbors");

    try {
      await this.trainer.researchNeighbors();
      this.ui.log("📡 Research broadcast sent. Waiting for responses...");
    } catch (error) {
      console.error("Research failed:", error);
      this.ui.log(`❌ Research failed: ${error.message}`);
    }
  }

  async startTraining() {
    if (!this.trainer) {
      this.ui.log("❌ Trainer not initialized. Connect first.");
      return;
    }

    this.ui.updateStatus("Starting training...");
    this.ui.log("Starting enhanced swarm training");

    try {
      await this.trainer.startTraining();

      this.ui.enableButton("stop-btn");
      this.ui.disableButton("start-btn");
      this.ui.updateStatus("Training in progress");
      this.ui.log("✅ Training started with database persistence");
    } catch (error) {
      console.error("Training start failed:", error);
      this.ui.log(`❌ Training failed to start: ${error.message}`);
      this.ui.updateStatus("Training failed");
    }
  }

  async stopTraining() {
    if (!this.trainer) {
      this.ui.log("❌ Cannot stop: Trainer not initialized.");
      return;
    }

    this.ui.updateStatus("Stopping training...");
    this.ui.log("Stopping trainer and saving state");

    try {
      await this.trainer.stopTraining();

      this.ui.enableButton("start-btn");
      this.ui.disableButton("stop-btn");
      this.ui.updateStatus("Training stopped");
      this.ui.log("✅ Training stopped. State saved to database.");
    } catch (error) {
      console.error("Stop failed:", error);
      this.ui.log(`❌ Stop failed: ${error.message}`);
    }
  }

  setTrainingPhase(phase) {
    if (!this.trainer) {
      this.ui.log("❌ Cannot change phase: Trainer not initialized. Connect first.");
      return;
    }

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
      this.ui.log("❌ No trainer available. Start training first.");
      return;
    }

    this.ui.log("Generating samples...");

    try {
      const samples = await this.generateSimulatedSamples(4);
      this.ui.displaySamples(samples);
      this.ui.log("✅ Samples generated");
    } catch (error) {
      console.error("Sample generation failed:", error);
      this.ui.log(`❌ Sample generation failed: ${error.message}`);
    }
  }

  async generateSimulatedSamples(count = 4) {
    const samples = [];
    for (let i = 0; i < count; i++) {
      const canvas = document.createElement("canvas");
      canvas.width = 64;
      canvas.height = 64;
      const ctx = canvas.getContext("2d");
      const hue = Math.random() * 360;
      ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
      ctx.fillRect(0, 0, 64, 64);
      ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
      for (let j = 0; j < 5; j++) {
        const x = Math.random() * 64;
        const y = Math.random() * 64;
        const size = 5 + Math.random() * 15;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }
      samples.push(canvas.toDataURL());
    }
    return samples;
  }

  async exportDatabase() {
    if (!this.trainer || !this.trainer.database) {
      this.ui.log("❌ Cannot export: Connect to swarm first.");
      return;
    }
    try {
      const exportData = await this.trainer.database.exportData();
      const blob = new Blob([exportData], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `swarm-training-backup-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      this.ui.log("✅ Database exported successfully");
    } catch (error) {
      console.error("Export failed:", error);
      this.ui.log(`❌ Export failed: ${error.message}`);
    }
  }

  async importDatabase() {
    if (!this.trainer || !this.trainer.database) {
      this.ui.log("❌ Cannot import: Connect to swarm first.");
      return;
    }
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        const text = await file.text();
        await this.trainer.database.importData(text);
        this.ui.log(`✅ Database imported from ${file.name}`);
        const stats = await this.trainer.database.getStatistics();
        this.ui.updatePeerCount(stats.neighbors.count);
      } catch (error) {
        console.error("Import failed:", error);
        this.ui.log(`❌ Import failed: ${error.message}`);
      }
    };
    input.click();
  }

  async importData() {
    this.ui.log("📁 Opening file picker for data import...");
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.multiple = true;
    fileInput.accept = "image/*,.txt,.json,.csv";
    fileInput.addEventListener("change", async (event) => {
      const files = Array.from(event.target.files);
      if (files.length === 0) return;
      this.ui.log(`📁 Importing ${files.length} file(s)...`);
      try {
        const imageFiles = files.filter((file) => file.type.startsWith("image/"));
        const textFiles = files.filter((file) => file.type.startsWith("text/") || file.name.endsWith(".txt") || file.name.endsWith(".json") || file.name.endsWith(".csv"));
        let importedCount = 0;
        if (imageFiles.length > 0) {
          const images = await this.dataImporter.importImages(imageFiles);
          if (this.trainer && this.trainer.database) {
            for (const image of images) {
              await this.trainer.database.saveTrainingData({ type: "image", data: image.data, metadata: image.metadata, timestamp: Date.now() });
            }
          }
          importedCount += images.length;
        }
        if (textFiles.length > 0) {
          const texts = await this.dataImporter.importText(textFiles);
          if (this.trainer && this.trainer.database) {
            for (const text of texts) {
              await this.trainer.database.saveTrainingData({ type: "text", data: text.content, metadata: text.metadata, timestamp: Date.now() });
            }
          }
          importedCount += texts.length;
        }
        this.ui.log(`🎉 Successfully imported ${importedCount} total item(s)`);
      } catch (error) {
        console.error("Data import failed:", error);
        this.ui.log(`❌ Import failed: ${error.message}`);
      }
    });
    fileInput.click();
  }

  async runInference() {
    if (!this.trainer) {
      this.ui.log("❌ Trainer not initialized. Connect first.");
      return;
    }
    if (!this.inferenceEngine) {
      this.inferenceEngine = new InferenceEngine(this.trainer.modelManager);
      await this.inferenceEngine.initialize();
    }
    this.ui.log("🎨 Running inference...");
    try {
      const label = prompt("Enter label (0-9) for conditioned generation, or leave empty:", "");
      const options = { sampleCount: 4, steps: 50, temperature: 0.7, cfgScale: 1.0 };
      if (label !== null && label !== "") options.label = parseInt(label);
      const result = await this.inferenceEngine.generateSamples(options);
      this.ui.displaySamples(result.samples.map((s) => s.image));
      this.ui.log(`✅ Generated ${result.samples.length} samples`);
    } catch (error) {
      console.error("Inference failed:", error);
      this.ui.log(`❌ Inference failed: ${error.message}`);
    }
  }

  async showDatabaseStats() {
    if (!this.trainer || !this.trainer.database) {
      this.ui.log("❌ Connect to swarm first.");
      return;
    }
    try {
      const stats = await this.trainer.database.getStatistics();
      this.ui.log(`📊 DB: ${stats.neighbors.count} neighbors, ${stats.results.count} results, size: ${Math.round(stats.database.size / 1024)} KB`);
    } catch (error) {
      console.error("Failed to get stats:", error);
    }
  }

  async cleanupDatabase() {
    if (!this.trainer || !this.trainer.database) return;
    if (confirm("Clear all database data?")) {
      try {
        await this.trainer.database.clearDatabase();
        this.ui.log("✅ Database cleared");
      } catch (error) {
        this.ui.log(`❌ Cleanup failed: ${error.message}`);
      }
    }
  }
}

const startApp = () => {
  if (window.enhancedApp) return;
  window.enhancedApp = new EnhancedSwarmApp();
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", startApp);
} else {
  startApp();
}

export { EnhancedSwarmApp };
