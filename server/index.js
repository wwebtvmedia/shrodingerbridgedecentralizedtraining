import express from "express";
import cors from "cors";
import { WebSocketServer } from "ws";
import { createServer } from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Simple JSON-based NoSQL storage
class JSONDatabase {
  constructor(filename) {
    this.filepath = path.join(__dirname, `../data/${filename}.json`);
    this.data = { logs: [], neighbors: {}, models: [] };
    this.ensureDirectory();
    this.load();
  }

  ensureDirectory() {
    const dir = path.dirname(this.filepath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  load() {
    try {
      if (fs.existsSync(this.filepath)) {
        const raw = fs.readFileSync(this.filepath, "utf8");
        const parsed = JSON.parse(raw);
        this.data.logs = parsed.logs || [];
        this.data.models = parsed.models || [];
        this.data.neighbors = parsed.neighbors || {};
      }
    } catch (e) {
      console.error(`Failed to load database ${this.filepath}:`, e);
    }
  }

  save() {
    try {
      this.ensureDirectory();
      fs.writeFileSync(this.filepath, JSON.stringify(this.data, null, 2));
    } catch (e) {
      console.error(`Failed to save database ${this.filepath}:`, e);
    }
  }

  log(message, type = "info") {
    this.data.logs.push({ timestamp: Date.now(), message, type });
    if (this.data.logs.length > 1000) this.data.logs.shift();
    this.save();
  }

  updateNeighbor(peerId, info) {
    // Sanitize peerId to prevent JSON key injection or traversal-like patterns
    const safePeerId = String(peerId).replace(/[^a-zA-Z0-9_-]/g, "_");
    this.data.neighbors[safePeerId] = { ...info, lastSeen: Date.now() };
    this.save();
  }

  getNeighbors() {
    return Object.values(this.data.neighbors);
  }

  addModel(modelInfo) {
    this.data.models.push({ ...modelInfo, timestamp: Date.now() });
    if (this.data.models.length > 100) this.data.models.shift();
    this.save();
  }
}

class ModelConsolidationServer {
  constructor() {
    this.app = express();
    this.server = createServer(this.app);
    this.wss = new WebSocketServer({ server: this.server });
    this.db = new JSONDatabase("swarm_db");

    // Model management
    this.bestModel = null;
    this.clients = new Map(); // WebSocket clients
    this.trainingClients = new Map(); // Training clients with metadata
    this.modelHistory = [];

    // File paths
    this.modelsDir = path.join(__dirname, "../models");
    this.latestModelPath = path.join(__dirname, "../latest.pt");

    // Ensure directories exist
    this.ensureDirectories();

    // Load initial model if exists
    this.loadInitialModel();

    // Setup middleware and routes
    this.setupMiddleware();
    this.setupRoutes();
    this.setupWebSocket();

    // Start periodic model evaluation
    this.startModelEvaluation();
  }

  ensureDirectories() {
    if (!fs.existsSync(this.modelsDir)) {
      fs.mkdirSync(this.modelsDir, { recursive: true });
    }
  }

  loadInitialModel() {
    try {
      if (fs.existsSync(this.latestModelPath)) {
        const stats = fs.statSync(this.latestModelPath);
        this.bestModel = {
          path: this.latestModelPath,
          size: stats.size,
          timestamp: stats.mtime,
          loss: 0.0, // Default loss
          epoch: 0,
          clientId: "initial",
          metrics: {},
        };
        console.log(
          `✅ Loaded initial model: ${this.latestModelPath} (${stats.size} bytes)`,
        );
      } else {
        console.log(
          "⚠️  No initial model found. Waiting for client submissions.",
        );
      }
    } catch (error) {
      console.error("❌ Failed to load initial model:", error);
    }
  }

  setupMiddleware() {
    this.app.use(cors());
    this.app.use(express.json({ limit: "100mb" })); // For large model uploads
    this.app.use(express.static(path.join(__dirname, "../public")));
    // SECURITY: Removed line that serves root directory
  }

  setupRoutes() {
    // Health check
    this.app.get("/api/health", (req, res) => {
      res.json({
        status: "healthy",
        bestModel: this.bestModel
          ? {
              hasModel: true,
              size: this.bestModel.size,
              timestamp: this.bestModel.timestamp,
              loss: this.bestModel.loss,
            }
          : { hasModel: false },
        clients: this.clients.size,
        trainingClients: this.trainingClients.size,
      });
    });

    // Get best model info
    this.app.get("/api/model/best", (req, res) => {
      if (!this.bestModel) {
        return res.status(404).json({ error: "No model available" });
      }

      res.json({
        model: {
          loss: this.bestModel.loss,
          epoch: this.bestModel.epoch,
          clientId: this.bestModel.clientId,
          timestamp: this.bestModel.timestamp,
          size: this.bestModel.size,
          metrics: this.bestModel.metrics,
        },
        downloadUrl: "/api/model/download",
      });
    });

    // Download best model
    this.app.get("/api/model/download", (req, res) => {
      if (!this.bestModel || !fs.existsSync(this.bestModel.path)) {
        return res.status(404).json({ error: "Model not found" });
      }

      res.download(this.bestModel.path, "latest.pt");
    });

    // Submit new model
    this.app.post("/api/model/submit", async (req, res) => {
      try {
        const { clientId, modelData, loss, epoch, metrics = {} } = req.body;

        if (!clientId || !modelData || loss === undefined) {
          return res.status(400).json({ error: "Missing required fields" });
        }

        console.log(
          `📥 Model submission from client ${clientId}: loss=${loss}, epoch=${epoch}`,
        );

        // Process model submission
        const isBetter = await this.evaluateModel({
          clientId,
          modelData,
          loss,
          epoch,
          metrics,
        });

        if (isBetter) {
          // Broadcast new best model
          this.broadcastNewBestModel();
          res.json({
            accepted: true,
            isBest: true,
            message: "Model accepted as new best model",
          });
        } else {
          res.json({
            accepted: true,
            isBest: false,
            message: "Model accepted but not better than current best",
          });
        }
      } catch (error) {
        console.error("❌ Model submission error:", error);
        res.status(500).json({ error: error.message });
      }
    });

    // Get training clients
    this.app.get("/api/clients", (req, res) => {
      const clients = Array.from(this.trainingClients.entries()).map(
        ([id, data]) => ({
          id,
          ...data,
          lastSeen: data.lastSeen
            ? new Date(data.lastSeen).toISOString()
            : null,
        }),
      );

      res.json({ clients });
    });

    // Get model history
    this.app.get("/api/model/history", (req, res) => {
      res.json({ history: this.modelHistory.slice(-20) }); // Last 20 entries
    });

    // Serve frontend
    this.app.get("/", (req, res) => {
      res.sendFile(path.join(__dirname, "../index.html"));
    });

    // Serve enhanced frontend
    this.app.get("/enhanced", (req, res) => {
      res.sendFile(path.join(__dirname, "../enhanced-index.html"));
    });
  }

  setupWebSocket() {
    this.wss.on("connection", (ws, req) => {
      const clientId = `ws_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      console.log(`🔌 WebSocket client connected: ${clientId}`);

      // Add to clients map
      this.clients.set(clientId, {
        ws,
        connectedAt: new Date(),
        lastActivity: Date.now(),
      });

      // PUSH initial data: Best Model + Neighbors
      this.pushInitialData(ws);

      // Handle messages
      ws.on("message", (data) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleWebSocketMessage(clientId, message);
        } catch (error) {
          console.error("❌ WebSocket message error:", error);
        }
      });

      // Handle disconnection
      ws.on("close", () => {
        console.log(`🔌 WebSocket client disconnected: ${clientId}`);
        this.clients.delete(clientId);

        // Also remove from training clients if present
        this.trainingClients.delete(clientId);
      });

      // Handle errors
      ws.on("error", (error) => {
        console.error(`❌ WebSocket error for client ${clientId}:`, error);
        this.clients.delete(clientId);
        this.trainingClients.delete(clientId);
      });
    });
  }

  pushInitialData(ws) {
    const neighbors = this.db.getNeighbors();
    const message = {
      type: "initial_sync",
      bestModel: this.bestModel
        ? {
            loss: this.bestModel.loss,
            epoch: this.bestModel.epoch,
            metrics: this.bestModel.metrics,
            timestamp: this.bestModel.timestamp,
            downloadUrl: "/api/model/download",
          }
        : null,
      neighbors: neighbors.slice(-10), // Latest 10 neighbors
    };

    if (ws.readyState === ws.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  handleWebSocketMessage(clientId, message) {
    const client = this.clients.get(clientId);
    if (client) {
      client.lastActivity = Date.now();
    }

    switch (message.type) {
      case "register_training":
        // Register as training client
        this.trainingClients.set(clientId, {
          ...message.data,
          lastSeen: Date.now(),
          ws: client.ws,
        });
        this.db.log(`Training client registered: ${clientId}`);
        console.log(
          `🏋️  Training client registered: ${clientId} (${message.data.name || "unnamed"})`,
        );
        break;

      case "heartbeat":
      case "status_update":
        // Update training client heartbeat and record stats
        if (this.trainingClients.has(clientId)) {
          const clientData = this.trainingClients.get(clientId);
          clientData.lastSeen = Date.now();
          clientData.metrics = message.metrics || clientData.metrics;

          // Record neighbors if provided
          if (message.neighbors && Array.isArray(message.neighbors)) {
            // SECURITY: Limit number of neighbors recorded to prevent DoS
            message.neighbors.slice(0, 50).forEach((n) => {
              if (n.peerId) this.db.updateNeighbor(n.peerId, n);
            });
          }

          // Record stats in log
          if (message.metrics) {
            this.db.log(
              `Stats from ${clientId}: loss=${message.metrics.loss}, epoch=${message.metrics.epoch}`,
            );
          }

          // Send acknowledgment
          client.ws.send(
            JSON.stringify({
              type: "heartbeat_ack",
              timestamp: Date.now(),
            }),
          );
        }
        break;

      case "final_sync":
        // Client is closing, record final data
        this.db.log(
          `Final sync from ${clientId}: loss=${message.metrics?.loss}`,
        );
        if (message.neighbors && Array.isArray(message.neighbors)) {
          message.neighbors.slice(0, 50).forEach((n) => {
            if (n.peerId) this.db.updateNeighbor(n.peerId, n);
          });
        }
        break;

      case "model_update":
        // Handle model update from client
        this.handleClientModelUpdate(clientId, message);
        break;

      default:
        console.log(
          `📨 Unknown message type from ${clientId}: ${message.type}`,
        );
    }
  }

  async handleClientModelUpdate(clientId, message) {
    const { modelData, loss, epoch, metrics } = message;

    console.log(
      `📤 Model update from ${clientId}: loss=${loss}, epoch=${epoch}`,
    );

    const isBetter = await this.evaluateModel({
      clientId,
      modelData,
      loss,
      epoch,
      metrics,
    });

    if (isBetter) {
      // Notify the client that their model is now the best
      const client = this.clients.get(clientId);
      if (client) {
        client.ws.send(
          JSON.stringify({
            type: "model_accepted_as_best",
            loss,
            epoch,
            timestamp: Date.now(),
          }),
        );
      }

      // Broadcast to all clients
      this.broadcastNewBestModel();
    }
  }

  async evaluateModel(submission) {
    const { clientId, modelData, loss, epoch, metrics } = submission;

    // Sanitize clientId to prevent path traversal
    const safeClientId = String(clientId).replace(/[^a-zA-Z0-9_-]/g, "_");

    // Simple evaluation: lower loss is better
    const isBetter = !this.bestModel || loss < this.bestModel.loss;

    if (isBetter) {
      console.log(
        `🏆 New best model from ${safeClientId}: loss=${loss} (previous: ${this.bestModel?.loss || "none"})`,
      );

      // Save model to file
      const modelPath = path.join(
        this.modelsDir,
        `model_${Date.now()}_${safeClientId}.pt`,
      );
      const latestPath = this.latestModelPath;

      try {
        // Decode base64 model data if needed
        let modelBuffer;
        if (typeof modelData === "string" && modelData.startsWith("data:")) {
          // Handle data URL
          const base64Data = modelData.split(",")[1];
          modelBuffer = Buffer.from(base64Data, "base64");
        } else if (typeof modelData === "string") {
          // Assume base64 string
          modelBuffer = Buffer.from(modelData, "base64");
        } else {
          // Assume binary buffer
          modelBuffer = Buffer.from(modelData);
        }

        // Save model
        fs.writeFileSync(modelPath, modelBuffer);
        fs.writeFileSync(latestPath, modelBuffer);

        // Update best model
        this.bestModel = {
          path: latestPath,
          size: modelBuffer.length,
          timestamp: new Date(),
          loss,
          epoch,
          clientId,
          metrics,
          sourcePath: modelPath,
        };

        // Record in DB
        this.db.addModel({ loss, epoch, clientId, size: modelBuffer.length });
        this.db.log(`New best model accepted from ${clientId}: loss=${loss}`);

        // Add to history
        this.modelHistory.push({
          timestamp: new Date(),
          clientId,
          loss,
          epoch,
          isBest: true,
        });

        // Keep history manageable
        if (this.modelHistory.length > 100) {
          this.modelHistory = this.modelHistory.slice(-100);
        }

        return true;
      } catch (error) {
        console.error("❌ Failed to save model:", error);
        return false;
      }
    }

    // Still add to history even if not best
    this.modelHistory.push({
      timestamp: new Date(),
      clientId,
      loss,
      epoch,
      isBest: false,
    });

    return false;
  }

  broadcastNewBestModel() {
    if (!this.bestModel) return;

    const message = JSON.stringify({
      type: "new_best_model",
      model: {
        loss: this.bestModel.loss,
        epoch: this.bestModel.epoch,
        clientId: this.bestModel.clientId,
        timestamp: this.bestModel.timestamp,
        metrics: this.bestModel.metrics,
      },
    });

    // Broadcast to all connected clients
    this.clients.forEach((client, clientId) => {
      if (client.ws.readyState === client.ws.OPEN) {
        client.ws.send(message);
      }
    });

    console.log(
      `📢 Broadcast new best model: loss=${this.bestModel.loss}, client=${this.bestModel.clientId}`,
    );
  }

  startModelEvaluation() {
    // Periodically check for stale clients
    setInterval(() => {
      const now = Date.now();
      const staleThreshold = 30000; // 30 seconds

      this.trainingClients.forEach((client, clientId) => {
        if (now - client.lastSeen > staleThreshold) {
          console.log(`⏰ Removing stale training client: ${clientId}`);
          this.trainingClients.delete(clientId);

          // Also close WebSocket if still open
          const wsClient = this.clients.get(clientId);
          if (wsClient && wsClient.ws.readyState === wsClient.ws.OPEN) {
            wsClient.ws.close();
          }
        }
      });
    }, 10000); // Check every 10 seconds
  }

  start(port = 8080) {
    this.server.listen(port, "0.0.0.0", () => {
      console.log(`🚀 Model Consolidation Server running on port ${port}`);
      console.log(`🌐 Web interface: http://localhost:${port}`);
      console.log(`📡 WebSocket: ws://localhost:${port}`);
      console.log(`📊 API: http://localhost:${port}/api/health`);
    });
  }
}

// Start server
const server = new ModelConsolidationServer();
server.start(process.env.PORT || 3001);

export { ModelConsolidationServer };
