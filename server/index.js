import express from "express";
import cors from "cors";
import { WebSocketServer } from "ws";
import { createServer } from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Simple JSON-based NoSQL storage with size limits
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
    // Cap logs at 500 entries (DoS mitigation)
    if (this.data.logs.length > 500) this.data.logs.shift();
    this.save();
  }

  updateNeighbor(peerId, info) {
    const safePeerId = String(peerId).replace(/[^a-zA-Z0-9_-]/g, "_");
    this.data.neighbors[safePeerId] = { ...info, lastSeen: Date.now() };
    
    // Cap neighbors at 100 entries
    const keys = Object.keys(this.data.neighbors);
    if (keys.length > 100) {
      delete this.data.neighbors[keys[0]];
    }
    this.save();
  }

  getNeighbors() {
    return Object.values(this.data.neighbors);
  }

  addModel(modelInfo) {
    this.data.models.push({ ...modelInfo, timestamp: Date.now() });
    // Cap model history at 100 entries
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
    
    // Auth Token from environment
    this.authToken = process.env.SECRET_TOKEN || "swarm-prototype-token-2026";

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
          loss: 0.0,
          epoch: 0,
          clientId: "initial",
          metrics: {},
        };
        console.log(`✅ Loaded initial model: ${this.latestModelPath}`);
      }
    } catch (error) {
      console.error("❌ Failed to load initial model:", error);
    }
  }

  setupMiddleware() {
    this.app.use(cors());
    this.app.use(express.json({ limit: "100mb" }));
    this.app.use(express.static(path.join(__dirname, "../public")));
    this.app.use("/src", express.static(path.join(__dirname, "../src")));
    
    // Auth Middleware
    this.authenticate = (req, res, next) => {
      const authHeader = req.headers.authorization;
      if (authHeader === `Bearer ${this.authToken}`) {
        next();
      } else {
        res.status(401).json({ error: "Unauthorized" });
      }
    };
  }

  setupRoutes() {
    // Public health check
    this.app.get("/api/health", (req, res) => {
      res.json({ status: "healthy", clients: this.clients.size });
    });

    // Protected endpoints
    this.app.get("/api/model/best", this.authenticate, (req, res) => {
      if (!this.bestModel) return res.status(404).json({ error: "No model" });
      res.json(this.bestModel);
    });

    this.app.get("/api/model/download", this.authenticate, (req, res) => {
      if (!this.bestModel || !fs.existsSync(this.bestModel.path)) {
        return res.status(404).json({ error: "Model not found" });
      }
      res.download(this.bestModel.path, "latest.pt");
    });

    this.app.post("/api/model/submit", this.authenticate, async (req, res) => {
      const { clientId, modelData, loss, epoch } = req.body;
      const isBetter = await this.evaluateModel({ clientId, modelData, loss, epoch });
      if (isBetter) this.broadcastNewBestModel();
      res.json({ accepted: true, isBest: isBetter });
    });

    this.app.get("/", (req, res) => res.sendFile(path.join(__dirname, "../index.html")));
    this.app.get("/enhanced", (req, res) => res.sendFile(path.join(__dirname, "../enhanced-index.html")));
    this.app.get("/enhanced-index.html", (req, res) => res.sendFile(path.join(__dirname, "../enhanced-index.html")));
    this.app.get("/readme.html", (req, res) => res.sendFile(path.join(__dirname, "../readme.html")));
    this.app.get("/beyond-labor.html", (req, res) => res.sendFile(path.join(__dirname, "../beyond-labor.html")));
    this.app.get("/training-consolidation.html", (req, res) => res.sendFile(path.join(__dirname, "../training-consolidation.html")));
    this.app.get("/test.html", (req, res) => res.sendFile(path.join(__dirname, "../test.html")));
    this.app.get("/test-hardware.html", (req, res) => res.sendFile(path.join(__dirname, "../test-hardware.html")));
    this.app.get("/test-inference.html", (req, res) => res.sendFile(path.join(__dirname, "../test-inference.html")));
  }

  setupWebSocket() {
    this.wss.on("connection", (ws, req) => {
      const url = new URL(req.url, `http://${req.headers.host}`);
      const token = url.searchParams.get("token");
      
      if (token !== this.authToken) {
        ws.close(4001, "Unauthorized");
        return;
      }

      const clientId = `ws_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      this.clients.set(clientId, { ws, connectedAt: new Date(), lastActivity: Date.now() });
      this.pushInitialData(ws);

      ws.on("message", (data) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleWebSocketMessage(clientId, message);
        } catch (error) {
          console.error("❌ WS Error:", error);
        }
      });

      ws.on("close", () => {
        this.clients.delete(clientId);
        this.trainingClients.delete(clientId);
      });
    });
  }

  pushInitialData(ws) {
    const neighbors = this.db.getNeighbors();
    const message = {
      type: "initial_sync",
      bestModel: this.bestModel ? { loss: this.bestModel.loss, epoch: this.bestModel.epoch } : null,
      neighbors: neighbors.slice(-10),
    };
    if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(message));
  }

  handleWebSocketMessage(clientId, message) {
    const client = this.clients.get(clientId);
    if (client) client.lastActivity = Date.now();

    switch (message.type) {
      case "register_training":
        this.trainingClients.set(clientId, { ...message.data, lastSeen: Date.now(), ws: client.ws });
        this.db.log(`Client registered: ${clientId}`);
        break;
      case "status_update":
        if (this.trainingClients.has(clientId)) {
          const clientData = this.trainingClients.get(clientId);
          clientData.lastSeen = Date.now();
          if (message.neighbors) message.neighbors.slice(0, 50).forEach(n => n.peerId && this.db.updateNeighbor(n.peerId, n));
          if (message.metrics) this.db.log(`Stats ${clientId}: loss=${message.metrics.loss}`);
        }
        break;
      case "model_update":
        this.handleClientModelUpdate(clientId, message);
        break;
      case "PEER_MESSAGE":
        this.relayToPeer(message.to, message);
        break;
      case "BROADCAST":
        this.broadcastToAll(message, clientId);
        break;
    }
  }

  relayToPeer(targetId, message) {
    const target = this.clients.get(targetId);
    if (target && target.ws.readyState === target.ws.OPEN) {
      target.ws.send(JSON.stringify(message));
    }
  }

  broadcastToAll(message, senderId) {
    this.clients.forEach((client, id) => {
      if (id !== senderId && client.ws.readyState === client.ws.OPEN) {
        client.ws.send(JSON.stringify(message));
      }
    });
  }

  async handleClientModelUpdate(clientId, message) {
    const isBetter = await this.evaluateModel(message);
    if (isBetter) this.broadcastNewBestModel();
  }

  async evaluateModel(submission) {
    const { clientId, modelData, loss, epoch } = submission;
    const safeClientId = String(clientId).replace(/[^a-zA-Z0-9_-]/g, "_");
    const isBetter = !this.bestModel || loss < this.bestModel.loss;

    if (isBetter) {
      const modelPath = path.join(this.modelsDir, `model_${Date.now()}_${safeClientId}.pt`);
      try {
        let modelBuffer = Buffer.from(modelData.startsWith("data:") ? modelData.split(",")[1] : modelData, "base64");
        fs.writeFileSync(modelPath, modelBuffer);
        fs.writeFileSync(this.latestModelPath, modelBuffer);

        this.bestModel = { path: this.latestModelPath, size: modelBuffer.length, timestamp: new Date(), loss, epoch, clientId };
        this.db.addModel({ loss, epoch, clientId });
        this.db.log(`New best model: ${loss}`);
        return true;
      } catch (error) {
        console.error("❌ Save Error:", error);
      }
    }
    return false;
  }

  broadcastNewBestModel() {
    if (!this.bestModel) return;
    const msg = JSON.stringify({ type: "new_best_model", model: { loss: this.bestModel.loss, epoch: this.bestModel.epoch, clientId: this.bestModel.clientId } });
    this.clients.forEach(c => c.ws.readyState === c.ws.OPEN && c.ws.send(msg));
  }

  startModelEvaluation() {
    setInterval(() => {
      const now = Date.now();
      this.trainingClients.forEach((c, id) => {
        if (now - c.lastSeen > 30000) {
          this.trainingClients.delete(id);
          const wsC = this.clients.get(id);
          if (wsC) wsC.ws.close();
        }
      });
    }, 10000);
  }

  start(port = 3001) {
    this.server.listen(port, "0.0.0.0", () => console.log(`🚀 Server running on port ${port}`));
  }
}

const server = new ModelConsolidationServer();
server.start(process.env.PORT || 3001);

export { ModelConsolidationServer };
