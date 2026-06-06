import express from "express";
import cors from "cors";
import { WebSocketServer } from "ws";
import { createServer } from "http";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";
import { Sanitizer } from "../src/utils/sanitizer.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Maximum decoded size of a submitted model (bytes). Prevents disk-fill DoS.
const MAX_MODEL_BYTES = 64 * 1024 * 1024; // 64 MB
// Keep at most this many per-client model files on disk.
const MAX_MODEL_FILES = 50;

// JSON.parse reviver that strips prototype-pollution keys from untrusted input.
const FORBIDDEN_KEYS = new Set(["__proto__", "constructor", "prototype"]);
function safeReviver(key, value) {
  if (FORBIDDEN_KEYS.has(key)) return undefined;
  return value;
}

// Constant-time token comparison to avoid timing side-channels.
function tokensMatch(provided, expected) {
  if (typeof provided !== "string" || typeof expected !== "string")
    return false;
  const a = Buffer.from(provided);
  const b = Buffer.from(expected);
  if (a.length !== b.length) return false;
  return crypto.timingSafeEqual(a, b);
}

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
        // Use a reviver to drop __proto__/constructor keys in case the file
        // was tampered with.
        const parsed = JSON.parse(raw, safeReviver);
        this.data.logs = parsed.logs || [];
        this.data.models = parsed.models || [];
        this.data.neighbors = parsed.neighbors || {};
      }
    } catch (e) {
      console.error(`Failed to load database ${this.filepath}:`, e);
    }
  }

  // Coalesce frequent mutations into one async write per tick instead of a
  // synchronous full-file rewrite on every log/status update (event-loop DoS).
  save() {
    if (this._saveScheduled) return;
    this._saveScheduled = true;
    setTimeout(() => {
      this._saveScheduled = false;
      this.flush();
    }, 250);
  }

  flush() {
    try {
      this.ensureDirectory();
      const tmp = `${this.filepath}.tmp`;
      fs.writeFile(tmp, JSON.stringify(this.data, null, 2), (err) => {
        if (err) {
          console.error(`Failed to save database ${this.filepath}:`, err);
          return;
        }
        fs.rename(tmp, this.filepath, (renameErr) => {
          if (renameErr)
            console.error(
              `Failed to commit database ${this.filepath}:`,
              renameErr,
            );
        });
      });
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

    // Auth token. Never ship a shared, known default: require SECRET_TOKEN, and
    // if it's missing, generate a random per-process token so an operator who
    // forgot to set one doesn't silently run with a public credential.
    this.authToken = process.env.SECRET_TOKEN;
    if (!this.authToken) {
      this.authToken = crypto.randomBytes(32).toString("hex");
      console.warn(
        "⚠️  SECRET_TOKEN not set — generated an ephemeral token for this run:\n" +
          `    ${this.authToken}\n` +
          "    Set SECRET_TOKEN in the environment for a stable, shared secret.",
      );
    }

    // Model management
    this.bestModel = null;
    this.clients = new Map(); // WebSocket clients
    this.trainingClients = new Map(); // Training clients with metadata
    this.modelHistory = [];

    // File paths
    this.modelsDir = path.join(__dirname, "../models");
    this.latestModelPath = path.join(__dirname, "../checkpoints/latest.pt");

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
    // Restrict CORS to an explicit allowlist (comma-separated ALLOWED_ORIGINS),
    // defaulting to localhost for development. A wide-open `cors()` combined with
    // token auth lets any site call protected APIs from a victim's browser.
    const allowedOrigins = (
      process.env.ALLOWED_ORIGINS ||
      "http://localhost:3000,http://localhost:3001"
    )
      .split(",")
      .map((o) => o.trim())
      .filter(Boolean);
    this.app.use(
      cors({
        origin(origin, cb) {
          // Allow same-origin/non-browser requests (no Origin header).
          if (!origin || allowedOrigins.includes(origin)) return cb(null, true);
          cb(new Error("Origin not allowed by CORS"));
        },
      }),
    );
    this.app.use(express.json({ limit: "100mb" }));
    this.app.use(express.static(path.join(__dirname, "../public")));
    this.app.use("/src", express.static(path.join(__dirname, "../src")));

    // Auth Middleware (constant-time token comparison).
    this.authenticate = (req, res, next) => {
      const authHeader = req.headers.authorization || "";
      const provided = authHeader.startsWith("Bearer ")
        ? authHeader.slice(7)
        : "";
      if (tokensMatch(provided, this.authToken)) {
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
      if (
        typeof clientId !== "string" ||
        typeof modelData !== "string" ||
        !Number.isFinite(loss) ||
        !Number.isFinite(epoch)
      ) {
        return res.status(400).json({ error: "Invalid submission" });
      }
      try {
        const isBetter = await this.evaluateModel({
          clientId,
          modelData,
          loss,
          epoch,
        });
        if (isBetter) this.broadcastNewBestModel();
        res.json({ accepted: true, isBest: isBetter });
      } catch (err) {
        console.error("❌ Submit error:", err);
        res.status(400).json({ error: "Submission failed" });
      }
    });

    this.app.get("/", (req, res) =>
      res.sendFile(path.join(__dirname, "../index.html")),
    );
    this.app.get("/enhanced", (req, res) =>
      res.sendFile(path.join(__dirname, "../enhanced-index.html")),
    );
    this.app.get("/enhanced-index.html", (req, res) =>
      res.sendFile(path.join(__dirname, "../enhanced-index.html")),
    );
    this.app.get("/knowledge-base.html", (req, res) =>
      res.sendFile(path.join(__dirname, "../knowledge-base.html")),
    );
    this.app.get("/beyond-labor.html", (req, res) =>
      res.redirect("/knowledge-base.html#beyond-labor"),
    );
    this.app.get("/swarm-architecture.html", (req, res) =>
      res.redirect("/knowledge-base.html#swarm-architecture"),
    );
    this.app.get("/hardware-sovereignty.html", (req, res) =>
      res.redirect("/knowledge-base.html#hardware-sovereignty"),
    );
    this.app.get("/cognitive-agency.html", (req, res) =>
      res.redirect("/knowledge-base.html#cognitive-agency"),
    );
    this.app.get("/attention-to-intention.html", (req, res) =>
      res.redirect("/knowledge-base.html#intention-economy"),
    );
    this.app.get("/sovereign-action.html", (req, res) =>
      res.redirect("/knowledge-base.html#intention-economy"),
    );
    this.app.get("/readme.html", (req, res) => res.redirect("/README.md"));
    this.app.get("/tokenizer-demo", (req, res) =>
      res.sendFile(
        path.join(__dirname, "../demotokenizer/tokenizationDemo.html"),
      ),
    );
    this.app.get("/embedding-demo", (req, res) =>
      res.sendFile(path.join(__dirname, "../demotokenizer/embeddingDemo.html")),
    );
    this.app.get("/attention-demo", (req, res) =>
      res.sendFile(
        path.join(__dirname, "../demotokenizer/multiheadattentionDemo.html"),
      ),
    );
    this.app.get("/embedding-render", (req, res) =>
      res.sendFile(
        path.join(__dirname, "../demotokenizer/render_embedding.html"),
      ),
    );
    this.app.get("/positional-encoding", (req, res) =>
      res.sendFile(
        path.join(__dirname, "../demotokenizer/codage_posembedded.html"),
      ),
    );
    this.app.get("/display-3d", (req, res) =>
      res.sendFile(path.join(__dirname, "../demotokenizer/display_3d.html")),
    );
    this.app.get("/rag-demo", (req, res) =>
      res.sendFile(
        path.join(__dirname, "../demotokenizer/ragchromadbwithprev.html"),
      ),
    );
    this.app.get("/scroll-screen", (req, res) =>
      res.sendFile(path.join(__dirname, "../demotokenizer/scrolscreen.html")),
    );
    this.app.get("/training-consolidation.html", (req, res) =>
      res.sendFile(path.join(__dirname, "../training-consolidation.html")),
    );
    this.app.get("/test.html", (req, res) =>
      res.sendFile(path.join(__dirname, "../test.html")),
    );
    this.app.get("/test-hardware.html", (req, res) =>
      res.sendFile(path.join(__dirname, "../test-hardware.html")),
    );
    this.app.get("/test-inference.html", (req, res) =>
      res.sendFile(path.join(__dirname, "../test-inference.html")),
    );
  }

  setupWebSocket() {
    this.wss.on("connection", (ws, req) => {
      // Prefer the auth subprotocol (not logged) and fall back to the query
      // param for compatibility. Query strings leak into proxy/access logs, so
      // the header form is preferred.
      const url = new URL(req.url, `http://${req.headers.host}`);
      const token =
        (req.headers["sec-websocket-protocol"] || "")
          .split(",")
          .map((s) => s.trim())
          .find(Boolean) || url.searchParams.get("token");

      if (!tokensMatch(token, this.authToken)) {
        ws.close(4001, "Unauthorized");
        return;
      }

      const clientIp =
        req.headers["x-forwarded-for"] || req.socket.remoteAddress;

      // Cap concurrent connections per IP to bound connection-flood growth.
      const perIpLimit = Number(process.env.MAX_CONN_PER_IP || 20);
      let ipCount = 0;
      this.clients.forEach((c) => {
        if (c.ip === clientIp) ipCount++;
      });
      if (ipCount >= perIpLimit) {
        ws.close(4002, "Too many connections");
        return;
      }

      const clientId = `ws_${Date.now()}_${crypto.randomBytes(6).toString("hex")}`;

      this.clients.set(clientId, {
        ws,
        ip: clientIp,
        connectedAt: new Date(),
        lastActivity: Date.now(),
      });
      this.pushInitialData(ws);

      ws.on("message", (data) => {
        try {
          const message = JSON.parse(data.toString(), safeReviver);
          // handleWebSocketMessage may dispatch async work; swallow rejections
          // so a single bad message can't crash the process.
          Promise.resolve(
            this.handleWebSocketMessage(clientId, message, clientIp),
          ).catch((err) => console.error("❌ WS handler error:", err));
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
      bestModel: this.bestModel
        ? { loss: this.bestModel.loss, epoch: this.bestModel.epoch }
        : null,
      neighbors: neighbors.slice(-10),
    };
    if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(message));
  }

  handleWebSocketMessage(clientId, message, ip) {
    const client = this.clients.get(clientId);
    if (client) client.lastActivity = Date.now();

    // Strict Whitelisting Validation
    const VALID_SCHEMAS = {
      register_training: { data: "object", timestamp: "number" },
      status_update: {
        clientId: "string",
        metrics: "object",
        neighbors: "array",
        timestamp: "number",
      },
      model_update: {
        clientId: "string",
        modelData: "string",
        loss: "number",
        epoch: "number",
        timestamp: "number",
      },
      PEER_MESSAGE: {
        from: "string",
        to: "string",
        data: "object",
        timestamp: "number",
        messageId: "string",
      },
      BROADCAST: {
        from: "string",
        data: "object",
        timestamp: "number",
        messageId: "string",
      },
    };

    const schema = VALID_SCHEMAS[message.type];
    if (!schema) {
      this.db.log(
        `Rejected unknown message type: ${message.type} from ${ip}`,
        "warn",
      );
      return;
    }

    // Check types and presence
    for (const [key, expectedType] of Object.entries(schema)) {
      const val = message[key];
      if (
        val === undefined ||
        (expectedType === "array"
          ? !Array.isArray(val)
          : typeof val !== expectedType)
      ) {
        this.db.log(
          `Rejected malformed ${message.type} from ${ip}: invalid field ${key}`,
          "warn",
        );
        return;
      }
    }

    // Strict field check (no extra fields)
    const allowedKeys = new Set([...Object.keys(schema), "type"]);
    for (const key of Object.keys(message)) {
      if (!allowedKeys.has(key)) {
        this.db.log(
          `Rejected ${message.type} from ${ip}: unexpected field ${key}`,
          "warn",
        );
        return;
      }
    }

    switch (message.type) {
      case "register_training": {
        const safeData = Sanitizer.sanitize(message.data);
        this.trainingClients.set(clientId, {
          ...safeData,
          ip,
          lastSeen: Date.now(),
          ws: client.ws,
        });
        this.db.updateNeighbor(clientId, {
          ...safeData,
          ip,
          peerId: clientId,
        });
        this.db.log(`Client registered: ${clientId} from IP: ${ip}`);
        break;
      }
      case "status_update":
        if (this.trainingClients.has(clientId)) {
          const clientData = this.trainingClients.get(clientId);
          clientData.lastSeen = Date.now();
          clientData.ip = ip; // Update IP if it changed

          if (message.neighbors) {
            message.neighbors.slice(0, 50).forEach((rawN) => {
              const n = Sanitizer.sanitize(rawN);
              if (n.peerId)
                this.db.updateNeighbor(n.peerId, {
                  ...n,
                  discoveredVia: clientId,
                });
            });
          }

          if (message.metrics) {
            const safeMetrics = Sanitizer.sanitizeMetrics(message.metrics);
            this.db.log(`Stats ${clientId} (${ip}): loss=${safeMetrics.loss}`);
            this.db.updateNeighbor(clientId, {
              metrics: safeMetrics,
              ip,
              lastSeen: Date.now(),
            });
          }
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
    // Reject non-finite losses (NaN/Infinity pass a typeof "number" check but
    // would corrupt the "is better" comparison forever).
    if (!Number.isFinite(loss) || typeof modelData !== "string") return false;

    const safeClientId = String(clientId).replace(/[^a-zA-Z0-9_-]/g, "_");
    const isBetter = !this.bestModel || loss < this.bestModel.loss;
    if (!isBetter) return false;

    const modelPath = path.join(
      this.modelsDir,
      `model_${Date.now()}_${safeClientId}.pt`,
    );
    try {
      const b64 = modelData.startsWith("data:")
        ? modelData.split(",")[1]
        : modelData;
      const modelBuffer = Buffer.from(b64, "base64");

      // Cap decoded size to prevent disk-fill DoS.
      if (modelBuffer.length === 0 || modelBuffer.length > MAX_MODEL_BYTES) {
        this.db.log(
          `Rejected model from ${safeClientId}: size ${modelBuffer.length} bytes`,
          "warn",
        );
        return false;
      }

      fs.writeFileSync(modelPath, modelBuffer);
      fs.writeFileSync(this.latestModelPath, modelBuffer);
      this.pruneModelFiles();

      this.bestModel = {
        path: this.latestModelPath,
        size: modelBuffer.length,
        timestamp: new Date(),
        loss,
        epoch,
        clientId,
      };
      this.db.addModel({ loss, epoch, clientId });
      this.db.log(`New best model: ${loss}`);
      return true;
    } catch (error) {
      console.error("❌ Save Error:", error);
    }
    return false;
  }

  // Keep only the most recent MAX_MODEL_FILES per-client model files on disk.
  pruneModelFiles() {
    try {
      const files = fs
        .readdirSync(this.modelsDir)
        .filter((f) => f.startsWith("model_") && f.endsWith(".pt"))
        .map((f) => ({
          f,
          t: fs.statSync(path.join(this.modelsDir, f)).mtimeMs,
        }))
        .sort((a, b) => b.t - a.t);
      for (const { f } of files.slice(MAX_MODEL_FILES)) {
        fs.unlinkSync(path.join(this.modelsDir, f));
      }
    } catch (e) {
      console.error("❌ Prune error:", e);
    }
  }

  broadcastNewBestModel() {
    if (!this.bestModel) return;
    const msg = JSON.stringify({
      type: "new_best_model",
      model: {
        loss: this.bestModel.loss,
        epoch: this.bestModel.epoch,
        clientId: this.bestModel.clientId,
      },
    });
    this.clients.forEach(
      (c) => c.ws.readyState === c.ws.OPEN && c.ws.send(msg),
    );
  }

  startModelEvaluation() {
    setInterval(() => {
      const now = Date.now();
      // Reap stale training clients.
      this.trainingClients.forEach((c, id) => {
        if (now - c.lastSeen > 30000) {
          this.trainingClients.delete(id);
          const wsC = this.clients.get(id);
          if (wsC) wsC.ws.close();
        }
      });
      // Reap any connected client that has been idle too long, even if it never
      // registered as a training client (bounds unbounded clients-map growth).
      this.clients.forEach((c, id) => {
        if (now - c.lastActivity > 120000) {
          try {
            c.ws.close();
          } catch {
            /* already closed */
          }
          this.clients.delete(id);
          this.trainingClients.delete(id);
        }
      });
    }, 10000);
  }

  start(port = 3001) {
    this.server.listen(port, "0.0.0.0", () =>
      console.log(`🚀 Server running on port ${port}`),
    );
  }
}

const server = new ModelConsolidationServer();
server.start(process.env.PORT || 3001);

export { ModelConsolidationServer };
