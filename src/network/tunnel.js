import { Sanitizer } from "../utils/sanitizer.js";
import { Validator } from "../utils/validator.js";

class CloudflareTunnel {
  constructor(config = {}) {
    this.config = {
      tunnelUrl: config.tunnelUrl || "https://tunnel.swarm-training.com",
      apiKey: config.apiKey || "",
      // No shipped default secret — must be supplied via config/env.
      authToken: config.authToken || "",
      tunnelId: config.tunnelId || this.generateTunnelId(),
      reconnectInterval: config.reconnectInterval || 5000,
      maxRetries: config.maxRetries || 3,
      ...config,
    };

    this.ws = null;
    this.peers = new Map(); // peerId -> {connection, metadata}
    this.messageQueue = [];
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.eventListeners = new Map();

    // Connection state
    this.connectionState = "disconnected";

    // Stats
    this.stats = {
      messagesSent: 0,
      messagesReceived: 0,
      bytesSent: 0,
      bytesReceived: 0,
      connections: 0,
      errors: 0,
    };
  }

  generateTunnelId() {
    return (
      "tunnel_" +
      Math.random().toString(36).substring(2, 15) +
      Math.random().toString(36).substring(2, 15)
    );
  }

  async connect() {
    if (this.isConnected) return;

    console.log(`🌐 Connecting to Cloudflare Tunnel: ${this.config.tunnelUrl}`);
    this.connectionState = "connecting";

    return new Promise((resolve, reject) => {
      // Pass the auth token as a WebSocket subprotocol rather than a query
      // string — URLs are logged by proxies/servers/history, leaking the secret.
      let settled = false;
      const settle = (fn, arg) => {
        if (settled) return;
        settled = true;
        fn(arg);
      };

      let ws;
      try {
        ws = this.config.authToken
          ? new WebSocket(this.config.tunnelUrl, [this.config.authToken])
          : new WebSocket(this.config.tunnelUrl);
      } catch (error) {
        this.connectionState = "error";
        this.stats.errors++;
        return settle(reject, error);
      }

      ws.onopen = () => {
        this.ws = ws;
        this.isConnected = true;
        this.connectionState = "connected";
        this.reconnectAttempts = 0;

        console.log("✅ Connected to Cloudflare Tunnel");
        this.emit("connected", { tunnelId: this.config.tunnelId });

        // We register only after the server issues our signed identity (the
        // `identity` message). Registration then carries the host-signed
        // credential, and the directory uses the host-assigned peer id.
        this.startHeartbeat();
        this.processMessageQueue();
        settle(resolve);
      };

      ws.onmessage = (event) => {
        try {
          const data = event.data;
          this.stats.messagesReceived++;
          // event.data may be a Blob/ArrayBuffer for binary frames.
          if (typeof data === "string") {
            this.stats.bytesReceived += data.length;
            this.handleTunnelMessage(JSON.parse(data));
          } else {
            this.stats.bytesReceived += data.byteLength || 0;
          }
        } catch (error) {
          console.error("Error parsing message:", error);
        }
      };

      ws.onclose = (event) => {
        this.isConnected = false;
        this.connectionState = "disconnected";
        console.log(`⚠️ Tunnel connection closed (Code: ${event.code})`);
        this.emit("disconnected");

        // If the socket closed before ever opening, settle the pending promise
        // so callers awaiting connect() don't hang forever.
        settle(
          reject,
          new Error(`Tunnel closed before open (code ${event.code})`),
        );

        if (event.code !== 1000 && event.code !== 1001) {
          this.attemptReconnect();
        }
      };

      ws.onerror = (error) => {
        console.error("❌ Tunnel connection error:", error);
        this.connectionState = "error";
        this.stats.errors++;
        this.emit("error", { error: "WebSocket Error" });
        settle(reject, new Error("WebSocket connection failed"));
      };
    });
  }

  async disconnect() {
    console.log("🔌 Disconnecting from tunnel...");
    this.connectionState = "disconnecting";
    if (this.ws) {
      // Detach handlers before closing so a late close/error event can't fire
      // app callbacks (e.g. trigger a reconnect) after an intentional teardown.
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      try {
        this.ws.close(1000, "client disconnect");
      } catch {
        /* already closing */
      }
      this.ws = null;
    }
    this.isConnected = false;
    this.connectionState = "disconnected";
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    for (const [peerId] of this.peers) {
      this.handlePeerDisconnected(peerId);
    }
    this.peers.clear();
    console.log("✅ Disconnected from tunnel");
    this.emit("disconnected");
  }

  async sendToPeer(peerId, message) {
    if (!this.isConnected) {
      // Bound the queue so a long disconnection can't grow it without limit.
      const MAX_QUEUE = 1000;
      if (this.messageQueue.length >= MAX_QUEUE) this.messageQueue.shift();
      this.messageQueue.push({ peerId, message, timestamp: Date.now() });
      return false;
    }
    const peer = this.peers.get(peerId);
    if (!peer) return false;

    try {
      const messageStr = JSON.stringify({
        type: "PEER_MESSAGE",
        from: this.config.tunnelId,
        to: peerId,
        data: Sanitizer.sanitize(message),
        timestamp: Date.now(),
        messageId: this.generateMessageId(),
      });
      this.ws.send(messageStr);
      this.emit("message:sent", { peerId, messageId: message.messageId });
      return true;
    } catch (error) {
      console.error(`Failed to send to ${peerId}:`, error);
      this.stats.errors++;
      return false;
    }
  }

  async broadcast(message, excludeSelf = true) {
    if (!this.isConnected) return;
    const broadcastMessage = JSON.stringify({
      type: "BROADCAST",
      from: this.config.tunnelId,
      data: Sanitizer.sanitize(message),
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
    });
    this.ws.send(broadcastMessage);
    this.emit("broadcast:sent", { peerCount: this.peers.size });
  }

  /**
   * Announce this client to the signaling server's peer directory. Sent on
   * every (re)connect; the server keys presence by our tunnelId.
   */
  register() {
    if (!this.isConnected || !this.ws) return;
    try {
      const data = {
        peerId: this.config.tunnelId,
        name: this.config.tunnelId,
      };
      // Echo the host-issued credential so the server can verify we registered
      // under the identity it minted for us.
      if (this.identity) {
        data.issuedAt = this.identity.issuedAt;
        data.signature = this.identity.signature;
      }
      this.ws.send(
        JSON.stringify({
          type: "register_training",
          data,
          timestamp: Date.now(),
        }),
      );
    } catch (error) {
      console.error("Failed to register with tunnel server:", error);
    }
  }

  /**
   * Periodically sends statistics and reachable neighbors to the server.
   */
  async sendStatusUpdate(metrics, neighbors) {
    if (!this.isConnected || !this.ws) return;
    const message = JSON.stringify({
      type: "status_update",
      clientId: this.config.tunnelId,
      metrics: Sanitizer.sanitizeMetrics(metrics),
      neighbors: neighbors ? Sanitizer.sanitize(neighbors) : this.getPeers(),
      timestamp: Date.now(),
    });
    this.ws.send(message);
  }

  /**
   * Sends final data to the server before closing.
   */
  async sendFinalSync(metrics, neighbors) {
    if (!this.isConnected || !this.ws) return;
    const message = JSON.stringify({
      type: "final_sync",
      clientId: this.config.tunnelId,
      metrics: Sanitizer.sanitizeMetrics(metrics),
      neighbors: neighbors ? Sanitizer.sanitize(neighbors) : this.getPeers(),
      timestamp: Date.now(),
    });
    this.ws.send(message);
    console.log("📤 Final sync sent to server");
  }

  handleTunnelMessage(message) {
    if (!message || !message.type) return;

    // Strict Whitelisting Validation
    if (!Validator.validate(message.type, message)) {
      console.warn(
        `Tunnel: Rejected malformed or unexpected ${message.type} message.`,
      );
      return;
    }

    switch (message.type) {
      case "identity":
        this.handleIdentity(message);
        break;
      case "PEER_CONNECTED":
        this.handlePeerConnected(message.peerId, message.metadata);
        break;
      case "PEER_DISCONNECTED":
        this.handlePeerDisconnected(message.peerId);
        break;
      case "PEER_MESSAGE":
        this.handlePeerMessage(message.from, message.data);
        break;
      case "BROADCAST":
        this.handleBroadcast(message.from, message.data);
        break;
      case "initial_sync":
        this.handleInitialSync(message);
        break;
      case "HEARTBEAT":
        this.handleHeartbeat(message);
        break;
      case "HEARTBEAT_RESPONSE":
        this.connectionState = "connected";
        break;
      case "TUNNEL_STATS":
        this.handleTunnelStats(message);
        break;
      case "PEER_RESEARCH_REQUEST":
        this.emit("peer:research_request", {
          from: message.from,
          data: message.data,
        });
        this.handleResearchRequest(message.from, message.data);
        break;
      case "PEER_RESEARCH_RESPONSE":
        this.emit("peer:research_result", {
          from: message.from,
          data: message.data,
        });
        this.handleResearchResponse(message.from, message.data);
        break;
      default:
        if (message && message.type)
          console.warn(`Unknown tunnel message type: ${message.type}`);
    }
  }

  /**
   * Adopt the host-issued, signed identity. The server mints this id; we use it
   * as our public peer id, then register so we appear in the shared directory.
   */
  handleIdentity(message) {
    this.identity = {
      peerId: message.peerId,
      issuedAt: message.issuedAt,
      signature: message.signature,
    };
    this.config.tunnelId = message.peerId;
    console.log(`🪪 Host-issued peer identity: ${message.peerId}`);
    this.emit("identity", this.identity);
    this.register();
  }

  handleInitialSync(message) {
    console.log("📥 Received initial sync from server");
    if (message.bestModel) {
      this.emit("model:initial", message.bestModel);
    }
    if (message.neighbors && message.neighbors.length > 0) {
      message.neighbors.forEach((n) =>
        this.handlePeerConnected(n.peerId, n.metadata || n),
      );
    }
  }

  handlePeerConnected(peerId, metadata) {
    if (peerId === this.config.tunnelId) return;
    console.log(`👋 Peer connected via tunnel: ${peerId}`);
    this.peers.set(peerId, {
      connection: null,
      metadata: {
        ...metadata,
        connectedAt: Date.now(),
        via: "cloudflare-tunnel",
      },
    });
    this.stats.connections++;
    this.emit("peer:connected", { peerId, metadata });
  }

  handlePeerDisconnected(peerId) {
    if (!this.peers.has(peerId)) return;
    console.log(`👋 Peer disconnected from tunnel: ${peerId}`);
    this.peers.delete(peerId);
    this.emit("peer:disconnected", { peerId });
  }

  handlePeerMessage(fromPeerId, data) {
    this.emit("peer:message", { from: fromPeerId, data });
  }

  handleBroadcast(fromPeerId, data) {
    if (fromPeerId === this.config.tunnelId) return;
    this.emit("broadcast:received", { from: fromPeerId, data });
  }

  handleHeartbeat(message) {
    this.connectionState = "connected";
    if (this.ws) {
      this.ws.send(
        JSON.stringify({
          type: "HEARTBEAT_RESPONSE",
          tunnelId: this.config.tunnelId,
          timestamp: Date.now(),
        }),
      );
    }
  }

  handleTunnelStats(stats) {
    this.emit("tunnel:stats", stats);
  }

  async researchNeighbors() {
    console.log("🔍 Broadcasting PEER_RESEARCH_REQUEST...");
    return this.broadcast({
      type: "PEER_RESEARCH_REQUEST",
      timestamp: Date.now(),
    });
  }

  handleResearchRequest(fromPeerId, data) {
    this.sendToPeer(fromPeerId, {
      type: "PEER_RESEARCH_RESPONSE",
      status: {
        via: "cloudflare-tunnel",
        peerCount: this.peers.size,
        timestamp: Date.now(),
      },
    });
  }

  handleResearchResponse(fromPeerId, data) {
    this.emit("peer:research_result", {
      peerId: fromPeerId,
      status: data.status,
    });
  }

  startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected && this.ws) {
        this.ws.send(
          JSON.stringify({
            type: "HEARTBEAT",
            tunnelId: this.config.tunnelId,
            timestamp: Date.now(),
          }),
        );
      }
    }, 30000);
  }

  async attemptReconnect() {
    if (this.reconnectAttempts >= this.config.maxRetries) return;
    this.reconnectAttempts++;
    const delay = this.config.reconnectInterval * this.reconnectAttempts;
    setTimeout(async () => {
      try {
        await this.connect();
      } catch (error) {
        console.error("Reconnection failed:", error);
      }
    }, delay);
  }

  async processMessageQueue() {
    if (this.messageQueue.length === 0) return;
    // Drain the current queue; sendToPeer is async and returns a boolean, so we
    // must await it (a Promise is always truthy). Take a snapshot and clear so
    // re-queued failures from sendToPeer aren't re-processed in this pass.
    const pending = this.messageQueue;
    this.messageQueue = [];
    const failed = [];
    for (const queued of pending) {
      const success = await this.sendToPeer(queued.peerId, queued.message);
      if (!success) failed.push(queued);
    }
    // sendToPeer already re-queues when disconnected; only add back genuine
    // send failures that weren't re-queued.
    for (const f of failed) {
      if (!this.messageQueue.includes(f)) this.messageQueue.push(f);
    }
  }

  generateMessageId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
  }

  on(event, callback) {
    if (!this.eventListeners.has(event)) this.eventListeners.set(event, []);
    this.eventListeners.get(event).push(callback);
  }

  off(event, callback) {
    if (!this.eventListeners.has(event)) return;
    const listeners = this.eventListeners.get(event);
    const index = listeners.indexOf(callback);
    if (index > -1) listeners.splice(index, 1);
  }

  emit(event, data) {
    if (!this.eventListeners.has(event)) return;
    const listeners = this.eventListeners.get(event);
    for (const listener of listeners) {
      try {
        listener(data);
      } catch (error) {
        console.error(`Error in event listener for ${event}:`, error);
      }
    }
  }

  getPeers() {
    return Array.from(this.peers.entries()).map(([peerId, info]) => ({
      peerId,
      ...info.metadata,
    }));
  }

  getPeerCount() {
    return this.peers.size;
  }
}

async function createCloudflareTunnel(credentials = {}) {
  const tunnel = new CloudflareTunnel({
    tunnelUrl: credentials.tunnelUrl || "https://tunnel.swarm-training.com",
    apiKey: credentials.apiKey || "",
    accountId: credentials.accountId || "",
    tunnelName: credentials.tunnelName || `swarm-tunnel-${Date.now()}`,
  });
  return tunnel;
}

export { CloudflareTunnel, createCloudflareTunnel };
