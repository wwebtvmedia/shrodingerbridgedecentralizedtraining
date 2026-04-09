import { Sanitizer } from "../utils/sanitizer.js";

class CloudflareTunnel {
  constructor(config = {}) {
    this.config = {
      tunnelUrl: config.tunnelUrl || "https://tunnel.swarm-training.com",
      apiKey: config.apiKey || "",
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

    try {
      // For prototype, we'll simulate WebSocket connection
      await this.simulateConnection();

      this.isConnected = true;
      this.connectionState = "connected";
      this.reconnectAttempts = 0;

      console.log("✅ Connected to Cloudflare Tunnel");
      this.emit("connected", { tunnelId: this.config.tunnelId });

      // Start heartbeat
      this.startHeartbeat();

      // Process queued messages
      this.processMessageQueue();
    } catch (error) {
      console.error("❌ Tunnel connection failed:", error);
      this.connectionState = "error";
      this.stats.errors++;
      this.emit("error", { error: error.message });
      await this.attemptReconnect();
    }
  }

  async simulateConnection() {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    this.ws = {
      send: (data) => {
        console.log("📤 Simulated send:", data.substring(0, 100) + "...");
        this.stats.messagesSent++;
        this.stats.bytesSent += data.length;
        setTimeout(() => this.simulateMessageReceive(data), 50);
      },
      close: () => {
        this.isConnected = false;
        this.connectionState = "disconnected";
      },
      readyState: 1 // OPEN
    };
    setTimeout(() => this.simulatePeerDiscovery(), 2000);
  }

  simulateMessageReceive(data) {
    try {
      const message = JSON.parse(data);
      this.stats.messagesReceived++;
      this.stats.bytesReceived += data.length;
      this.handleTunnelMessage(message);
    } catch (error) {
      console.error("Error parsing simulated message:", error);
    }
  }

  simulatePeerDiscovery() {
    const peerCount = 2 + Math.floor(Math.random() * 3);
    for (let i = 0; i < peerCount; i++) {
      const peerId = `cf-peer-${i}`;
      this.handlePeerConnected(peerId, {
        region: ["us-east", "eu-west", "asia-southeast"][i % 3],
        latency: 50 + Math.random() * 100,
      });
    }
  }

  async disconnect() {
    console.log("🔌 Disconnecting from tunnel...");
    this.connectionState = "disconnecting";
    if (this.ws) {
      this.ws.close();
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
   * Periodically sends statistics and reachable neighbors to the server.
   */
  async sendStatusUpdate(metrics, neighbors) {
    if (!this.isConnected || !this.ws) return;
    const message = JSON.stringify({
      type: "status_update",
      clientId: this.config.tunnelId,
      metrics: Sanitizer.sanitizeMetrics(metrics),
      neighbors: neighbors ? Sanitizer.sanitize(neighbors) : this.getPeers(),
      timestamp: Date.now()
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
      timestamp: Date.now()
    });
    this.ws.send(message);
    console.log("📤 Final sync sent to server");
  }

  handleTunnelMessage(message) {
    switch (message.type) {
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
        this.handleResearchRequest(message.from, message.data);
        break;
      case "PEER_RESEARCH_RESPONSE":
        this.handleResearchResponse(message.from, message.data);
        break;
      default:
        if (message && message.type) console.warn(`Unknown tunnel message type: ${message.type}`);
    }
  }

  handleInitialSync(message) {
    console.log("📥 Received initial sync from server");
    if (message.bestModel) {
      this.emit("model:initial", message.bestModel);
    }
    if (message.neighbors && message.neighbors.length > 0) {
      message.neighbors.forEach(n => this.handlePeerConnected(n.peerId, n.metadata || n));
    }
  }

  handlePeerConnected(peerId, metadata) {
    if (peerId === this.config.tunnelId) return;
    console.log(`👋 Peer connected via tunnel: ${peerId}`);
    this.peers.set(peerId, {
      connection: null,
      metadata: { ...metadata, connectedAt: Date.now(), via: "cloudflare-tunnel" },
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
      this.ws.send(JSON.stringify({
        type: "HEARTBEAT_RESPONSE",
        tunnelId: this.config.tunnelId,
        timestamp: Date.now(),
      }));
    }
  }

  handleTunnelStats(stats) {
    this.emit("tunnel:stats", stats);
  }

  async researchNeighbors() {
    return this.broadcast({ type: "PEER_RESEARCH_REQUEST", timestamp: Date.now() });
  }

  handleResearchRequest(fromPeerId, data) {
    this.sendToPeer(fromPeerId, {
      type: "PEER_RESEARCH_RESPONSE",
      status: { via: "cloudflare-tunnel", peerCount: this.peers.size, timestamp: Date.now() }
    });
  }

  handleResearchResponse(fromPeerId, data) {
    this.emit("peer:research_result", { peerId: fromPeerId, status: data.status });
  }

  startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected && this.ws) {
        this.ws.send(JSON.stringify({
          type: "HEARTBEAT",
          tunnelId: this.config.tunnelId,
          timestamp: Date.now(),
        }));
      }
    }, 30000);
  }

  async attemptReconnect() {
    if (this.reconnectAttempts >= this.config.maxRetries) return;
    this.reconnectAttempts++;
    const delay = this.config.reconnectInterval * this.reconnectAttempts;
    setTimeout(async () => {
      try { await this.connect(); } catch (error) { console.error("Reconnection failed:", error); }
    }, delay);
  }

  processMessageQueue() {
    if (this.messageQueue.length === 0) return;
    const failedMessages = [];
    for (const queued of this.messageQueue) {
      const success = this.sendToPeer(queued.peerId, queued.message);
      if (!success) failedMessages.push(queued);
    }
    this.messageQueue = failedMessages;
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
      try { listener(data); } catch (error) { console.error(`Error in event listener for ${event}:`, error); }
    }
  }

  getPeers() {
    return Array.from(this.peers.entries()).map(([peerId, info]) => ({ peerId, ...info.metadata }));
  }

  getPeerCount() { return this.peers.size; }
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
