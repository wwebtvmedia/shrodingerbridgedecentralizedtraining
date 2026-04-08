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
      // In production, this would connect to actual Cloudflare Tunnel

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

      // Attempt reconnect
      await this.attemptReconnect();
    }
  }

  async simulateConnection() {
    // Simulate connection delay
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // Create simulated WebSocket
    this.ws = {
      send: (data) => {
        console.log("📤 Simulated send:", data.substring(0, 100) + "...");
        this.stats.messagesSent++;
        this.stats.bytesSent += data.length;

        // Simulate network delay
        setTimeout(() => {
          this.simulateMessageReceive(data);
        }, 50);
      },
      close: () => {
        this.isConnected = false;
        this.connectionState = "disconnected";
      },
    };

    // Simulate peer discovery
    setTimeout(() => {
      this.simulatePeerDiscovery();
    }, 2000);
  }

  simulateMessageReceive(data) {
    try {
      const message = JSON.parse(data);

      // Update stats
      this.stats.messagesReceived++;
      this.stats.bytesReceived += data.length;

      // Handle different message types
      this.handleTunnelMessage(message);
    } catch (error) {
      console.error("Error parsing simulated message:", error);
    }
  }

  simulatePeerDiscovery() {
    // Simulate discovering 2-4 peers
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

    // Clear heartbeat
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    // Close all peer connections
    for (const [peerId] of this.peers) {
      this.handlePeerDisconnected(peerId);
    }

    this.peers.clear();

    console.log("✅ Disconnected from tunnel");
    this.emit("disconnected");
  }

  async sendToPeer(peerId, message) {
    if (!this.isConnected) {
      // Queue message for when we reconnect
      this.messageQueue.push({ peerId, message, timestamp: Date.now() });
      console.log(
        `📨 Queued message for ${peerId} (queue size: ${this.messageQueue.length})`,
      );
      return false;
    }

    const peer = this.peers.get(peerId);
    if (!peer) {
      console.warn(`Peer ${peerId} not found`);
      return false;
    }

    try {
      const messageStr = JSON.stringify({
        type: "PEER_MESSAGE",
        from: this.config.tunnelId,
        to: peerId,
        data: message,
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

    const peerIds = Array.from(this.peers.keys());
    if (excludeSelf) {
      // Exclude self from broadcast
      // In tunnel, all peers receive broadcast
    }

    const broadcastMessage = JSON.stringify({
      type: "BROADCAST",
      from: this.config.tunnelId,
      data: message,
      timestamp: Date.now(),
      messageId: this.generateMessageId(),
    });

    this.ws.send(broadcastMessage);

    console.log(`📢 Broadcast to ${peerIds.length} peers`);
    this.emit("broadcast:sent", { peerCount: peerIds.length });
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

      case "HEARTBEAT":
        this.handleHeartbeat(message);
        break;

      case "HEARTBEAT_RESPONSE":
        // Heartbeat responses are handled internally by the tunnel server
        // but we acknowledge them here to avoid warnings.
        break;

      case "TUNNEL_STATS":
        this.handleTunnelStats(message);
        break;

      default:
        console.warn(`Unknown tunnel message type: ${message.type}`);
    }
  }

  handlePeerConnected(peerId, metadata) {
    if (peerId === this.config.tunnelId) return; // Don't add self

    console.log(`👋 Peer connected via tunnel: ${peerId}`);

    this.peers.set(peerId, {
      connection: null, // Tunnel handles connection
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
    console.log(`📨 Message from ${fromPeerId}:`, data.type || "data");

    this.emit("peer:message", { from: fromPeerId, data });

    // Forward to application if there's a callback
    if (this.onPeerMessage) {
      this.onPeerMessage(fromPeerId, data);
    }
  }

  handleBroadcast(fromPeerId, data) {
    if (fromPeerId === this.config.tunnelId) return; // Ignore own broadcasts

    console.log(`📢 Broadcast from ${fromPeerId}:`, data.type || "data");

    this.emit("broadcast:received", { from: fromPeerId, data });
  }

  handleHeartbeat(message) {
    // Update connection health
    this.connectionState = "connected";

    // Send response
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
    console.log("📊 Tunnel stats:", stats);
    this.emit("tunnel:stats", stats);
  }

  startHeartbeat() {
    // Send heartbeat every 30 seconds
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
    if (this.reconnectAttempts >= this.config.maxRetries) {
      console.error("Max reconnection attempts reached");
      this.emit("connection:failed");
      return;
    }

    this.reconnectAttempts++;
    const delay = this.config.reconnectInterval * this.reconnectAttempts;

    console.log(
      `🔄 Reconnection attempt ${this.reconnectAttempts} in ${delay}ms`,
    );

    setTimeout(async () => {
      try {
        await this.connect();
      } catch (error) {
        console.error("Reconnection failed:", error);
      }
    }, delay);
  }

  processMessageQueue() {
    if (this.messageQueue.length === 0) return;

    console.log(`Processing ${this.messageQueue.length} queued messages`);

    const failedMessages = [];

    for (const queued of this.messageQueue) {
      const success = this.sendToPeer(queued.peerId, queued.message);
      if (!success) {
        failedMessages.push(queued);
      }
    }

    this.messageQueue = failedMessages;

    if (failedMessages.length > 0) {
      console.log(`${failedMessages.length} messages failed to send`);
    }
  }

  generateMessageId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
  }

  // Event system
  on(event, callback) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event).push(callback);
  }

  off(event, callback) {
    if (!this.eventListeners.has(event)) return;

    const listeners = this.eventListeners.get(event);
    const index = listeners.indexOf(callback);
    if (index > -1) {
      listeners.splice(index, 1);
    }
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

  // Peer management
  getPeers() {
    return Array.from(this.peers.entries()).map(([peerId, info]) => ({
      peerId,
      ...info.metadata,
    }));
  }

  getPeerCount() {
    return this.peers.size;
  }

  getPeer(peerId) {
    const peer = this.peers.get(peerId);
    return peer ? { peerId, ...peer.metadata } : null;
  }

  // Statistics
  getStats() {
    return {
      ...this.stats,
      isConnected: this.isConnected,
      connectionState: this.connectionState,
      peerCount: this.peers.size,
      queueSize: this.messageQueue.length,
      reconnectAttempts: this.reconnectAttempts,
    };
  }

  // Tunnel configuration
  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
    console.log("Tunnel config updated");
  }

  // Utility methods for Cloudflare Tunnel API
  async createTunnel() {
    // In production, this would call Cloudflare API to create a tunnel
    console.log("Creating Cloudflare Tunnel...");

    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const tunnelInfo = {
      id: this.config.tunnelId,
      url: `https://${this.config.tunnelId}.tunnel.swarm-training.com`,
      createdAt: new Date().toISOString(),
      status: "active",
    };

    console.log("✅ Tunnel created:", tunnelInfo.url);
    return tunnelInfo;
  }

  async deleteTunnel() {
    // In production, this would call Cloudflare API to delete the tunnel
    console.log("Deleting Cloudflare Tunnel...");

    await new Promise((resolve) => setTimeout(resolve, 1000));

    console.log("✅ Tunnel deleted");
    return true;
  }

  async getTunnelInfo() {
    // Simulate getting tunnel info
    return {
      id: this.config.tunnelId,
      url: `https://${this.config.tunnelId}.tunnel.swarm-training.com`,
      peers: this.peers.size,
      status: this.isConnected ? "connected" : "disconnected",
      stats: this.getStats(),
    };
  }
}

// Factory function for creating tunnel with Cloudflare credentials
async function createCloudflareTunnel(credentials = {}) {
  const tunnel = new CloudflareTunnel({
    tunnelUrl: credentials.tunnelUrl || "https://tunnel.swarm-training.com",
    apiKey: credentials.apiKey || "",
    accountId: credentials.accountId || "",
    tunnelName: credentials.tunnelName || `swarm-tunnel-${Date.now()}`,
  });

  // Create tunnel if credentials provided
  if (credentials.apiKey && credentials.accountId) {
    await tunnel.createTunnel();
  }

  return tunnel;
}

export { CloudflareTunnel, createCloudflareTunnel };
