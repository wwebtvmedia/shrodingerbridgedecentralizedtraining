import { v4 as uuidv4 } from "uuid";

class PeerNetwork {
  constructor() {
    this.id = uuidv4();
    this.peers = new Map(); // peerId -> connection
    this.dataChannels = new Map(); // peerId -> dataChannel
    this.knownModels = new Map(); // modelHash -> {data, peerId, timestamp}
    this.gossipCache = new Map(); // messageId -> timestamp

    // Callbacks
    this.onPeerUpdate = null;
    this.onModelReceived = null;
    this.onMessageReceived = null;

    // Configuration
    this.config = {
      maxPeers: 10,
      gossipFanout: 3,
      gossipTTL: 5,
      modelCacheSize: 20,
    };

    // STUN servers
    this.iceServers = [
      { urls: "stun:stun.l.google.com:19302" },
      { urls: "stun:global.stun.twilio.com:3478" },
    ];

    // Signaling (simulated for prototype)
    this.signalingChannel = this.createSignalingChannel();
  }

  async connect() {
    console.log(`🔗 Connecting peer ${this.id.substring(0, 8)}...`);

    // In a real implementation, this would connect to a signaling server
    // For prototype, we'll simulate connections

    // Simulate discovering some peers
    await this.simulatePeerDiscovery();

    return true;
  }

  async simulatePeerDiscovery() {
    // Simulate finding 2-4 peers
    const peerCount = 2 + Math.floor(Math.random() * 3);

    for (let i = 0; i < peerCount; i++) {
      const peerId = `sim-peer-${i}`;
      await this.connectToPeer(peerId);
    }

    console.log(`✅ Connected to ${this.peers.size} simulated peers`);
  }

  async connectToPeer(peerId) {
    if (this.peers.has(peerId) || peerId === this.id) {
      return;
    }

    if (this.peers.size >= this.config.maxPeers) {
      console.warn("Max peers reached");
      return;
    }

    try {
      // Create WebRTC connection
      const connection = new RTCPeerConnection({
        iceServers: this.iceServers,
      });

      // Create data channel
      const dataChannel = connection.createDataChannel("swarm");

      // Setup event handlers
      this.setupDataChannel(dataChannel, peerId);
      this.setupConnectionHandlers(connection, peerId);

      // Store connection
      this.peers.set(peerId, connection);
      this.dataChannels.set(peerId, dataChannel);

      // Simulate connection establishment
      setTimeout(() => {
        this.handlePeerConnected(peerId);
      }, 100);
    } catch (error) {
      console.error(`Failed to connect to peer ${peerId}:`, error);
    }
  }

  setupDataChannel(dataChannel, peerId) {
    dataChannel.onopen = () => {
      console.log(`📨 Data channel open with ${peerId}`);
      this.sendWelcomeMessage(peerId);
    };

    dataChannel.onclose = () => {
      console.log(`📪 Data channel closed with ${peerId}`);
      this.handlePeerDisconnected(peerId);
    };

    dataChannel.onerror = (error) => {
      console.error(`Data channel error with ${peerId}:`, error);
    };

    dataChannel.onmessage = (event) => {
      this.handleMessage(peerId, event.data);
    };
  }

  setupConnectionHandlers(connection, peerId) {
    connection.onicecandidate = (event) => {
      if (event.candidate) {
        // Send ICE candidate to peer (simulated)
      }
    };

    connection.onconnectionstatechange = () => {
      console.log(
        `Connection state with ${peerId}:`,
        connection.connectionState,
      );

      if (
        connection.connectionState === "disconnected" ||
        connection.connectionState === "failed" ||
        connection.connectionState === "closed"
      ) {
        this.handlePeerDisconnected(peerId);
      }
    };
  }

  handlePeerConnected(peerId) {
    console.log(`✅ Peer connected: ${peerId}`);

    // Update UI
    if (this.onPeerUpdate) {
      this.onPeerUpdate(this.peers.size);
    }

    // Share our known models
    this.shareKnownModels(peerId);
  }

  handlePeerDisconnected(peerId) {
    this.peers.delete(peerId);
    this.dataChannels.delete(peerId);

    console.log(`❌ Peer disconnected: ${peerId}`);

    // Update UI
    if (this.onPeerUpdate) {
      this.onPeerUpdate(this.peers.size);
    }
  }

  sendWelcomeMessage(peerId) {
    const welcome = {
      type: "WELCOME",
      peerId: this.id,
      timestamp: Date.now(),
    };

    this.sendToPeer(peerId, welcome);
  }

  async gossip(message) {
    if (!message.id) {
      message.id = uuidv4();
    }

    // Check if we've seen this message recently
    if (this.gossipCache.has(message.id)) {
      return; // Already gossiped
    }

    // Cache the message
    this.gossipCache.set(message.id, Date.now());

    // Forward to other peers
    return this.forwardMessage(message);
  }

  async forwardMessage(message) {
    // Clean old cache entries
    this.cleanGossipCache();

    // Select random peers to gossip to
    const peers = Array.from(this.peers.keys());
    const fanout = Math.min(this.config.gossipFanout, peers.length);
    const selectedPeers = this.selectRandomPeers(peers, fanout);

    // Send to selected peers
    for (const peerId of selectedPeers) {
      this.sendToPeer(peerId, {
        ...message,
        ttl: message.ttl || this.config.gossipTTL,
      });
    }
  }

  async handleMessage(peerId, data) {
    try {
      const message = typeof data === "string" ? JSON.parse(data) : data;

      // Check if we've seen this message recently (deduplication)
      if (message.id && this.gossipCache.has(message.id)) {
        return;
      }

      // Cache the message ID
      if (message.id) {
        this.gossipCache.set(message.id, Date.now());
      }

      // Handle different message types
      switch (message.type) {
        case "WELCOME":
          this.handleWelcome(peerId, message);
          break;

        case "TRAINING_RESULT":
          this.handleTrainingResult(peerId, message);
          break;

        case "MODEL_SHARE":
          this.handleModelShare(peerId, message);
          break;

        case "MODEL_REQUEST":
          this.handleModelRequest(peerId, message);
          break;

        case "GOSSIP":
          this.handleGossip(peerId, message);
          break;

        case "SAMPLE_SHARE":
          this.handleSampleShare(peerId, message);
          break;

        default:
          console.warn(`Unknown message type: ${message.type}`);
      }

      // Forward to callback
      if (this.onMessageReceived) {
        this.onMessageReceived(peerId, message);
      }
    } catch (error) {
      console.error("Error handling message:", error);
    }
  }

  handleWelcome(peerId, message) {
    console.log(`👋 Welcome from ${peerId}`);
    // Could exchange more info here
  }

  handleTrainingResult(peerId, message) {
    // Store the result
    this.knownModels.set(message.modelHash, {
      data: message,
      peerId,
      timestamp: Date.now(),
    });

    // Clean old models
    this.cleanModelCache();

    // Forward via gossip if TTL > 0
    if (message.ttl > 0) {
      this.forwardMessage({
        ...message,
        ttl: message.ttl - 1,
      });
    }
  }

  handleModelShare(peerId, message) {
    // Store the model
    this.knownModels.set(message.modelHash, {
      data: message.modelData,
      peerId,
      timestamp: Date.now(),
    });

    // Notify callback
    if (this.onModelReceived) {
      this.onModelReceived(message);
    }
  }

  handleModelRequest(peerId, message) {
    // Check if we have the requested model
    const model = this.knownModels.get(message.modelHash);

    if (model) {
      this.sendToPeer(peerId, {
        type: "MODEL_SHARE",
        modelHash: message.modelHash,
        modelData: model.data,
        timestamp: Date.now(),
      });
    }
  }

  handleGossip(peerId, message) {
    // Forward if TTL > 0
    if (message.ttl > 0) {
      this.forwardMessage({
        ...message,
        ttl: message.ttl - 1,
      });
    }
  }

  handleSampleShare(peerId, message) {
    // Could display samples in UI
    console.log(`Received samples from ${peerId}`);
  }

  /**
   * Broadcasts a research request to all neighbors.
   * Neighbors will respond with their current status.
   */
  async researchNeighbors() {
    console.log("🔍 Broadcasting neighbor research request...");

    const message = {
      type: "PEER_RESEARCH_REQUEST",
      id: uuidv4(),
      from: this.id,
      timestamp: Date.now(),
      ttl: this.config.gossipTTL,
    };

    // Use gossip to reach all reachable peers
    return this.gossip(message);
  }

  handleResearchRequest(peerId, message) {
    console.log(`🔎 Research request from ${peerId}`);

    // Prepare our status response
    // In a real app, this would include metrics from the trainer
    const response = {
      type: "PEER_RESEARCH_RESPONSE",
      requestId: message.id,
      peerId: this.id,
      timestamp: Date.now(),
      status: {
        isConnected: true,
        peerCount: this.peers.size,
        knownModels: this.knownModels.size,
        uptime: process.uptime ? process.uptime() : 0,
        // Metrics would be injected here in production
        metrics: {
          loss: 0.25 + Math.random() * 0.1,
          epoch: 100,
          accuracy: 0.85,
        },
      },
    };

    // Send directly back to requester if we have a channel,
    // otherwise it would need to be routed back
    this.sendToPeer(peerId, response);

    // Also forward the research request if TTL allows
    if (message.ttl > 0) {
      this.forwardMessage({
        ...message,
        ttl: message.ttl - 1,
      });
    }
  }

  handleResearchResponse(peerId, message) {
    console.log(
      `📊 Received research response from ${peerId}:`,
      message.status,
    );

    // If there's a specific callback for research results
    if (this.onResearchResult) {
      this.onResearchResult(peerId, message.status);
    }
  }

  async getBestModel() {
    if (this.knownModels.size === 0) {
      return null;
    }

    // Find model with lowest loss
    let bestModel = null;
    let bestLoss = Infinity;

    for (const [hash, info] of this.knownModels.entries()) {
      if (info.data.loss < bestLoss) {
        bestLoss = info.data.loss;
        bestModel = {
          ...info.data,
          modelHash: hash,
          peerId: info.peerId,
        };
      }
    }

    return bestModel;
  }

  async requestModel(peerId, modelHash) {
    // Send request
    this.sendToPeer(peerId, {
      type: "MODEL_REQUEST",
      modelHash,
      timestamp: Date.now(),
    });

    // In real implementation, would wait for response
    // For prototype, return simulated model
    return {
      modelHash,
      peerId,
      loss: 0.3 + Math.random() * 0.2,
      epoch: 50 + Math.floor(Math.random() * 100),
    };
  }

  async shareSamples(samples) {
    this.gossip({
      type: "SAMPLE_SHARE",
      ...samples,
      id: uuidv4(),
    });
  }

  shareKnownModels(peerId) {
    // Share a few known models with new peer
    const models = Array.from(this.knownModels.entries()).slice(0, 3);

    for (const [hash, info] of models) {
      this.sendToPeer(peerId, {
        type: "MODEL_SHARE",
        modelHash: hash,
        modelData: info.data,
        timestamp: Date.now(),
      });
    }
  }

  sendToPeer(peerId, message) {
    const dataChannel = this.dataChannels.get(peerId);

    if (dataChannel && dataChannel.readyState === "open") {
      try {
        dataChannel.send(JSON.stringify(message));
      } catch (error) {
        console.error(`Failed to send to ${peerId}:`, error);
      }
    }
  }

  selectRandomPeers(peers, count) {
    const shuffled = [...peers].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  }

  cleanGossipCache() {
    const now = Date.now();
    const maxAge = 60000; // 1 minute

    for (const [id, timestamp] of this.gossipCache.entries()) {
      if (now - timestamp > maxAge) {
        this.gossipCache.delete(id);
      }
    }
  }

  cleanModelCache() {
    if (this.knownModels.size <= this.config.modelCacheSize) {
      return;
    }

    // Remove oldest models
    const entries = Array.from(this.knownModels.entries()).sort(
      (a, b) => a[1].timestamp - b[1].timestamp,
    );

    const toRemove = entries.slice(
      0,
      this.knownModels.size - this.config.modelCacheSize,
    );

    for (const [hash] of toRemove) {
      this.knownModels.delete(hash);
    }
  }

  createSignalingChannel() {
    // Simulated signaling channel for prototype
    return {
      send: (message) => {
        console.log("Signaling send:", message);
      },
      onmessage: null,
    };
  }

  disconnect() {
    // Close all connections
    for (const [peerId, connection] of this.peers.entries()) {
      connection.close();
    }

    this.peers.clear();
    this.dataChannels.clear();

    console.log("Disconnected from all peers");
  }
}

export { PeerNetwork };
