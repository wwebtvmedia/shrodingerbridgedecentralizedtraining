class ConsolidationClient {
  constructor() {
    this.ws = null;
    this.clientId = null;
    this.isConnected = false;
    this.isRegistered = false;
    this.isTraining = false;
    this.trainingInterval = null;

    // Local training state
    this.localLoss = 1.0;
    this.localEpoch = 0;
    this.localPhase = "vae";
    this.modelsSubmitted = 0;

    // Server state
    this.bestModel = null;
    this.connectedClients = [];
    this.modelUpdates = 0;

    // Chart
    this.chart = null;
    this.trainingData = {
      epochs: [],
      losses: [],
    };

    // Initialize
    this.init();
  }

  init() {
    console.log("🚀 Initializing Consolidation Client...");

    // Generate client ID
    this.clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Setup event listeners
    this.setupEventListeners();

    // Initialize chart
    this.initChart();

    // Update UI
    this.updateUI();

    this.log("Client initialized", "info");
  }

  setupEventListeners() {
    // Connection buttons (null-safe: a missing element shouldn't throw and
    // abort the rest of the bindings).
    this.bindClick("connect-btn", () => this.connect());
    this.bindClick("disconnect-btn", () => this.disconnect());
    this.bindClick("register-btn", () => this.registerAsTrainer());

    // Training buttons
    this.bindClick("start-training-btn", () => this.startTraining());
    this.bindClick("stop-training-btn", () => this.stopTraining());
    this.bindClick("submit-model-btn", () => this.submitModel());

    // Other buttons
    this.bindClick("refresh-clients-btn", () => this.refreshClients());
    this.bindClick("broadcast-test-btn", () => this.testBroadcast());
    this.bindClick("clear-log-btn", () => this.clearLog());
    this.bindClick("download-model-btn", () => this.downloadBestModel());
  }

  initChart() {
    const ctx = document.getElementById("training-chart").getContext("2d");
    this.chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: this.trainingData.epochs,
        datasets: [
          {
            label: "Training Loss",
            data: this.trainingData.losses,
            borderColor: "#4a6fa5",
            backgroundColor: "rgba(74, 111, 165, 0.1)",
            borderWidth: 2,
            fill: true,
            tension: 0.4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: "top",
          },
          tooltip: {
            mode: "index",
            intersect: false,
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: "Epoch",
            },
            grid: {
              display: true,
              color: "rgba(0,0,0,0.05)",
            },
          },
          y: {
            title: {
              display: true,
              text: "Loss",
            },
            grid: {
              display: true,
              color: "rgba(0,0,0,0.05)",
            },
            beginAtZero: false,
          },
        },
      },
    });
  }

  async connect() {
    if (this.isConnected) {
      this.log("Already connected to server", "warning");
      return;
    }

    const serverUrl = "ws://localhost:3001";
    this.log(`Connecting to server at ${serverUrl}...`, "info");

    try {
      this.manualClose = false;
      this.ws = new WebSocket(serverUrl);

      this.ws.onopen = () => {
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.log("Connected to consolidation server", "success");
        this.updateConnectionStatus();
        this.updateButtonStates();

        // Start heartbeat
        this.startHeartbeat();
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };

      this.ws.onclose = () => {
        this.isConnected = false;
        this.isRegistered = false;
        this.log("Disconnected from server", "warning");
        this.updateConnectionStatus();
        this.updateButtonStates();

        // Stop heartbeat
        this.stopHeartbeat();

        // Auto-reconnect with bounded backoff unless the user disconnected.
        if (!this.manualClose) this.scheduleReconnect();
      };

      this.ws.onerror = (event) => {
        // WebSocket error events have no `message`; log the event type instead.
        this.log(`WebSocket error: ${event?.type || "error"}`, "error");
      };
    } catch (error) {
      this.log(`Connection failed: ${error.message}`, "error");
    }
  }

  disconnect() {
    this.manualClose = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
    this.isRegistered = false;
    this.log("Disconnected from server", "info");
    this.updateConnectionStatus();
    this.updateButtonStates();
  }

  scheduleReconnect() {
    const maxAttempts = 5;
    this.reconnectAttempts = (this.reconnectAttempts || 0) + 1;
    if (this.reconnectAttempts > maxAttempts) {
      this.log(
        "Reconnect attempts exhausted; click Connect to retry.",
        "error",
      );
      return;
    }
    const delay = Math.min(1000 * 2 ** (this.reconnectAttempts - 1), 30000);
    this.log(
      `Reconnecting in ${delay / 1000}s (attempt ${this.reconnectAttempts}/${maxAttempts})...`,
      "info",
    );
    this.reconnectTimer = setTimeout(() => {
      if (!this.manualClose) this.connect();
    }, delay);
  }

  registerAsTrainer() {
    if (!this.isConnected || !this.ws) {
      this.log("Not connected to server", "error");
      return;
    }

    const trainerData = {
      type: "register_training",
      data: {
        name: `Trainer_${this.clientId.substr(0, 8)}`,
        clientId: this.clientId,
        capabilities: ["vae", "drift", "both"],
        startedAt: new Date().toISOString(),
      },
    };

    this.ws.send(JSON.stringify(trainerData));
    this.isRegistered = true;
    this.log("Registered as training client", "success");
    this.updateButtonStates();
  }

  handleMessage(data) {
    try {
      const message = JSON.parse(data);

      switch (message.type) {
        case "model_update":
          this.handleModelUpdate(message.model);
          break;

        case "new_best_model":
          this.handleNewBestModel(message.model);
          break;

        case "heartbeat_ack":
          // Heartbeat acknowledged
          break;

        case "model_accepted_as_best":
          this.log(
            `Our model accepted as new best! Loss: ${message.loss}`,
            "success",
          );
          break;

        default:
          this.log(`Received message: ${message.type}`, "info");
      }
    } catch (error) {
      this.log(`Error parsing message: ${error.message}`, "error");
    }
  }

  handleModelUpdate(model) {
    this.bestModel = model;
    this.modelUpdates++;
    this.updateModelInfo();
    this.log(
      `Model update received: loss=${model.loss}, client=${model.clientId}`,
      "info",
    );
  }

  handleNewBestModel(model) {
    this.bestModel = model;
    this.modelUpdates++;
    this.updateModelInfo();
    this.log(
      `🎉 NEW BEST MODEL! Loss: ${model.loss} from client ${model.clientId}`,
      "success",
    );

    // Update server status
    this.setText("best-model-loss", this.fmtNum(model.loss));
    this.setText("model-updates", this.modelUpdates);
  }

  startHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    this.heartbeatInterval = setInterval(() => {
      if (
        this.isConnected &&
        this.ws &&
        this.ws.readyState === WebSocket.OPEN
      ) {
        const heartbeat = {
          type: "heartbeat",
          clientId: this.clientId,
          timestamp: Date.now(),
          metrics: {
            loss: this.localLoss,
            epoch: this.localEpoch,
            phase: this.localPhase,
          },
        };

        this.ws.send(JSON.stringify(heartbeat));
      }
    }, 10000); // Every 10 seconds
  }

  stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  startTraining() {
    if (this.isTraining) return;

    this.isTraining = true;
    this.log("Starting training simulation...", "info");
    this.updateButtonStates();

    // Reset training state
    this.localLoss = 1.0;
    this.localEpoch = 0;
    this.trainingData = { epochs: [], losses: [] };

    // Start training loop
    this.trainingInterval = setInterval(() => {
      this.trainingStep();
    }, 1000); // One epoch per second

    this.log("Training simulation started", "success");
  }

  stopTraining() {
    if (!this.isTraining) return;

    this.isTraining = false;
    if (this.trainingInterval) {
      clearInterval(this.trainingInterval);
      this.trainingInterval = null;
    }

    this.log("Training simulation stopped", "info");
    this.updateButtonStates();
  }

  trainingStep() {
    // Simulate training progress
    this.localEpoch++;

    // Simulate loss improvement with some randomness
    const improvement = 0.01 + Math.random() * 0.05;
    this.localLoss = Math.max(0.1, this.localLoss - improvement);

    // Cycle through phases
    const phases = ["vae", "drift", "both"];
    this.localPhase = phases[this.localEpoch % 3];

    // Update training data
    this.trainingData.epochs.push(this.localEpoch);
    this.trainingData.losses.push(this.localLoss);

    // Keep only last 50 points
    if (this.trainingData.epochs.length > 50) {
      this.trainingData.epochs.shift();
      this.trainingData.losses.shift();
    }

    // Update chart
    this.updateChart();

    // Update UI
    this.updateLocalTrainingStatus();

    // Randomly submit model (10% chance per epoch)
    if (Math.random() < 0.1 && this.isConnected && this.isRegistered) {
      this.submitModel();
    }
  }

  async submitModel() {
    if (!this.isConnected || !this.ws || !this.isRegistered) {
      this.log("Cannot submit model: not connected or registered", "warning");
      return;
    }

    // Simulate model data (in real implementation, this would be actual model weights)
    const modelData = {
      type: "model_update",
      clientId: this.clientId,
      modelData: this.generateMockModelData(),
      loss: this.localLoss,
      epoch: this.localEpoch,
      metrics: {
        phase: this.localPhase,
        diversity: 0.5 + Math.random() * 0.3,
        klDivergence: 0.1 + Math.random() * 0.2,
        reconstruction: 0.7 + Math.random() * 0.2,
      },
    };

    this.ws.send(JSON.stringify(modelData));
    this.modelsSubmitted++;

    this.log(
      `Model submitted: loss=${this.localLoss.toFixed(4)}, epoch=${this.localEpoch}`,
      "success",
    );
    this.updateLocalTrainingStatus();
  }

  generateMockModelData() {
    // Generate mock model data (base64 encoded random bytes)
    // In a real implementation, this would be actual model weights
    const size = 1024 * 100; // 100KB mock data
    const buffer = new Uint8Array(size);
    for (let i = 0; i < size; i++) {
      buffer[i] = Math.floor(Math.random() * 256);
    }

    // Convert to base64 in chunks. String.fromCharCode.apply(null, bigArray)
    // throws RangeError (max call stack / arg count) for ~100KB inputs.
    let binary = "";
    const CHUNK = 0x8000;
    for (let i = 0; i < buffer.length; i += CHUNK) {
      binary += String.fromCharCode.apply(null, buffer.subarray(i, i + CHUNK));
    }
    return btoa(binary);
  }

  async refreshClients() {
    if (!this.isConnected) {
      this.log("Not connected to server", "error");
      return;
    }

    try {
      const response = await fetch("http://localhost:3001/api/clients");
      if (response.ok) {
        const data = await response.json();
        this.connectedClients = data.clients;
        this.updateClientsList();
        this.log(
          `Refreshed clients list: ${data.clients.length} clients`,
          "info",
        );

        // Update connected clients count
        document.getElementById("connected-clients").textContent =
          data.clients.length;
      }
    } catch (error) {
      this.log(`Failed to refresh clients: ${error.message}`, "error");
    }
  }

  testBroadcast() {
    if (!this.isConnected || !this.ws) {
      this.log("Not connected to server", "error");
      return;
    }

    // Send a test message that should trigger a broadcast
    const testMessage = {
      type: "test_broadcast",
      clientId: this.clientId,
      message: "Test broadcast from client",
      timestamp: Date.now(),
    };

    this.ws.send(JSON.stringify(testMessage));
    this.log("Test broadcast sent to server", "info");
  }

  async downloadBestModel() {
    try {
      const response = await fetch("http://localhost:3001/api/model/download");
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "latest.pt";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        this.log("Best model downloaded successfully", "success");
      } else {
        this.log("Failed to download model: No model available", "error");
      }
    } catch (error) {
      this.log(`Download failed: ${error.message}`, "error");
    }
  }

  updateConnectionStatus() {
    const statusElement = document.getElementById("connection-status");
    const statusText = statusElement.querySelector("span:last-child");

    if (this.isConnected) {
      statusElement.className = "connection-status connected";
      statusText.textContent = "Connected";
      document.getElementById("server-status").textContent = "Online";
    } else {
      statusElement.className = "connection-status disconnected";
      statusText.textContent = "Disconnected";
      document.getElementById("server-status").textContent = "Offline";
    }
  }

  updateButtonStates() {
    const connectBtn = document.getElementById("connect-btn");
    const disconnectBtn = document.getElementById("disconnect-btn");
    const registerBtn = document.getElementById("register-btn");
    const startTrainingBtn = document.getElementById("start-training-btn");
    const stopTrainingBtn = document.getElementById("stop-training-btn");
    const submitModelBtn = document.getElementById("submit-model-btn");

    // Connection buttons
    connectBtn.disabled = this.isConnected;
    disconnectBtn.disabled = !this.isConnected;
    registerBtn.disabled = !this.isConnected || this.isRegistered;

    // Training buttons
    startTrainingBtn.disabled = !this.isRegistered;
    stopTrainingBtn.disabled = !this.isTraining;
    submitModelBtn.disabled = !this.isRegistered;
  }

  updateModelInfo() {
    const modelInfoElement = document.getElementById("current-model-info");
    const modelStatsElement = document.getElementById("model-stats");

    if (!this.bestModel) {
      modelInfoElement.innerHTML =
        "<p>No model available. Waiting for client submissions.</p>";
      modelStatsElement.innerHTML = "";
      return;
    }

    const model = this.bestModel;
    const timestamp = new Date(model.timestamp).toLocaleString();
    // All server-supplied fields are escaped/validated before interpolation —
    // a malicious/compromised server must not be able to inject markup here.
    const clientId = this.escapeHtml(model.clientId);
    const clientShort = this.escapeHtml(
      String(model.clientId ?? "").substring(0, 8),
    );
    const sizeMB = Number.isFinite(model.size)
      ? (model.size / 1024 / 1024).toFixed(2)
      : "—";
    const epoch = Number.isFinite(model.epoch) ? model.epoch : "—";

    modelInfoElement.innerHTML = `
            <p><strong>Best Model</strong> from client <code>${clientId}</code></p>
            <p>Last updated: ${this.escapeHtml(timestamp)}</p>
        `;

    modelStatsElement.innerHTML = `
            <div class="stat-item">
                <div class="label">Loss</div>
                <div class="value">${this.fmtNum(model.loss)}</div>
            </div>
            <div class="stat-item">
                <div class="label">Epoch</div>
                <div class="value">${epoch}</div>
            </div>
            <div class="stat-item">
                <div class="label">Client</div>
                <div class="value">${clientShort}...</div>
            </div>
            <div class="stat-item">
                <div class="label">Size</div>
                <div class="value">${sizeMB} MB</div>
            </div>
        `;

    // Update server status
    this.setText("best-model-loss", this.fmtNum(model.loss));
    this.setText("model-updates", this.modelUpdates);
  }

  updateLocalTrainingStatus() {
    document.getElementById("local-loss").textContent =
      this.localLoss.toFixed(4);
    document.getElementById("local-epoch").textContent = this.localEpoch;
    document.getElementById("local-phase").textContent =
      this.localPhase.toUpperCase();
    document.getElementById("models-submitted").textContent =
      this.modelsSubmitted;
  }

  updateChart() {
    if (this.chart) {
      this.chart.data.labels = this.trainingData.epochs;
      this.chart.data.datasets[0].data = this.trainingData.losses;
      this.chart.update("none");
    }
  }

  updateClientsList() {
    const clientsListElement = document.getElementById("clients-list");

    if (this.connectedClients.length === 0) {
      clientsListElement.innerHTML = `
                <div class="client-item">
                    <div class="client-info">
                        <div class="client-name">No clients connected</div>
                        <div class="client-metrics">Waiting for connections...</div>
                    </div>
                </div>
            `;
      return;
    }

    clientsListElement.innerHTML = "";

    this.connectedClients.forEach((client) => {
      const lastSeen = client.lastSeen ? new Date(client.lastSeen) : null;
      const isActive = lastSeen && Date.now() - lastSeen.getTime() < 30000; // Active if seen in last 30 seconds

      const clientElement = document.createElement("div");
      clientElement.className = `client-item ${isActive ? "" : "inactive"}`;

      const lastSeenText = lastSeen
        ? `${Math.floor((Date.now() - lastSeen.getTime()) / 1000)} seconds ago`
        : "Never";

      // Escape peer-supplied fields (name/capabilities/id) before rendering.
      const name = this.escapeHtml(client.name || client.id);
      const caps = Array.isArray(client.capabilities)
        ? this.escapeHtml(client.capabilities.join(", "))
        : "No capabilities";

      clientElement.innerHTML = `
                <div class="client-info">
                    <div class="client-name">${name}</div>
                    <div class="client-metrics">
                        ${caps} |
                        Last seen: ${lastSeenText}
                    </div>
                </div>
                <div class="client-status">
                    ${isActive ? "🟢 Active" : "⚫ Inactive"}
                </div>
            `;

      clientsListElement.appendChild(clientElement);
    });
  }

  updateUI() {
    this.updateConnectionStatus();
    this.updateButtonStates();
    this.updateModelInfo();
    this.updateLocalTrainingStatus();
    this.updateChart();
  }

  // --- helpers ---
  // Escape untrusted strings before interpolating into innerHTML (XSS guard).
  escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Format a value as a fixed-decimal number, tolerating missing/non-numeric.
  fmtNum(value, digits = 4) {
    return Number.isFinite(value) ? value.toFixed(digits) : "—";
  }

  // Null-safe element lookup + click binding.
  bindClick(id, handler) {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", handler);
  }

  setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  log(message, type = "info") {
    const logContainer = document.getElementById("log-container");
    if (!logContainer) return;
    const logEntry = document.createElement("div");
    logEntry.className = `log-entry ${type}`;

    const timestamp = new Date().toLocaleTimeString();
    logEntry.textContent = `[${timestamp}] ${message}`;

    logContainer.appendChild(logEntry);

    // Auto-scroll to bottom
    logContainer.scrollTop = logContainer.scrollHeight;

    // Also log to console
    console.log(`[${type.toUpperCase()}] ${message}`);
  }

  clearLog() {
    const logContainer = document.getElementById("log-container");
    logContainer.innerHTML =
      '<div class="log-entry info">[System] Log cleared</div>';
  }
}

// Initialize the client when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  window.consolidationClient = new ConsolidationClient();
});
