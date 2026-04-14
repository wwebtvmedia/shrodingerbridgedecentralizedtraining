import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

// Register Chart.js components
Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend,
  Filler,
);

class UIManager {
  constructor() {
    this.lossChart = null;
    this.diversityChart = null;
    this.lossData = [];
    this.diversityData = [];

    this.initCharts();
  }

  init() {
    console.log("🎨 Initializing UI manager...");
    this.updateStatus("Initializing...");
    this.updatePeerCount(0);
    this.updateEpoch(0);
    this.updateLoss("-");
    this.updatePhase("initializing");

    // Clear log
    this.clearLog();
    this.log("System initialized and ready.");
  }

  initCharts() {
    if (typeof Chart === "undefined") {
      console.warn("⚠️ Chart.js not loaded. Charts will not be available.");
      return;
    }

    // Loss chart
    const lossCtx = document.getElementById("loss-chart")?.getContext("2d");
    if (lossCtx) {
      this.lossChart = new Chart(lossCtx, {
        type: "line",
        data: {
          labels: [],
          datasets: [
            {
              label: "Training Loss",
              data: [],
              borderColor: "#ea4335", // Google Red
              backgroundColor: "rgba(234, 67, 53, 0.1)",
              tension: 0.2,
              fill: true,
              pointRadius: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false,
            },
          },
          scales: {
            x: {
              display: false,
              grid: { display: false },
            },
            y: {
              beginAtZero: true,
              grid: { color: "rgba(255, 255, 255, 0.05)" },
              ticks: { color: "#8d9199", font: { size: 10 } },
            },
          },
        },
      });
    }

    // Diversity chart
    const diversityCtx = document
      .getElementById("diversity-chart")
      ?.getContext("2d");
    if (diversityCtx) {
      this.diversityChart = new Chart(diversityCtx, {
        type: "line",
        data: {
          labels: [],
          datasets: [
            {
              label: "Diversity",
              data: [],
              borderColor: "#4285f4", // Google Blue
              backgroundColor: "rgba(66, 133, 244, 0.1)",
              tension: 0.2,
              fill: true,
              pointRadius: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
          },
          scales: {
            x: { display: false, grid: { display: false } },
            y: {
              min: 0,
              max: 1,
              grid: { color: "rgba(255, 255, 255, 0.05)" },
              ticks: { color: "#8d9199", font: { size: 10 } },
            },
          },
        },
      });
    }
  }

  updateStatus(status) {
    const textElement = document.getElementById("tunnel-status-text");
    const indicator = document.querySelector(".status-indicator");
    if (textElement) {
      const isConnected =
        status.toLowerCase().includes("connected") ||
        status.toLowerCase().includes("training");
      textElement.innerHTML = `<span class="status-indicator ${isConnected ? "connected" : "disconnected"}"></span>${status}`;
    }
  }

  updatePeerCount(count) {
    const element = document.getElementById("peer-count");
    if (element) {
      element.textContent = count;
    }
  }

  updatePeerList(count) {
    // Legacy method, not used in new M3 layout but kept for compatibility
  }

  updateEpoch(epoch) {
    const element = document.getElementById("current-epoch");
    if (element) {
      element.textContent = epoch;
    }
  }

  updateLoss(loss) {
    const element = document.getElementById("best-loss");
    if (element) {
      if (typeof loss === "number") {
        element.textContent = loss.toFixed(6);
      } else {
        element.textContent = loss;
      }
    }
  }

  updatePhase(phase) {
    // Update labels if needed
    document.querySelectorAll(".phase-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.phase === phase);
    });
  }

  updateHardwareInfo(state, device) {
    const element = document.getElementById("hardware-state");
    if (element) {
      const fullStatus = device
        ? `${device.toUpperCase()}`
        : state.toUpperCase();
      element.textContent = fullStatus;

      if (
        state.toLowerCase().includes("error") ||
        state.toLowerCase().includes("failed")
      ) {
        element.style.color = "var(--google-red)";
      } else if (
        device &&
        (device.toLowerCase().includes("gpu") ||
          device.toLowerCase().includes("webgpu") ||
          device.toLowerCase().includes("webgl"))
      ) {
        element.style.color = "var(--google-green)";
      } else {
        element.style.color = "var(--google-blue)";
      }
    }
  }

  updateMetrics(metrics) {
    // Update local database UI if present
    if (metrics.dbStats) {
      const ids = {
        neighbors: "db-neighbors",
        results: "db-results",
        models: "db-models",
        checkpoints: "db-checkpoints",
      };
      for (const [key, id] of Object.entries(ids)) {
        const el = document.getElementById(id);
        if (el && metrics.dbStats[key])
          el.textContent = metrics.dbStats[key].count || 0;
      }
      const sizeEl = document.getElementById("db-size");
      if (sizeEl && metrics.dbStats.database) {
        sizeEl.textContent = `${Math.round(metrics.dbStats.database.size / 1024)} KB`;
      }
    }
  }

  updateLossChart(epoch, loss) {
    if (!this.lossChart) return;
    this.lossData.push({ x: epoch, y: loss });
    if (this.lossData.length > 50) this.lossData.shift();
    this.lossChart.data.labels = this.lossData.map((d) => d.x);
    this.lossChart.data.datasets[0].data = this.lossData.map((d) => d.y);
    this.lossChart.update("none");
  }

  updateDiversityChart(epoch, diversity) {
    if (!this.diversityChart) return;
    this.diversityData.push({ x: epoch, y: diversity });
    if (this.diversityData.length > 50) this.diversityData.shift();
    this.diversityChart.data.labels = this.diversityData.map((d) => d.x);
    this.diversityChart.data.datasets[0].data = this.diversityData.map(
      (d) => d.y,
    );
    this.diversityChart.update("none");
  }

  displaySamples(samples) {
    const grid = document.getElementById("sample-grid");
    if (!grid) return;
    grid.innerHTML = "";
    samples.forEach((sample) => {
      const img = document.createElement("img");
      img.src = sample;
      grid.appendChild(img);
    });
  }

  log(message) {
    const logElement = document.getElementById("training-log");
    if (!logElement) return;
    const timestamp = new Date().toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    const logEntry = `[${timestamp}] ${message}\n`;
    logElement.textContent += logEntry;
    logElement.scrollTop = logElement.scrollHeight;
  }

  clearLog() {
    const logElement = document.getElementById("training-log");
    if (logElement) logElement.textContent = "";
  }

  enableButton(id) {
    const el = document.getElementById(id);
    if (el) el.disabled = false;
  }
  disableButton(id) {
    const el = document.getElementById(id);
    if (el) el.disabled = true;
  }

  showNotification(message, type = "info") {
    const notification = document.createElement("div");
    notification.className = `notification`;
    notification.textContent = message;
    const colors = {
      info: "#4285f4",
      success: "#34a853",
      warning: "#fbbc05",
      error: "#ea4335",
    };
    notification.style.cssText = `
      position: fixed; bottom: 24px; left: 24px; padding: 14px 24px; border-radius: 8px;
      color: white; font-size: 14px; font-family: 'Google Sans', sans-serif;
      z-index: 10000; background: ${colors[type] || colors.info};
      box-shadow: 0 4px 12px rgba(0,0,0,0.4); animation: slideUp 0.3s ease;
    `;
    document.body.appendChild(notification);
    setTimeout(() => {
      notification.style.animation = "fadeOut 0.3s ease";
      setTimeout(() => notification.remove(), 300);
    }, 4000);
  }

  showLoading(show = true) {
    const btns = document.querySelectorAll(".btn");
    btns.forEach((btn) => {
      if (show) {
        btn.disabled = true;
        btn.dataset.oldText = btn.innerHTML;
        btn.innerHTML =
          '<span class="material-icons-outlined spin" style="font-size: 18px;">sync</span>';
      } else if (btn.dataset.oldText) {
        btn.disabled = false;
        btn.innerHTML = btn.dataset.oldText;
      }
    });
  }
}

export { UIManager };
