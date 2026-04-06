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
    this.log("UI initialized");
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
              borderColor: "rgb(255, 99, 132)",
              backgroundColor: "rgba(255, 99, 132, 0.1)",
              tension: 0.4,
              fill: true,
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              labels: {
                color: "#e0e0e0",
              },
            },
          },
          scales: {
            x: {
              title: {
                display: true,
                text: "Epoch",
                color: "#e0e0e0",
              },
              grid: {
                color: "rgba(255, 255, 255, 0.1)",
              },
              ticks: {
                color: "#e0e0e0",
              },
            },
            y: {
              title: {
                display: true,
                text: "Loss",
                color: "#e0e0e0",
              },
              grid: {
                color: "rgba(255, 255, 255, 0.1)",
              },
              ticks: {
                color: "#e0e0e0",
              },
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
              borderColor: "rgb(54, 162, 235)",
              backgroundColor: "rgba(54, 162, 235, 0.1)",
              tension: 0.4,
              fill: true,
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              labels: {
                color: "#e0e0e0",
              },
            },
          },
          scales: {
            x: {
              title: {
                display: true,
                text: "Epoch",
                color: "#e0e0e0",
              },
              grid: {
                color: "rgba(255, 255, 255, 0.1)",
              },
              ticks: {
                color: "#e0e0e0",
              },
            },
            y: {
              title: {
                display: true,
                text: "Diversity",
                color: "#e0e0e0",
              },
              min: 0,
              max: 1,
              grid: {
                color: "rgba(255, 255, 255, 0.1)",
              },
              ticks: {
                color: "#e0e0e0",
              },
            },
          },
        },
      });
    }
  }

  updateStatus(status) {
    const element = document.getElementById("training-phase");
    if (element) {
      element.textContent = status;

      // Add color coding
      element.className = "value";
      if (status.includes("Connected") || status.includes("Training")) {
        element.classList.add("status-good");
      } else if (status.includes("Failed") || status.includes("Error")) {
        element.classList.add("status-bad");
      }
    }
  }

  updatePeerCount(count) {
    const element = document.getElementById("peer-count");
    if (element) {
      element.textContent = count;

      // Update peer list
      this.updatePeerList(count);
    }
  }

  updatePeerList(count) {
    const list = document.getElementById("peer-list");
    if (!list) return;

    list.innerHTML = "";

    // Add simulated peers
    for (let i = 0; i < count; i++) {
      const li = document.createElement("li");
      li.className = "peer-join";
      li.innerHTML = `
                <span>Peer ${i + 1}</span>
                <span class="peer-status">🟢</span>
            `;
      list.appendChild(li);
    }

    // Remove animation class after animation completes
    setTimeout(() => {
      const items = list.querySelectorAll(".peer-join");
      items.forEach((item) => item.classList.remove("peer-join"));
    }, 1000);
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
        element.textContent = loss.toFixed(4);
      } else {
        element.textContent = loss;
      }
    }
  }

  updatePhase(phase) {
    const element = document.getElementById("training-phase");
    if (element) {
      element.textContent = phase;

      // Update phase buttons
      document.querySelectorAll(".phase-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.phase === phase);
      });
    }
  }

  updateMetrics(metrics) {
    // Update models evaluated
    const modelsElement = document.getElementById("models-evaluated");
    if (modelsElement && metrics.modelsEvaluated !== undefined) {
      modelsElement.textContent = metrics.modelsEvaluated;
    }

    // Update sync count
    const syncElement = document.getElementById("sync-count");
    if (syncElement && metrics.syncCount !== undefined) {
      syncElement.textContent = metrics.syncCount;
    }
  }

  updateLossChart(epoch, loss) {
    if (!this.lossChart) return;

    this.lossData.push({ x: epoch, y: loss });

    // Keep only last 100 points
    if (this.lossData.length > 100) {
      this.lossData.shift();
    }

    this.lossChart.data.labels = this.lossData.map((d) => d.x);
    this.lossChart.data.datasets[0].data = this.lossData.map((d) => d.y);
    this.lossChart.update("none");
  }

  updateDiversityChart(epoch, diversity) {
    if (!this.diversityChart) return;

    this.diversityData.push({ x: epoch, y: diversity });

    // Keep only last 100 points
    if (this.diversityData.length > 100) {
      this.diversityData.shift();
    }

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

    samples.forEach((sample, index) => {
      const img = document.createElement("img");
      img.src = sample;
      img.alt = `Sample ${index + 1}`;
      img.title = `Generated sample`;
      grid.appendChild(img);
    });
  }

  log(message) {
    const logElement = document.getElementById("training-log");
    if (!logElement) return;

    const timestamp = new Date().toLocaleTimeString();
    const logEntry = `[${timestamp}] ${message}\n`;

    logElement.textContent += logEntry;

    // Auto-scroll to bottom
    logElement.scrollTop = logElement.scrollHeight;
  }

  clearLog() {
    const logElement = document.getElementById("training-log");
    if (logElement) {
      logElement.textContent = "";
    }
  }

  enableButton(buttonId) {
    const button = document.getElementById(buttonId);
    if (button) {
      button.disabled = false;
    }
  }

  disableButton(buttonId) {
    const button = document.getElementById(buttonId);
    if (button) {
      button.disabled = true;
    }
  }

  incrementSyncCount() {
    const element = document.getElementById("sync-count");
    if (element) {
      const current = parseInt(element.textContent) || 0;
      element.textContent = current + 1;
    }
  }

  incrementModelsEvaluated() {
    const element = document.getElementById("models-evaluated");
    if (element) {
      const current = parseInt(element.textContent) || 0;
      element.textContent = current + 1;
    }
  }

  showNotification(message, type = "info") {
    // Create notification element
    const notification = document.createElement("div");
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    // Style
    notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;

    // Type-specific colors
    const colors = {
      info: "linear-gradient(90deg, #2196f3, #21cbf3)",
      success: "linear-gradient(90deg, #00c853, #64dd17)",
      warning: "linear-gradient(90deg, #ff9800, #ff5722)",
      error: "linear-gradient(90deg, #f44336, #ff5252)",
    };

    notification.style.background = colors[type] || colors.info;

    // Add to document
    document.body.appendChild(notification);

    // Remove after 3 seconds
    setTimeout(() => {
      notification.style.animation = "slideOut 0.3s ease";
      setTimeout(() => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification);
        }
      }, 300);
    }, 3000);

    // Add CSS animations if not already present
    if (!document.querySelector("#notification-styles")) {
      const style = document.createElement("style");
      style.id = "notification-styles";
      style.textContent = `
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes slideOut {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(100%); opacity: 0; }
                }
            `;
      document.head.appendChild(style);
    }
  }

  updateExplorationRate(rate) {
    const slider = document.getElementById("exploration-slider");
    const value = document.getElementById("exploration-value");

    if (slider) {
      slider.value = Math.round(rate * 100);
    }

    if (value) {
      value.textContent = `${Math.round(rate * 100)}%`;
    }
  }

  showLoading(show = true) {
    const buttons = document.querySelectorAll(".btn");

    buttons.forEach((button) => {
      if (show) {
        button.disabled = true;
        const originalText = button.textContent;
        button.dataset.originalText = originalText;
        button.innerHTML = `<span class="loading"></span> ${originalText}`;
      } else {
        button.disabled = false;
        if (button.dataset.originalText) {
          button.textContent = button.dataset.originalText;
        }
      }
    });
  }
}

export { UIManager };
