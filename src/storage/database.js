class LocalDatabase {
  constructor(name = "swarm_training") {
    this.dbName = name;
    this.dbVersion = 1;
    this.db = null;
    this.initialized = false;
  }

  async init() {
    if (this.initialized) return;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onerror = (event) => {
        console.error("IndexedDB error:", event.target.error);
        reject(event.target.error);
      };

      request.onsuccess = (event) => {
        this.db = event.target.result;
        this.initialized = true;
        console.log("✅ Local database initialized");
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        // Create object stores
        if (!db.objectStoreNames.contains("neighbors")) {
          const neighborStore = db.createObjectStore("neighbors", {
            keyPath: "peerId",
          });
          neighborStore.createIndex("timestamp", "timestamp", {
            unique: false,
          });
          neighborStore.createIndex("loss", "loss", { unique: false });
        }

        if (!db.objectStoreNames.contains("models")) {
          const modelStore = db.createObjectStore("models", {
            keyPath: "hash",
          });
          modelStore.createIndex("peerId", "peerId", { unique: false });
          modelStore.createIndex("timestamp", "timestamp", { unique: false });
        }

        if (!db.objectStoreNames.contains("results")) {
          const resultStore = db.createObjectStore("results", {
            keyPath: "id",
            autoIncrement: true,
          });
          resultStore.createIndex("epoch", "epoch", { unique: false });
          resultStore.createIndex("phase", "phase", { unique: false });
          resultStore.createIndex("timestamp", "timestamp", { unique: false });
        }

        if (!db.objectStoreNames.contains("checkpoints")) {
          db.createObjectStore("checkpoints", { keyPath: "epoch" });
        }

        if (!db.objectStoreNames.contains("metrics")) {
          const metricsStore = db.createObjectStore("metrics", {
            keyPath: "timestamp",
          });
          metricsStore.createIndex("type", "type", { unique: false });
        }

        if (!db.objectStoreNames.contains("training_data")) {
          const trainingStore = db.createObjectStore("training_data", {
            keyPath: "id",
            autoIncrement: true,
          });
          trainingStore.createIndex("type", "type", { unique: false });
          trainingStore.createIndex("timestamp", "timestamp", {
            unique: false,
          });
          trainingStore.createIndex("format", "format", { unique: false });
        }

        console.log("📊 Database schema created");
      };
    });
  }

  // Neighbor management
  async saveNeighbor(neighborData) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["neighbors"], "readwrite");
      const store = transaction.objectStore("neighbors");

      const request = store.put({
        ...neighborData,
        timestamp: Date.now(),
        updated: Date.now(),
      });

      request.onsuccess = () => resolve();
      request.onerror = (event) => reject(event.target.error);
    });
  }

  async getNeighbor(peerId) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["neighbors"], "readonly");
      const store = transaction.objectStore("neighbors");

      const request = store.get(peerId);

      request.onsuccess = () => resolve(request.result);
      request.onerror = (event) => reject(event.target.error);
    });
  }

  async getAllNeighbors() {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["neighbors"], "readonly");
      const store = transaction.objectStore("neighbors");
      const index = store.index("timestamp");

      const request = index.getAll();

      request.onsuccess = () => resolve(request.result);
      request.onerror = (event) => reject(event.target.error);
    });
  }

  async getBestNeighbors(limit = 5) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["neighbors"], "readonly");
      const store = transaction.objectStore("neighbors");

      const request = store.getAll();

      request.onsuccess = () => {
        const neighbors = request.result;
        // Sort by loss (ascending) and get top N
        const best = neighbors
          .filter((n) => n.loss !== undefined)
          .sort((a, b) => a.loss - b.loss)
          .slice(0, limit);
        resolve(best);
      };

      request.onerror = (event) => reject(event.target.error);
    });
  }

  async deleteNeighbor(peerId) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["neighbors"], "readwrite");
      const store = transaction.objectStore("neighbors");

      const request = store.delete(peerId);

      request.onsuccess = () => resolve();
      request.onerror = (event) => reject(event.target.error);
    });
  }

  async cleanupOldNeighbors(maxAge = 7 * 24 * 60 * 60 * 1000) {
    // 7 days
    await this.ensureInitialized();

    const cutoff = Date.now() - maxAge;
    const neighbors = await this.getAllNeighbors();

    const oldNeighbors = neighbors.filter((n) => n.timestamp < cutoff);

    for (const neighbor of oldNeighbors) {
      await this.deleteNeighbor(neighbor.peerId);
    }

    console.log(`🧹 Cleaned up ${oldNeighbors.length} old neighbors`);
    return oldNeighbors.length;
  }

  // Model storage
  async saveModel(modelData) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["models"], "readwrite");
      const store = transaction.objectStore("models");

      const request = store.put({
        ...modelData,
        timestamp: Date.now(),
      });

      request.onsuccess = () => resolve();
      request.onerror = (event) => reject(event.target.error);
    });
  }

  async getModel(hash) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["models"], "readonly");
      const store = transaction.objectStore("models");

      const request = store.get(hash);

      request.onsuccess = () => resolve(request.result);
      request.onerror = (event) => reject(event.target.error);
    });
  }

  async getModelsByPeer(peerId) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["models"], "readonly");
      const store = transaction.objectStore("models");
      const index = store.index("peerId");

      const request = index.getAll(peerId);

      request.onsuccess = () => resolve(request.result);
      request.onerror = (event) => reject(event.target.error);
    });
  }

  // Training results
  async saveResult(result) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["results"], "readwrite");
      const store = transaction.objectStore("results");

      const request = store.put({
        ...result,
        timestamp: Date.now(),
        id:
          result.id ||
          Date.now() + "_" + Math.random().toString(36).substr(2, 9),
      });

      request.onsuccess = () => resolve();
      request.onerror = (event) => reject(event.target.error);
    });
  }

  async getRecentResults(limit = 100) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["results"], "readonly");
      const store = transaction.objectStore("results");
      const index = store.index("timestamp");

      const request = index.openCursor(null, "prev");
      const results = [];

      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor && results.length < limit) {
          results.push(cursor.value);
          cursor.continue();
        } else {
          resolve(results);
        }
      };

      request.onerror = (event) => reject(event.target.error);
    });
  }

  async getResultsByPhase(phase, limit = 50) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["results"], "readonly");
      const store = transaction.objectStore("results");
      const index = store.index("phase");

      const request = index.getAll(phase);

      request.onsuccess = () => {
        const results = request.result
          .sort((a, b) => b.timestamp - a.timestamp)
          .slice(0, limit);
        resolve(results);
      };

      request.onerror = (event) => reject(event.target.error);
    });
  }

  // Checkpoints
  async saveCheckpoint(checkpoint) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["checkpoints"], "readwrite");
      const store = transaction.objectStore("checkpoints");

      // 1. Save new checkpoint
      const putRequest = store.put({
        ...checkpoint,
        timestamp: Date.now(),
      });

      putRequest.onsuccess = () => {
        // 2. Prune old checkpoints (keep only latest 3)
        const cursorRequest = store.openCursor(null, "prev");
        let count = 0;
        const keepLimit = 3;

        cursorRequest.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            count++;
            if (count > keepLimit) {
              cursor.delete();
            }
            cursor.continue();
          } else {
            resolve();
          }
        };
      };

      putRequest.onerror = (event) => reject(event.target.error);
    });
  }

  async getCheckpoint(epoch) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["checkpoints"], "readonly");
      const store = transaction.objectStore("checkpoints");

      const request = store.get(epoch);

      request.onsuccess = () => resolve(request.result);
      request.onerror = (event) => reject(event.target.error);
    });
  }

  async getLatestCheckpoint() {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["checkpoints"], "readonly");
      const store = transaction.objectStore("checkpoints");

      const request = store.openCursor(null, "prev");

      request.onsuccess = (event) => {
        const cursor = event.target.result;
        resolve(cursor ? cursor.value : null);
      };

      request.onerror = (event) => reject(event.target.error);
    });
  }

  // Metrics
  async saveMetric(metric) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["metrics"], "readwrite");
      const store = transaction.objectStore("metrics");

      const request = store.put({
        ...metric,
        timestamp: Date.now(),
      });

      request.onsuccess = () => resolve();
      request.onerror = (event) => reject(event.target.error);
    });
  }

  async getMetrics(type, startTime, endTime = Date.now()) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["metrics"], "readonly");
      const store = transaction.objectStore("metrics");
      const index = store.index("type");

      const range = IDBKeyRange.bound(startTime, endTime);
      const request = index.getAll(type);

      request.onsuccess = () => {
        const metrics = request.result
          .filter((m) => m.timestamp >= startTime && m.timestamp <= endTime)
          .sort((a, b) => a.timestamp - b.timestamp);
        resolve(metrics);
      };

      request.onerror = (event) => reject(event.target.error);
    });
  }

  // Statistics
  async getStatistics() {
    await this.ensureInitialized();

    const [neighbors, results, models, checkpoints] = await Promise.all([
      this.getAllNeighbors(),
      this.getRecentResults(1000),
      this.getModelsByPeer("all"),
      this.getLatestCheckpoint(),
    ]);

    return {
      neighbors: {
        count: neighbors.length,
        active: neighbors.filter((n) => Date.now() - n.updated < 5 * 60 * 1000)
          .length, // Last 5 minutes
        bestLoss:
          neighbors.length > 0
            ? Math.min(...neighbors.map((n) => n.loss || Infinity))
            : null,
      },
      results: {
        count: results.length,
        byPhase: {
          vae: results.filter((r) => r.phase === "vae").length,
          drift: results.filter((r) => r.phase === "drift").length,
          both: results.filter((r) => r.phase === "both").length,
        },
        averageLoss:
          results.length > 0
            ? results.reduce((sum, r) => sum + (r.loss || 0), 0) /
              results.length
            : null,
      },
      models: {
        count: models.length,
        uniquePeers: new Set(models.map((m) => m.peerId)).size,
      },
      training: {
        latestEpoch: checkpoints?.epoch || 0,
        latestCheckpoint: checkpoints?.timestamp || null,
      },
      database: {
        size: await this.estimateSize(),
        lastCleanup: localStorage.getItem("db_last_cleanup") || "never",
      },
    };
  }

  async estimateSize() {
    // Rough estimation of database size
    const stores = ["neighbors", "models", "results", "checkpoints", "metrics"];
    let total = 0;

    for (const storeName of stores) {
      try {
        const size = await new Promise((resolve, reject) => {
          const transaction = this.db.transaction([storeName], "readonly");
          const store = transaction.objectStore(storeName);
          const request = store.openCursor();
          let storeSize = 0;

          request.onsuccess = (event) => {
            const cursor = event.target.result;
            if (cursor) {
              // Estimate individual item size
              try {
                storeSize += JSON.stringify(cursor.value).length;
              } catch (e) {
                storeSize += 1024; // Fallback for very large/complex objects
              }
              cursor.continue();
            } else {
              resolve(storeSize);
            }
          };
          request.onerror = reject;
        });

        total += size;
      } catch (error) {
        console.warn(`Could not estimate size for ${storeName}:`, error);
      }
    }

    return total;
  }

  async clearDatabase() {
    await this.ensureInitialized();

    const stores = ["neighbors", "models", "results", "checkpoints", "metrics"];

    for (const storeName of stores) {
      await new Promise((resolve, reject) => {
        const transaction = this.db.transaction([storeName], "readwrite");
        const store = transaction.objectStore(storeName);
        const request = store.clear();

        request.onsuccess = () => resolve();
        request.onerror = (event) => reject(event.target.error);
      });
    }

    console.log("🗑️ Database cleared");
    localStorage.setItem("db_last_cleanup", new Date().toISOString());
  }

  // Training data management
  async saveTrainingData(data) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["training_data"], "readwrite");
      const store = transaction.objectStore("training_data");

      const trainingRecord = {
        ...data,
        timestamp: Date.now(),
        id: `td_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      };

      const request = store.put(trainingRecord);

      request.onsuccess = () => resolve(trainingRecord.id);
      request.onerror = (event) => reject(event.target.error);
    });
  }

  async getTrainingData(limit = 100, type = null) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["training_data"], "readonly");
      const store = transaction.objectStore("training_data");
      const index = store.index("timestamp");

      const request = index.getAll();

      request.onsuccess = () => {
        let data = request.result;

        // Filter by type if specified
        if (type) {
          data = data.filter((item) => item.type === type);
        }

        // Sort by timestamp (newest first) and limit
        data.sort((a, b) => b.timestamp - a.timestamp);
        data = data.slice(0, limit);

        resolve(data);
      };

      request.onerror = (event) => reject(event.target.error);
    });
  }

  async getTrainingDataCount(type = null) {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["training_data"], "readonly");
      const store = transaction.objectStore("training_data");

      const request = store.count();

      request.onsuccess = () => {
        // For type filtering, we'd need to get all and filter
        // This is simplified - in production you'd use an index
        if (!type) {
          resolve(request.result);
        } else {
          // Get all and filter
          const getAllRequest = store.getAll();
          getAllRequest.onsuccess = () => {
            const filtered = getAllRequest.result.filter(
              (item) => item.type === type,
            );
            resolve(filtered.length);
          };
          getAllRequest.onerror = (event) => reject(event.target.error);
        }
      };

      request.onerror = (event) => reject(event.target.error);
    });
  }

  async clearTrainingData() {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(["training_data"], "readwrite");
      const store = transaction.objectStore("training_data");

      const request = store.clear();

      request.onsuccess = () => resolve();
      request.onerror = (event) => reject(event.target.error);
    });
  }

  async exportData() {
    const data = {
      neighbors: await this.getAllNeighbors(),
      results: await this.getRecentResults(10000),
      models: await this.getModelsByPeer("all"),
      checkpoints: await this.getLatestCheckpoint(),
      trainingData: await this.getTrainingData(1000),
      timestamp: Date.now(),
      version: "1.1", // Bump version for training data support
    };

    return JSON.stringify(data, null, 2);
  }

  async importData(jsonData) {
    const data = JSON.parse(jsonData);

    if (data.version !== "1.0" && data.version !== "1.1") {
      throw new Error(`Unsupported data version: ${data.version}`);
    }

    // Import neighbors
    for (const neighbor of data.neighbors || []) {
      await this.saveNeighbor(neighbor);
    }

    // Import results
    for (const result of data.results || []) {
      await this.saveResult(result);
    }

    // Import models
    for (const model of data.models || []) {
      await this.saveModel(model);
    }

    // Import checkpoint
    if (data.checkpoints) {
      await this.saveCheckpoint(data.checkpoints);
    }

    // Import training data (version 1.1+)
    if (data.version === "1.1" && data.trainingData) {
      for (const trainingItem of data.trainingData) {
        await this.saveTrainingData(trainingItem);
      }
      console.log(
        `📥 Imported ${data.trainingData.length} training data items`,
      );
    }

    console.log(
      `📥 Imported ${data.neighbors?.length || 0} neighbors, ${data.results?.length || 0} results`,
    );
  }

  // Utility methods
  async ensureInitialized() {
    if (!this.initialized) {
      await this.init();
    }
  }

  async close() {
    if (this.db) {
      this.db.close();
      this.initialized = false;
      console.log("🔒 Database closed");
    }
  }
}

export { LocalDatabase };
