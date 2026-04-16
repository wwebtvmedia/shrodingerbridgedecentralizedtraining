class PhaseManager {
  constructor() {
    this.currentPhase = "vae";
    this.phaseHistory = [];
    this.phaseMetrics = {
      vae: { reconstruction: [], kl_divergence: [] },
      drift: { diversity: [], loss: [] },
      both: { combined: [], stability: [] },
    };

    // Phase weights for auto mode
    this.phaseWeights = {
      vae: 0.4,
      drift: 0.3,
      both: 0.3,
    };

    // Phase transition thresholds
    this.thresholds = {
      vaeToDrift: {
        minEpochs: 30,
        maxReconstructionLoss: 0.15,
        maxKLD: 0.05,
      },
      driftToBoth: {
        minEpochs: 50,
        minDiversity: 0.6,
        maxLoss: 0.3,
      },
    };
  }

  determinePhase(epoch) {
    // Simple rule-based phase determination
    if (epoch < 30) {
      return "vae";
    } else if (epoch < 100) {
      return "drift";
    } else {
      return "both";
    }
  }

  determinePhaseAdaptive(epoch, metrics) {
    // Adaptive phase determination based on performance

    // Update metrics for current phase
    this.updatePhaseMetrics(this.currentPhase, metrics);

    // Check for phase transitions
    if (
      this.currentPhase === "vae" &&
      this.shouldTransitionFromVAE(epoch, metrics)
    ) {
      return "drift";
    }

    if (
      this.currentPhase === "drift" &&
      this.shouldTransitionFromDrift(epoch, metrics)
    ) {
      return "both";
    }

    // Stay in current phase
    return this.currentPhase;
  }

  determinePhaseStochastic(epoch) {
    // Stochastic phase selection based on weights
    const rand = Math.random();
    let cumulative = 0;

    for (const [phase, weight] of Object.entries(this.phaseWeights)) {
      cumulative += weight;
      if (rand < cumulative) {
        return phase;
      }
    }

    return "both"; // fallback
  }

  shouldTransitionFromVAE(epoch, metrics) {
    const t = this.thresholds.vaeToDrift;

    return (
      epoch >= t.minEpochs &&
      metrics.reconstruction <= t.maxReconstructionLoss &&
      metrics.kl_divergence <= t.maxKLD
    );
  }

  shouldTransitionFromDrift(epoch, metrics) {
    const t = this.thresholds.driftToBoth;

    return (
      epoch >= t.minEpochs &&
      metrics.diversity >= t.minDiversity &&
      metrics.loss <= t.maxLoss
    );
  }

  updatePhaseMetrics(phase, metrics) {
    if (!this.phaseMetrics[phase]) return;

    for (const [key, value] of Object.entries(metrics)) {
      if (this.phaseMetrics[phase][key] !== undefined) {
        this.phaseMetrics[phase][key].push(value);

        // Keep only recent metrics (last 50)
        if (this.phaseMetrics[phase][key].length > 50) {
          this.phaseMetrics[phase][key].shift();
        }
      }
    }

    // Record phase change
    this.phaseHistory.push({
      epoch: this.getCurrentEpoch(),
      phase,
      timestamp: Date.now(),
    });
  }

  getPhaseParameters(phase) {
    const params = {
      vae: {
        learningRate: 2e-4,
        trainVAE: true,
        trainDrift: false,
        lossWeights: {
          reconstruction: 1.0,
          kl_divergence: 0.01,
          diversity: 0.5,
        },
      },
      drift: {
        learningRate: 4e-4,
        trainVAE: false,
        trainDrift: true,
        lossWeights: {
          drift: 1.0,
          consistency: 0.8,
          diversity: 1.0,
        },
      },
      both: {
        learningRate: 1e-4,
        trainVAE: true,
        trainDrift: true,
        lossWeights: {
          reconstruction: 0.7,
          drift: 0.7,
          consistency: 0.5,
          diversity: 0.8,
        },
      },
    };

    return params[phase] || params.both;
  }

  adjustWeightsBasedOnPerformance(metrics) {
    // Adjust phase weights based on recent performance

    // Calculate performance scores for each phase
    const scores = {
      vae: this.calculateVAEScore(metrics),
      drift: this.calculateDriftScore(metrics),
      both: this.calculateBothScore(metrics),
    };

    // Normalize scores to get new weights
    const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);

    if (totalScore > 0) {
      for (const phase of Object.keys(this.phaseWeights)) {
        this.phaseWeights[phase] = scores[phase] / totalScore;
      }
    }

    // Add some randomness for exploration
    this.addRandomnessToWeights(0.1);
  }

  calculateVAEScore(metrics) {
    // Higher score if reconstruction is poor (loss is high)
    const reconstructionScore = Math.max(0, metrics.reconstruction - 0.1) * 5;

    // Higher score if KL divergence is high
    const klScore = Math.max(0, metrics.kl_divergence - 0.05) * 10;

    return reconstructionScore + klScore + 0.1; // Base score
  }

  calculateDriftScore(metrics) {
    // Higher score if diversity is low
    const diversityScore = Math.max(0, 0.7 - metrics.diversity) * 3;

    // Higher score if loss is high
    const lossScore = Math.max(0, metrics.loss - 0.3) * 2;

    return diversityScore + lossScore + 0.1;
  }

  calculateBothScore(metrics) {
    // Balanced score when both metrics are moderate
    const reconstructionBalance = 1 - Math.abs(metrics.reconstruction - 0.1);
    const diversityBalance = 1 - Math.abs(metrics.diversity - 0.7);

    return reconstructionBalance + diversityBalance + 0.2;
  }

  addRandomnessToWeights(amount) {
    for (const phase of Object.keys(this.phaseWeights)) {
      const randomAdjustment = (Math.random() * 2 - 1) * amount;
      this.phaseWeights[phase] = Math.max(
        0.1,
        Math.min(0.8, this.phaseWeights[phase] + randomAdjustment),
      );
    }

    // Renormalize
    this.normalizeWeights();
  }

  normalizeWeights() {
    const total = Object.values(this.phaseWeights).reduce((a, b) => a + b, 0);

    if (total > 0) {
      for (const phase of Object.keys(this.phaseWeights)) {
        this.phaseWeights[phase] /= total;
      }
    }
  }

  getPhaseStatistics() {
    const stats = {};

    for (const phase of ["vae", "drift", "both"]) {
      const metrics = this.phaseMetrics[phase];
      stats[phase] = {};

      for (const [metric, values] of Object.entries(metrics)) {
        if (values.length > 0) {
          stats[phase][metric] = {
            mean: this.mean(values),
            std: this.std(values),
            trend: this.trend(values),
          };
        }
      }
    }

    return stats;
  }

  getCurrentEpoch() {
    // This should be provided by the trainer
    return 0;
  }

  // Utility methods
  mean(array) {
    return array.reduce((a, b) => a + b, 0) / array.length;
  }

  std(array) {
    const avg = this.mean(array);
    const squareDiffs = array.map((value) => Math.pow(value - avg, 2));
    return Math.sqrt(this.mean(squareDiffs));
  }

  trend(array) {
    if (array.length < 2) return 0;

    const recent = array.slice(-10);
    const first = recent[0];
    const last = recent[recent.length - 1];

    return (last - first) / first;
  }

  reset() {
    this.currentPhase = "vae";
    this.phaseHistory = [];
    this.phaseMetrics = {
      vae: { reconstruction: [], kl_divergence: [] },
      drift: { diversity: [], loss: [] },
      both: { combined: [], stability: [] },
    };

    this.phaseWeights = {
      vae: 0.4,
      drift: 0.3,
      both: 0.3,
    };
  }
}

export { PhaseManager };
