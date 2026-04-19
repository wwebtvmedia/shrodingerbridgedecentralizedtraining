/**
 * Trajectory Advantage Estimator (TAE)
 * RL-inspired adaptive function to identify high-impact layer updates.
 */
export class TrajectoryAdvantageEstimator {
  // ... (keep existing implementation)
  constructor(layerIds, alpha = 0.2) {
    this.alpha = alpha; // Learning rate for the Value estimate
    this.impactMap = new Map();

    // Initialize scores for each layer
    if (layerIds && Array.isArray(layerIds)) {
      layerIds.forEach((id) => {
        this.impactMap.set(id, {
          value: 1.0, // Expected impact
          lastMagnitude: 0, // For credit assignment
        });
      });
    }
  }

  /**
   * Estimates the 'Distance to Expected Result' (Impact)
   * @param {string} layerId
   * @param {number} gradientMagnitude - Norm of the weight update
   * @param {number} deltaLoss - (PrevLoss - CurrentLoss)
   */
  updateImpact(layerId, gradientMagnitude, deltaLoss) {
    if (!this.impactMap.has(layerId)) {
      this.impactMap.set(layerId, { value: 1.0, lastMagnitude: 0 });
    }
    const stats = this.impactMap.get(layerId);

    // Reward is positive if loss decreased AND the layer changed significantly
    const reward = deltaLoss * gradientMagnitude;

    // Update Value estimate using Exponential Moving Average (RL update)
    stats.value = (1 - this.alpha) * stats.value + this.alpha * reward;
    stats.lastMagnitude = gradientMagnitude;
  }

  /**
   * Selects the 'Best' layers to gossip for the next step
   * @param {number} topK - Number of layers to share
   */
  getGossipSelection(topK = 3) {
    return Array.from(this.impactMap.entries())
      .sort((a, b) => b[1].value - a[1].value) // Sort by descending Value
      .slice(0, topK)
      .map((entry) => entry[0]);
  }

  /**
   * Heuristic for 'Distance to Converged Bridge'
   * Returns 0-1, where 1 is 'Needs heavy training'
   */
  estimateDriftDistance() {
    if (this.impactMap.size === 0) return 1.0;
    const totalValue = Array.from(this.impactMap.values()).reduce(
      (acc, s) => acc + Math.abs(s.value),
      0,
    );
    // As training stabilizes, totalValue (Advantage) should decrease
    return Math.tanh(totalValue / this.impactMap.size);
  }
}

/**
 * Evolutionary Optimizer
 * Implements mutation and crossover for model weights (LoRA adapters).
 */
export class EvolutionaryOptimizer {
  constructor(mutationRate = 0.05, mutationScale = 0.01) {
    this.mutationRate = mutationRate;
    this.mutationScale = mutationScale;
  }

  /**
   * Performs mutation on model parameters
   * @param {Object} parameters - Model parameters (vae_params, drift_params)
   */
  mutate(parameters) {
    const mutated = JSON.parse(JSON.stringify(parameters));

    const applyMutation = (arr) => {
      if (!Array.isArray(arr)) return arr;
      return arr.map((val) => {
        if (Array.isArray(val)) return applyMutation(val);
        if (Math.random() < this.mutationRate) {
          return val + (Math.random() * 2 - 1) * this.mutationScale;
        }
        return val;
      });
    };

    if (mutated.vae_params) {
      mutated.vae_params = mutated.vae_params.map((p) => applyMutation(p));
    }
    if (mutated.drift_params) {
      mutated.drift_params = mutated.drift_params.map((p) => applyMutation(p));
    }

    return mutated;
  }

  /**
   * Performs crossover between two sets of model parameters
   * @param {Object} p1 - Parent 1 parameters
   * @param {Object} p2 - Parent 2 parameters
   * @param {number} ratio - Crossover ratio (0.5 for uniform)
   */
  crossover(p1, p2, ratio = 0.5) {
    const child = JSON.parse(JSON.stringify(p1));

    const combine = (arr1, arr2) => {
      if (!Array.isArray(arr1) || !Array.isArray(arr2)) return arr1;
      return arr1.map((val, i) => {
        if (Array.isArray(val)) return combine(val, arr2[i]);
        return Math.random() < ratio ? val : arr2[i];
      });
    };

    if (p1.vae_params && p2.vae_params) {
      child.vae_params = p1.vae_params.map((p, i) =>
        combine(p, p2.vae_params[i]),
      );
    }
    if (p1.drift_params && p2.drift_params) {
      child.drift_params = p1.drift_params.map((p, i) =>
        combine(p, p2.drift_params[i]),
      );
    }

    return child;
  }
}
