/**
 * Trajectory Advantage Estimator (TAE)
 * RL-inspired adaptive function to identify high-impact layer updates.
 */
export class TrajectoryAdvantageEstimator {
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
