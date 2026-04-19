/**
 * SARSA-based Bridge Optimizer
 * Learns the best training/gossip strategy for Schrödinger Bridge.
 */
export class SarsaBridgeOptimizer {
  constructor(alpha = 0.1, gamma = 0.9, epsilon = 0.1) {
    this.alpha = alpha; // Learning rate
    this.gamma = gamma; // Discount factor
    this.epsilon = epsilon; // Exploration rate

    // Actions: 0: VAE, 1: Drift, 2: Both
    this.actions = [0, 1, 2];
    this.actionNames = ["vae", "drift", "both"];

    // Q-Table: Map<StateKey, Array[3]>
    // StateKey: "phase_lossBin" (e.g., "vae_high")
    this.qTable = new Map();

    // Tracking for SARSA update: S, A, R
    this.prevState = null;
    this.prevAction = null;
  }

  /**
   * Discretizes the current system state
   */
  getState(phase, currentLoss) {
    let lossBin = "low";
    if (currentLoss > 0.5) lossBin = "high";
    else if (currentLoss > 0.1) lossBin = "med";

    return `${phase}_${lossBin}`;
  }

  /**
   * Epsilon-greedy action selection
   */
  chooseAction(state) {
    if (!this.qTable.has(state)) {
      this.qTable.set(state, [0.1, 0.1, 0.1]); // Optimistic initialization
    }

    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actions.length);
    }

    const qValues = this.qTable.get(state);
    let maxQ = -Infinity;
    let bestActions = [];

    for (let i = 0; i < qValues.length; i++) {
      if (qValues[i] > maxQ) {
        maxQ = qValues[i];
        bestActions = [i];
      } else if (qValues[i] === maxQ) {
        bestActions.push(i);
      }
    }

    // Tie-breaking
    return bestActions[Math.floor(Math.random() * bestActions.length)];
  }

  /**
   * The SARSA Update Step
   * Q(S,A) = Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S,A)]
   */
  update(currentPhase, loss, deltaLoss, timeMs) {
    const currentState = this.getState(currentPhase, loss);
    const nextAction = this.chooseAction(currentState);

    // Calculate Reward: Improvement per millisecond
    // High reward if deltaLoss is positive (improvement) and time is low
    // We normalize to make it more stable
    const reward = timeMs > 0 ? (deltaLoss * 1000) / timeMs : 0;

    if (this.prevState !== null && this.prevAction !== null) {
      if (!this.qTable.has(this.prevState)) {
        this.qTable.set(this.prevState, [0.1, 0.1, 0.1]);
      }
      const prevQValues = this.qTable.get(this.prevState);
      const currentQValues = this.qTable.get(currentState);

      // SARSA Update rule
      const target = reward + this.gamma * currentQValues[nextAction];
      prevQValues[this.prevAction] +=
        this.alpha * (target - prevQValues[this.prevAction]);
    }

    this.prevState = currentState;
    this.prevAction = nextAction;

    return {
      action: nextAction, // 0: VAE, 1: Drift, 2: Both
      actionName: this.actionNames[nextAction],
      qValues: this.qTable.get(currentState),
    };
  }

  getStats() {
    const stats = {};
    for (const [state, values] of this.qTable.entries()) {
      stats[state] = values.map((v) => v.toFixed(4));
    }
    return stats;
  }
}
