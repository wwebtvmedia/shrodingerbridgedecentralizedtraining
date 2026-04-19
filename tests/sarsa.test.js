import { SarsaBridgeOptimizer } from "../src/core/sarsa-optimizer.js";

async function testSarsa() {
  console.log("🧪 Testing SarsaBridgeOptimizer...");

  const optimizer = new SarsaBridgeOptimizer(0.1, 0.9, 0.0); // 0 epsilon for deterministic testing

  // Mock a sequence of training steps
  const states = [
    { phase: "vae", loss: 0.8, delta: 0.1, time: 100 },
    { phase: "vae", loss: 0.7, delta: 0.05, time: 100 },
    { phase: "drift", loss: 0.4, delta: 0.2, time: 200 },
    { phase: "drift", loss: 0.2, delta: 0.01, time: 200 },
  ];

  for (const s of states) {
    const result = optimizer.update(s.phase, s.loss, s.delta, s.time);
    console.log(
      `State: ${s.phase}_${s.loss > 0.5 ? "high" : "med"}, Action: ${result.actionName}, Q-Values: ${result.qValues}`,
    );
  }

  const stats = optimizer.getStats();
  console.log("Final Q-Table Stats:", JSON.stringify(stats, null, 2));

  // Basic assertions
  if (Object.keys(stats).length > 0) {
    console.log("✅ SARSA Q-Table updated successfully");
  } else {
    throw new Error("SARSA Q-Table is empty");
  }

  console.log("✅ SARSA tests passed!");
}

// Run test
testSarsa().catch((err) => {
  console.error("❌ SARSA tests failed:", err);
  process.exit(1);
});
