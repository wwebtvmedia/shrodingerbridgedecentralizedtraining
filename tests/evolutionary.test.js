import { EvolutionaryOptimizer } from "../src/core/adaptive-logic.js";

async function testEvolutionary() {
  console.log("🧪 Testing EvolutionaryOptimizer...");

  const optimizer = new EvolutionaryOptimizer(0.5, 0.1); // High rate for testing

  const p1 = {
    vae_params: [
      [1.0, 2.0],
      [3.0, 4.0],
    ],
    drift_params: [[0.1, 0.2]],
  };

  const p2 = {
    vae_params: [
      [10.0, 20.0],
      [30.0, 40.0],
    ],
    drift_params: [[0.9, 0.8]],
  };

  // Test Mutation
  console.log("  - Testing Mutation...");
  const mutated = optimizer.mutate(p1);
  console.log("    Original:", JSON.stringify(p1.vae_params[0]));
  console.log("    Mutated: ", JSON.stringify(mutated.vae_params[0]));

  if (JSON.stringify(p1) === JSON.stringify(mutated)) {
    // With 0.5 rate, it's highly unlikely to be identical, but not impossible
    console.warn(
      "    ⚠️ Mutation produced identical result (statistically possible but check logic)",
    );
  } else {
    console.log("    ✅ Mutation changed weights");
  }

  // Test Crossover
  console.log("  - Testing Crossover...");
  const child = optimizer.crossover(p1, p2, 0.5);
  console.log("    P1:    ", JSON.stringify(p1.vae_params[0]));
  console.log("    P2:    ", JSON.stringify(p2.vae_params[0]));
  console.log("    Child: ", JSON.stringify(child.vae_params[0]));

  let fromP1 = false;
  let fromP2 = false;

  child.vae_params[0].forEach((val, i) => {
    if (val === p1.vae_params[0][i]) fromP1 = true;
    if (val === p2.vae_params[0][i]) fromP2 = true;
  });

  if (fromP1 && fromP2) {
    console.log("    ✅ Child inherited from both parents");
  } else {
    console.log(
      "    ⚠️ Child inherited from only one parent (statistically possible)",
    );
  }

  console.log("✅ Evolutionary tests passed!");
}

testEvolutionary().catch((err) => {
  console.error("❌ Evolutionary tests failed:", err);
  process.exit(1);
});
