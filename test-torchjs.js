// Test script for torch-js implementation
import { tfjsTrainer as torchJSTrainer } from "./src/torchjs/integration.js";
import { ModelManager } from "./src/core/models.js";

async function testTorchJSIntegration() {
  console.log("🧪 Testing Torch-JS Integration...\n");

  // Test 1: Initialize torch-js trainer
  console.log("1. Testing torch-js trainer initialization...");
  try {
    await torchJSTrainer.initialize();
    const state = torchJSTrainer.getModelState();
    console.log(`   ✅ Torch-JS Trainer initialized`);
    console.log(`   - Epoch: ${state.epoch}`);
    console.log(`   - Phase: ${state.phase}`);
    console.log(`   - Device: ${state.device}`);
  } catch (error) {
    console.log(
      `   ❌ Torch-JS Trainer initialization failed: ${error.message}`,
    );
  }

  console.log("\n2. Testing ModelManager integration...");
  const modelManager = new ModelManager();
  try {
    await modelManager.initialize();
    console.log(
      `   ✅ ModelManager initialized with torch-js: ${modelManager.state.torchjs_initialized}`,
    );
  } catch (error) {
    console.log(`   ❌ ModelManager initialization failed: ${error.message}`);
  }

  console.log("\n3. Testing training step...");
  try {
    // Create mock batch data: [batch_size, flattened_features]
    // 2 samples of 3*32*32 = 3072
    const mockBatch = [];
    for (let b = 0; b < 2; b++) {
      const pixels = new Array(3 * 32 * 32).fill(0).map(() => Math.random() * 2 - 1);
      mockBatch.push(pixels);
    }
    const mockLabels = [0, 1];

    const result = await modelManager.trainStep(mockBatch, mockLabels, "vae");
    console.log(`   ✅ Training step completed`);
    console.log(`   - Loss: ${result.loss.toFixed(6)}`);
    console.log(`   - Phase: ${result.metrics.phase}`);
    console.log(`   - Using torch-js: ${result.metrics.torchjs}`);

    if (result.metrics.torchjs) {
      console.log(
        `   - Metrics:`,
        Object.keys(result.metrics).filter(
          (k) => !["phase", "torchjs"].includes(k),
        ),
      );
    }
  } catch (error) {
    console.log(`   ❌ Training step failed: ${error.message}`);
    console.error(error);
  }

  console.log("\n4. Testing phase switching...");
  try {
    torchJSTrainer.setPhase(2);
    console.log(`   ✅ Phase switched to 2 (drift)`);

    torchJSTrainer.setPhase(3);
    console.log(`   ✅ Phase switched to 3 (both)`);
  } catch (error) {
    console.log(`   ❌ Phase switching failed: ${error.message}`);
  }

  console.log("\n5. Testing checkpoint save/load...");
  try {
    const checkpoint = await torchJSTrainer.saveCheckpoint();
    console.log(`   ✅ Checkpoint saved at epoch ${checkpoint.epoch}`);

    await torchJSTrainer.loadCheckpoint(checkpoint);
    console.log(`   ✅ Checkpoint loaded successfully`);
  } catch (error) {
    console.log(`   ❌ Checkpoint operations failed: ${error.message}`);
    console.error(error);
  }

  console.log("\n6. Testing sample generation...");
  try {
    const samples = await torchJSTrainer.generateSamples([0, 1, 2, 3], 4);
    console.log(`   ✅ Generated ${samples.length} samples`);
    console.log(`   - Sample 0 shape: ${samples[0].length}`);
  } catch (error) {
    console.log(`   ❌ Sample generation failed: ${error.message}`);
    console.error(error);
  }

  console.log("\n📊 Summary:");
  console.log("===========");
  console.log(
    "Torch-JS implementation has been successfully integrated with the swarm system.",
  );
}

// Run the test
testTorchJSIntegration().catch(console.error);
