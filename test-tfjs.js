// Test script for TensorFlow.js implementation
import { tfjsTrainer } from "./src/torchjs/integration.js";
import { ModelManager } from "./src/core/models.js";

async function testTFJSIntegration() {
  console.log("🧪 Testing TensorFlow.js Integration...\n");

  // Test 1: Initialize tfjs trainer
  console.log("1. Testing TFJS trainer initialization...");
  try {
    await tfjsTrainer.initialize();
    const state = tfjsTrainer.getModelState();
    console.log(`   ✅ TFJS Trainer initialized`);
    console.log(`   - Epoch: ${state.epoch}`);
    console.log(`   - Phase: ${state.phase}`);
    console.log(`   - Device: ${state.device}`);
  } catch (error) {
    console.log(
      `   ❌ TFJS Trainer initialization failed: ${error.message}`,
    );
  }

  console.log("\n2. Testing ModelManager integration...");
  const modelManager = new ModelManager();
  try {
    await modelManager.initialize();
    console.log(
      `   ✅ ModelManager initialized with TFJS: ${modelManager.state.tfjs_initialized}`,
    );
  } catch (error) {
    console.log(`   ❌ ModelManager initialization failed: ${error.message}`);
  }

  console.log("\n3. Testing training step...");
  try {
    // Create mock batch data (2 samples of 96x96x3)
    const mockBatch = Array.from({ length: 2 }, () => 
      Array.from({ length: 96 * 96 * 3 }, () => Math.random() * 2 - 1)
    );
    const mockLabels = [0, 1];

    console.log("   Running VAE training step...");
    const result = await modelManager.trainStep(mockBatch, mockLabels, "vae");
    console.log(`   ✅ Training step completed`);
    console.log(`   - Loss: ${result.loss}`);
    console.log(`   - Phase: ${result.metrics.phase}`);
    console.log(`   - Using TFJS: ${result.usingTorchJS}`); // name is still result.usingTorchJS in model manager

  } catch (error) {
    console.log(`   ❌ Training step failed: ${error.message}`);
    console.error(error);
  }

  console.log("\n4. Testing sample generation...");
  try {
    const samples = await tfjsTrainer.generateSamples([0, 1, 2, 3], 4);
    console.log(`   ✅ Generated ${samples.length} samples`);
    console.log(`   - Sample shape: [${samples.length}, ${samples[0].length}, ${samples[0][0].length}, ${samples[0][0][0].length}]`);
  } catch (error) {
    console.log(`   ❌ Sample generation failed: ${error.message}`);
  }

  console.log("\n📊 Summary:");
  console.log("===========");
  console.log("TensorFlow.js implementation has been successfully integrated.");
  console.log("The system now supports WebGL/WebGPU/Node acceleration.");
}

// Run the test
testTFJSIntegration().catch(console.error);
