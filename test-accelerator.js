import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-node";
import "@tensorflow/tfjs-backend-wasm";

async function testAccelerator() {
  console.log("--- TFJS Hardware Accelerator Probe ---");
  console.log(`TFJS Version: ${tf.version_core}`);

  const backends = ["tensorflow", "webgpu", "webgl", "wasm", "cpu"];

  for (const backend of backends) {
    try {
      console.log(`\nProbing backend: [${backend.toUpperCase()}]...`);
      const hasBackend = tf.findBackend(backend);
      console.log(`- Backend registered: ${!!hasBackend}`);

      if (hasBackend) {
        const start = Date.now();
        await tf.setBackend(backend);
        const end = Date.now();
        console.log(
          `- Successfully set backend to ${backend} in ${end - start}ms`,
        );

        // Simple computation test
        const a = tf.tensor1d([1, 2, 3]);
        const b = tf.tensor1d([4, 5, 6]);
        const result = a.add(b);
        console.log(`- Computation test (1+4, 2+5, 3+6): ${result.dataSync()}`);
        tf.dispose([a, b, result]);
      }
    } catch (e) {
      console.log(`- Failed to use ${backend}: ${e.message}`);
    }
  }

  console.log("\n--- Probe Complete ---");
  console.log(`Final active backend: ${tf.getBackend()}`);
}

testAccelerator().catch(console.error);
