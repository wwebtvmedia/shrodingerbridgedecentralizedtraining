import * as tf from "@tensorflow/tfjs";

async function test() {
  console.log("Setting backend to cpu...");
  await tf.setBackend("cpu");
  console.log("Backend set to:", tf.getBackend());

  try {
    console.log("Attempting tf.zeros...");
    const a = tf.zeros([1]);
    console.log("Success!");
  } catch (e) {
    console.log("Failed:", e.message);

    console.log("Awaiting tf.ready()...");
    await tf.ready();
    console.log("Attempting tf.zeros again...");
    const b = tf.zeros([1]);
    console.log("Success after tf.ready()!");
  }
}

test();
