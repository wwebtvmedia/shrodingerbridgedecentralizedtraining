// Low-Rank Adaptation (LoRA) for TensorFlow.js
// Implementation of LoRA for Linear and Conv2D layers
// Inspired by enhancedoptimaltransport/lora.py

import * as tf from "@tensorflow/tfjs";

/**
 * LoRA Layer Wrapper for Dense (Linear) layers
 */
export class LoRADense extends tf.layers.Layer {
  constructor(baseLayer, r = 8, loraAlpha = 16, loraDropout = 0.05, name) {
    super({ name: name || `${baseLayer.name}/lora` });
    this.baseLayer = baseLayer;
    this.r = r;
    this.loraAlpha = loraAlpha;
    this.loraDropout = loraDropout;
    this.scaling = loraAlpha / r;

    // Freeze base layer weights
    this.baseLayer.trainable = false;
  }

  build(inputShape) {
    const inFeatures = inputShape[inputShape.length - 1];
    const outFeatures = this.baseLayer.units;

    // LoRA A: [inFeatures, r]
    this.loraA = this.addWeight(
      "lora_A",
      [inFeatures, this.r],
      "float32",
      tf.initializers.heUniform(),
    );

    // LoRA B: [r, outFeatures]
    this.loraB = this.addWeight(
      "lora_B",
      [this.r, outFeatures],
      "float32",
      tf.initializers.zeros(),
    );

    this.trainableWeights = [this.loraA, this.loraB];

    this.dropout = tf.layers.dropout({ rate: this.loraDropout });
    this.built = true;
  }

  call(inputs) {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      const baseOut = this.baseLayer.apply(input);

      let loraOut = this.dropout.apply(input);
      loraOut = tf.matMul(loraOut, this.loraA.read());
      loraOut = tf.matMul(loraOut, this.loraB.read());
      loraOut = tf.mul(loraOut, this.scaling);

      try {
        return tf.add(baseOut, loraOut);
      } catch (e) {
        console.error(`LoRADense shape mismatch: baseOut=${baseOut.shape}, loraOut=${loraOut.shape}`);
        throw e;
      }
    });
  }

  computeOutputShape(inputShape) {
    return this.baseLayer.computeOutputShape(inputShape);
  }

  dispose() {
    if (this.baseLayer) this.baseLayer.dispose();
    if (this.dropout) this.dropout.dispose();
    super.dispose();
  }
}

/**
 * LoRA Layer Wrapper for Conv2D layers using 1x1 convolutions
 */
export class LoRAConv2D extends tf.layers.Layer {
  constructor(baseLayer, r = 8, loraAlpha = 16, loraDropout = 0.05, name) {
    super({ name: name || `${baseLayer.name}/lora` });
    this.baseLayer = baseLayer;
    this.r = r;
    this.loraAlpha = loraAlpha;
    this.loraDropout = loraDropout;
    this.scaling = loraAlpha / r;

    // Freeze base layer weights
    this.baseLayer.trainable = false;
  }

  build(inputShape) {
    const inChannels = inputShape[inputShape.length - 1];
    const outChannels = this.baseLayer.filters;

    // LoRA A: [1, 1, inChannels, r] (1x1 Conv)
    this.loraA = this.addWeight(
      "lora_A",
      [1, 1, inChannels, this.r],
      "float32",
      tf.initializers.heUniform(),
    );

    // LoRA B: [1, 1, r, outChannels] (1x1 Conv)
    this.loraB = this.addWeight(
      "lora_B",
      [1, 1, this.r, outChannels],
      "float32",
      tf.initializers.zeros(),
    );

    this.trainableWeights = [this.loraA, this.loraB];

    this.dropout = tf.layers.dropout({ rate: this.loraDropout });
    this.built = true;
  }

  call(inputs) {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      const baseOut = this.baseLayer.apply(input);

      let loraOut = this.dropout.apply(input);
      // Apply 1x1 convolutions as low-rank path
      // IMPORTANT: Use the same strides as the base layer to match output resolution
      const strides = this.baseLayer.strides;
      loraOut = tf.conv2d(loraOut, this.loraA.read(), strides, "same");
      loraOut = tf.conv2d(loraOut, this.loraB.read(), [1, 1], "same");
      loraOut = tf.mul(loraOut, this.scaling);

      try {
        return tf.add(baseOut, loraOut);
      } catch (e) {
        console.error(`${this.getClassName()} shape mismatch: baseOut=${baseOut.shape}, loraOut=${loraOut.shape}`);
        throw e;
      }
    });
  }

  computeOutputShape(inputShape) {
    return this.baseLayer.computeOutputShape(inputShape);
  }

  dispose() {
    if (this.baseLayer) this.baseLayer.dispose();
    if (this.dropout) this.dropout.dispose();
    super.dispose();
  }
}

/**
 * LoRA Layer Wrapper for Conv2DTranspose layers
 */
export class LoRAConv2DTranspose extends tf.layers.Layer {
  constructor(baseLayer, r = 8, loraAlpha = 16, loraDropout = 0.05, name) {
    super({ name: name || `${baseLayer.name}/lora` });
    this.baseLayer = baseLayer;
    this.r = r;
    this.loraAlpha = loraAlpha;
    this.loraDropout = loraDropout;
    this.scaling = loraAlpha / r;

    // Freeze base layer weights
    this.baseLayer.trainable = false;
  }

  build(inputShape) {
    const inChannels = inputShape[inputShape.length - 1];
    const outChannels = this.baseLayer.filters;

    // LoRA A: [1, 1, r, inChannels] (for Conv2DTranspose, weights are [H, W, out, in])
    this.loraA = this.addWeight(
      "lora_A",
      [1, 1, this.r, inChannels],
      "float32",
      tf.initializers.heUniform(),
    );

    // LoRA B: [1, 1, r, outChannels] (Regular Conv2D for channel mapping)
    this.loraB = this.addWeight(
      "lora_B",
      [1, 1, this.r, outChannels],
      "float32",
      tf.initializers.zeros(),
    );

    this.trainableWeights = [this.loraA, this.loraB];

    this.dropout = tf.layers.dropout({ rate: this.loraDropout });
    this.built = true;
  }

  call(inputs) {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      const baseOut = this.baseLayer.apply(input);

      let loraOut = this.dropout.apply(input);
      const strides = this.baseLayer.strides;

      // For transposed convolution, we use conv2dTranspose for the low-rank path too
      loraOut = tf.conv2dTranspose(loraOut, this.loraA.read(), [baseOut.shape[0] || 1, baseOut.shape[1], baseOut.shape[2], this.r], strides, "same");
      loraOut = tf.conv2d(loraOut, this.loraB.read(), [1, 1], "same");
      loraOut = tf.mul(loraOut, this.scaling);

      try {
        return tf.add(baseOut, loraOut);
      } catch (e) {
        console.error(`${this.getClassName()} shape mismatch: baseOut=${baseOut.shape}, loraOut=${loraOut.shape}`);
        throw e;
      }
    });
  }

  computeOutputShape(inputShape) {
    return this.baseLayer.computeOutputShape(inputShape);
  }

  dispose() {
    if (this.baseLayer) this.baseLayer.dispose();
    if (this.dropout) this.dropout.dispose();
    super.dispose();
  }
}

/**
 * Helper to wrap a layer with LoRA if applicable
 */
export function wrapWithLoRA(layer, r = 8, loraAlpha = 16, loraDropout = 0.05) {
  if (layer instanceof tf.layers.Layer) {
    const className = layer.getClassName();
    if (className === "Dense") {
      return new LoRADense(layer, r, loraAlpha, loraDropout);
    } else if (className === "Conv2D") {
      return new LoRAConv2D(layer, r, loraAlpha, loraDropout);
    } else if (className === "Conv2DTranspose") {
      return new LoRAConv2DTranspose(layer, r, loraAlpha, loraDropout);
    }
  }
  return layer;
}
