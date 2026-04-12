// Low-Rank Adaptation (LoRA) for TensorFlow.js
// Implementation of LoRA for Linear and Conv2D layers
// Inspired by enhancedoptimaltransport/lora.py

import * as tf from '@tensorflow/tfjs';

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
      'lora_A',
      [inFeatures, this.r],
      'float32',
      tf.initializers.heUniform()
    );

    // LoRA B: [r, outFeatures]
    this.loraB = this.addWeight(
      'lora_B',
      [this.r, outFeatures],
      'float32',
      tf.initializers.zeros()
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
      
      return tf.add(baseOut, loraOut);
    });
  }

  computeOutputShape(inputShape) {
    return this.baseLayer.computeOutputShape(inputShape);
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
      'lora_A',
      [1, 1, inChannels, this.r],
      'float32',
      tf.initializers.heUniform()
    );

    // LoRA B: [1, 1, r, outChannels] (1x1 Conv)
    this.loraB = this.addWeight(
      'lora_B',
      [1, 1, this.r, outChannels],
      'float32',
      tf.initializers.zeros()
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
      loraOut = tf.conv2d(loraOut, this.loraA.read(), strides, 'same');
      loraOut = tf.conv2d(loraOut, this.loraB.read(), [1, 1], 'same');
      loraOut = tf.mul(loraOut, this.scaling);
      
      return tf.add(baseOut, loraOut);
    });
  }

  computeOutputShape(inputShape) {
    return this.baseLayer.computeOutputShape(inputShape);
  }
}

/**
 * Helper to wrap a layer with LoRA if applicable
 */
export function wrapWithLoRA(layer, r = 8, loraAlpha = 16, loraDropout = 0.05) {
  if (layer instanceof tf.layers.Layer) {
    if (layer.getClassName() === 'Dense') {
      return new LoRADense(layer, r, loraAlpha, loraDropout);
    } else if (layer.getClassName() === 'Conv2D') {
      return new LoRAConv2D(layer, r, loraAlpha, loraDropout);
    }
  }
  return layer;
}
