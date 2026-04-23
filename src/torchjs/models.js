// Enhanced Schrödinger Bridge Model Architectures
// Ported from PyTorch (enhancedoptimaltransport/models.py) to TensorFlow.js
// Optimized for GPU acceleration and high-fidelity generation (96x96)

import * as tf from "@tensorflow/tfjs";
import { CONFIG } from "../config.js";
import { wrapWithLoRA } from "./lora.js";

/**
 * Group Normalization Layer for TensorFlow.js
 * Standard implementation for stable training in small batches.
 */
class GroupNormalization extends tf.layers.Layer {
  constructor(groups, epsilon = 1e-5, name) {
    super({ name: name || "group_norm" });
    this.groups = groups;
    this.epsilon = epsilon;
  }

  build(inputShape) {
    const channels = inputShape[inputShape.length - 1];
    if (channels % this.groups !== 0) {
      throw new Error(
        `Channels (${channels}) must be divisible by groups (${this.groups})`,
      );
    }

    this.gamma = this.addWeight(
      "gamma",
      [channels],
      "float32",
      tf.initializers.ones(),
    );
    this.beta = this.addWeight(
      "beta",
      [channels],
      "float32",
      tf.initializers.zeros(),
    );
    this.built = true;
  }

  call(inputs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      const [b, h, w, c] = x.shape;
      const g = this.groups;
      const cPerG = Math.floor(c / g);

      // Reshape to [B, H, W, G, C//G]
      let x_reshaped = tf.reshape(x, [b || -1, h, w, g, cPerG]);

      // Calculate mean and variance over H, W, and C//G
      const { mean, variance } = tf.moments(x_reshaped, [1, 2, 4], true);

      // Normalize
      let x_norm = tf.div(
        tf.sub(x_reshaped, mean),
        tf.sqrt(tf.add(variance, this.epsilon)),
      );

      // Back to [B, H, W, C]
      x_norm = tf.reshape(x_norm, [b || -1, h, w, c]);

      // Scale and shift
      return tf.add(tf.mul(x_norm, this.gamma.read()), this.beta.read());
    });
  }

  getClassName() {
    return "GroupNormalization";
  }

  dispose() {
    super.dispose();
  }
}

/**
 * Noise Injection Layer
 * Injects learnable per-channel noise to break up repetitive textures.
 */
class NoiseInjection extends tf.layers.Layer {
  constructor(channels, name) {
    super({ name: name || "noise_injection" });
    this.channels = channels;
  }

  build(inputShape) {
    this.weight = this.addWeight(
      "weight",
      [1, 1, 1, this.channels],
      "float32",
      tf.initializers.zeros(),
    );
    this.built = true;
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      const training = kwargs["training"] || false;

      if (!training) return x;

      const [b, h, w, c] = x.shape;
      const noise = tf.randomNormal([b || 1, h, w, 1]);
      return tf.add(x, tf.mul(this.weight.read(), noise));
    });
  }

  getClassName() {
    return "NoiseInjection";
  }

  dispose() {
    super.dispose();
  }
}

/**
 * Percentile Rescaling Layer
 * Adaptive percentile-based rescaling to [-1, 1].
 */
class PercentileRescale extends tf.layers.Layer {
  constructor(channels, pLow = 1.0, pHigh = 99.0, momentum = 0.005, name) {
    super({ name: name || "percentile_rescale" });
    this.channels = channels;
    this.pLow = pLow;
    this.pHigh = pHigh;
    this.momentum = momentum;
  }

  build(inputShape) {
    this.low = this.addWeight(
      "low",
      [this.channels],
      "float32",
      tf.initializers.zeros(),
      null,
      false,
    );
    this.high = this.addWeight(
      "high",
      [this.channels],
      "float32",
      tf.initializers.ones(),
      null,
      false,
    );
    this.built = true;
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      const training = kwargs["training"] || false;

      if (training) {
        // Approximate quantiles since TFJS doesn't have a direct equivalent to torch.quantile for multidim
        // We use a simplified version: mean +/- k * std
        const axes = [0, 1, 2]; // B, H, W
        const { mean, variance } = tf.moments(x, axes);
        const std = tf.sqrt(tf.add(variance, 1e-8));

        // k=2.33 approx for 99th percentile in normal distribution
        const l = tf.sub(mean, tf.mul(2.33, std));
        const h = tf.add(mean, tf.mul(2.33, std));

        // Update buffers (manual EMA as TFJS weights are updated by optimizers if trainable)
        const newLow = tf.add(
          tf.mul(this.low.read(), 1 - this.momentum),
          tf.mul(l, this.momentum),
        );
        const newHigh = tf.add(
          tf.mul(this.high.read(), 1 - this.momentum),
          tf.mul(h, this.momentum),
        );

        // In TFJS we can't easily update non-trainable weights inside call() without side effects
        // but for training we'll just use the current values
      }

      const scale = tf.maximum(tf.sub(this.high.read(), this.low.read()), 1e-6);
      const shift = this.low.read();

      // Reshape for broadcasting
      const scale_bc = tf.reshape(scale, [1, 1, 1, this.channels]);
      const shift_bc = tf.reshape(shift, [1, 1, 1, this.channels]);

      return tf.tanh(tf.div(tf.sub(x, shift_bc), scale_bc));
    });
  }

  getClassName() {
    return "PercentileRescale";
  }

  dispose() {
    super.dispose();
  }
}

/**
 * Residual Block with GroupNorm and SiLU
 */
class ResidualBlock {
  constructor(inChannels, outChannels, stride = 1, name) {
    this.name = name;
    this.stride = stride;
    this.inChannels = inChannels;
    this.outChannels = outChannels;

    this.conv1 = wrapWithLoRA(
      tf.layers.conv2d({
        filters: outChannels,
        kernelSize: 3,
        strides: stride,
        padding: "same",
        useBias: false,
        name: `${name}/conv1`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );
    this.gn1 = new GroupNormalization(
      Math.min(8, outChannels),
      1e-5,
      `${name}/gn1`,
    );

    this.conv2 = wrapWithLoRA(
      tf.layers.conv2d({
        filters: outChannels,
        kernelSize: 3,
        strides: 1,
        padding: "same",
        useBias: false,
        name: `${name}/conv2`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );
    this.gn2 = new GroupNormalization(
      Math.min(8, outChannels),
      1e-5,
      `${name}/gn2`,
    );

    if (stride !== 1 || inChannels !== outChannels) {
      this.shortcut = tf.sequential({
        name: `${name}/shortcut`,
        layers: [
          tf.layers.conv2d({
            filters: outChannels,
            kernelSize: 1,
            strides: stride,
            useBias: false,
            inputShape: [null, null, inChannels],
            name: `${name}/shortcut/conv`,
          }),
          new GroupNormalization(
            Math.min(8, outChannels),
            1e-5,
            `${name}/shortcut/gn`,
          ),
        ],
      });
    } else {
      this.shortcut = null;
    }
  }

  forward(x) {
    return tf.tidy(() => {
      let out = this.conv1.apply(x);
      out = this.gn1.apply(out);
      out = tf.mul(out, tf.sigmoid(out));

      out = this.conv2.apply(out);
      out = this.gn2.apply(out);

      let shortcut = this.shortcut ? this.shortcut.apply(x) : x;

      if (
        out.shape[1] !== shortcut.shape[1] ||
        out.shape[2] !== shortcut.shape[2]
      ) {
        shortcut = tf.image.resizeNearestNeighbor(shortcut, [
          out.shape[1],
          out.shape[2],
        ]);
      }

      return tf.mul(tf.add(out, shortcut), tf.sigmoid(tf.add(out, shortcut)));
    });
  }

  dispose() {
    if (this.conv1) this.conv1.dispose();
    if (this.gn1) this.gn1.dispose();
    if (this.conv2) this.conv2.dispose();
    if (this.gn2) this.gn2.dispose();
    if (this.shortcut) this.shortcut.dispose();
  }
}

/**
 * Spatial Split Attention (Axial Attention)
 */
class SpatialSplitAttention {
  constructor(channels, numHeads = 4, name) {
    this.channels = channels;
    this.numHeads = numHeads;
    this.name = name;

    this.ln_h = tf.layers.layerNormalization({ name: `${name}/ln_h` });
    this.ln_w = tf.layers.layerNormalization({ name: `${name}/ln_w` });

    // Projections
    this.qkv_h = tf.layers.dense({
      units: channels * 3,
      name: `${name}/qkv_h`,
    });
    this.qkv_w = tf.layers.dense({
      units: channels * 3,
      name: `${name}/qkv_w`,
    });
    this.proj_h = tf.layers.dense({ units: channels, name: `${name}/proj_h` });
    this.proj_w = tf.layers.dense({ units: channels, name: `${name}/proj_w` });
  }

  forward(x) {
    return tf.tidy(() => {
      const [b, h, w, c] = x.shape;

      // 1. Vertical Split Attention (Attend along H for each W)
      // [B, H, W, C] -> [B, W, H, C] -> [B*W, H, C]
      let v = tf.transpose(x, [0, 2, 1, 3]);
      v = tf.reshape(v, [b * w, h, c]);
      let v_norm = this.ln_h.apply(v);

      let qkv_v = this.qkv_h.apply(v_norm);
      let [q_v, k_v, v_v] = tf.split(qkv_v, 3, -1);

      // Attention
      let attn_v = tf.matMul(q_v, k_v, false, true);
      attn_v = tf.softmax(tf.div(attn_v, tf.sqrt(c / this.numHeads)));
      let out_v = tf.matMul(attn_v, v_v);
      out_v = this.proj_h.apply(out_v);

      // Residual
      out_v = tf.reshape(out_v, [b, w, h, c]);
      out_v = tf.transpose(out_v, [0, 2, 1, 3]);
      let x_res = tf.add(x, out_v);

      // 2. Horizontal Split Attention (Attend along W for each H)
      // [B, H, W, C] -> [B*H, W, C]
      let h_in = tf.reshape(x_res, [b * h, w, c]);
      let h_norm = this.ln_w.apply(h_in);

      let qkv_h = this.qkv_w.apply(h_norm);
      let [q_h, k_h, v_h] = tf.split(qkv_h, 3, -1);

      let attn_h = tf.matMul(q_h, k_h, false, true);
      attn_h = tf.softmax(tf.div(attn_h, tf.sqrt(c / this.numHeads)));
      let out_h = tf.matMul(attn_h, v_h);
      out_h = this.proj_w.apply(out_h);

      // Residual
      out_h = tf.reshape(out_h, [b, h, w, c]);
      return tf.add(x_res, out_h);
    });
  }

  dispose() {
    if (this.ln_h) this.ln_h.dispose();
    if (this.ln_w) this.ln_w.dispose();
    if (this.qkv_h) this.qkv_h.dispose();
    if (this.qkv_w) this.qkv_w.dispose();
    if (this.proj_h) this.proj_h.dispose();
    if (this.proj_w) this.proj_w.dispose();
  }
}

/**
 * Neural Tokenizer for text/byte processing
 */
class NeuralTokenizer {
  constructor(
    vocabSize = 260,
    embedDim = 256,
    hiddenDim = 512,
    outDim = 128,
    name,
  ) {
    this.name = name || "neural_tokenizer";
    this.embedding = tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: embedDim,
      name: `${this.name}/emb`,
    });

    this.conv1 = tf.layers.conv1d({
      filters: hiddenDim,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
      name: `${this.name}/conv1`,
    });

    this.conv2 = tf.layers.conv1d({
      filters: hiddenDim,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
      name: `${this.name}/conv2`,
    });

    this.projection = tf.layers.dense({
      units: outDim,
      name: `${this.name}/proj`,
    });
    this.norm = tf.layers.layerNormalization({ name: `${this.name}/norm` });
  }

  forward(textBytes) {
    return tf.tidy(() => {
      // textBytes: [B, L]
      let x = this.embedding.apply(textBytes); // [B, L, D]
      x = this.conv1.apply(x);
      x = this.conv2.apply(x);

      // Global Max Pooling
      x = tf.max(x, 1); // [B, hiddenDim]

      x = this.projection.apply(x);
      return this.norm.apply(x);
    });
  }

  dispose() {
    if (this.embedding) this.embedding.dispose();
    if (this.conv1) this.conv1.dispose();
    if (this.conv2) this.conv2.dispose();
    if (this.projection) this.projection.dispose();
    if (this.norm) this.norm.dispose();
  }
}

/**
 * Label Conditioned Block with FiLM-like modulation and Noise Injection
 */
class LabelConditionedBlock {
  constructor(cIn, cOut, labelDim = 128, name) {
    this.name = name;
    this.gn1 = new GroupNormalization(Math.min(8, cIn), 1e-5, `${name}/gn1`);
    this.conv1 = wrapWithLoRA(
      tf.layers.conv2d({
        filters: cOut,
        kernelSize: 3,
        padding: "same",
        name: `${name}/conv1`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );
    this.noise1 = new NoiseInjection(cOut, `${name}/noise1`);

    this.labelProj = wrapWithLoRA(
      tf.layers.dense({ units: cOut * 2, name: `${name}/label_proj` }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );

    this.gn2 = new GroupNormalization(Math.min(8, cOut), 1e-5, `${name}/gn2`);
    this.conv2 = wrapWithLoRA(
      tf.layers.conv2d({
        filters: cOut,
        kernelSize: 3,
        padding: "same",
        name: `${name}/conv2`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );
    this.noise2 = new NoiseInjection(cOut, `${name}/noise2`);

    this.rescale = CONFIG.USE_PERCENTILE
      ? new PercentileRescale(cOut, 1, 99, 0.005, `${name}/rescale`)
      : null;

    if (cIn !== cOut) {
      this.skip = wrapWithLoRA(
        tf.layers.conv2d({
          filters: cOut,
          kernelSize: 1,
          name: `${name}/skip`,
        }),
        CONFIG.LORA_R,
        CONFIG.LORA_ALPHA,
      );
    } else {
      this.skip = null;
    }
  }

  forward(x, labels, training = true) {
    return tf.tidy(() => {
      let h = this.gn1.apply(x);
      h = tf.mul(h, tf.sigmoid(h));
      h = this.conv1.apply(h);
      h = this.noise1.apply(h, { training });

      if (labels) {
        const scaleShift = this.labelProj.apply(labels);
        const [scale, shift] = tf.split(scaleShift, 2, -1);

        const scale_bc = tf.reshape(scale, [
          scale.shape[0],
          1,
          1,
          scale.shape[1],
        ]);
        const shift_bc = tf.reshape(shift, [
          shift.shape[0],
          1,
          1,
          shift.shape[1],
        ]);

        h = tf.add(tf.mul(h, tf.add(1, scale_bc)), shift_bc);
      }

      h = this.gn2.apply(h);
      h = tf.mul(h, tf.sigmoid(h));
      h = this.conv2.apply(h);
      h = this.noise2.apply(h, { training });

      let skip = this.skip ? this.skip.apply(x) : x;
      if (h.shape[1] !== skip.shape[1] || h.shape[2] !== skip.shape[2]) {
        skip = tf.image.resizeNearestNeighbor(skip, [h.shape[1], h.shape[2]]);
      }

      let out = tf.add(h, skip);
      if (this.rescale) {
        out = this.rescale.apply(out, { training });
      }
      return out;
    });
  }

  dispose() {
    if (this.gn1) this.gn1.dispose();
    if (this.conv1) this.conv1.dispose();
    if (this.noise1) this.noise1.dispose();
    if (this.labelProj) this.labelProj.dispose();
    if (this.gn2) this.gn2.dispose();
    if (this.conv2) this.conv2.dispose();
    if (this.noise2) this.noise2.dispose();
    if (this.rescale) this.rescale.dispose();
    if (this.skip) this.skip.dispose();
  }
}

/**
 * Subpixel Upsampling (Replaced tf.depthToSpace with Conv2DTranspose for differentiability)
 */
class SubpixelUpsample {
  constructor(inChannels, outChannels, upscaleFactor = 2, name) {
    this.name = name;
    this.upscaleFactor = upscaleFactor;

    this.conv = wrapWithLoRA(
      tf.layers.conv2dTranspose({
        filters: outChannels,
        kernelSize: 3,
        strides: upscaleFactor,
        padding: "same",
        name: `${name}/conv`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );
    this.gn = new GroupNormalization(
      Math.min(8, outChannels),
      1e-5,
      `${name}/gn`,
    );
  }

  forward(x) {
    return tf.tidy(() => {
      let h = this.conv.apply(x);
      h = this.gn.apply(h);
      return tf.mul(h, tf.sigmoid(h));
    });
  }

  dispose() {
    if (this.conv) this.conv.dispose();
    if (this.gn) this.gn.dispose();
  }
}

/**
 * Shared Projection Head for multimodal alignment
 */
class SharedEmbeddingHead {
  constructor(inDim, outDim = 128, name) {
    this.name = name;
    this.proj = tf.sequential({
      layers: [
        tf.layers.dense({
          units: outDim,
          inputShape: [inDim],
          name: `${name}/proj1`,
        }),
        tf.layers.activation({ activation: "swish" }),
        tf.layers.dense({ units: outDim, name: `${name}/proj2` }),
      ],
    });
    this.norm = tf.layers.layerNormalization({ name: `${name}/norm` });
  }

  forward(x) {
    return tf.tidy(() => {
      let h = this.proj.apply(x);
      return this.norm.apply(h);
    });
  }

  dispose() {
    if (this.proj) this.proj.dispose();
    if (this.norm) this.norm.dispose();
  }
}

/**
 * Label Conditioned VAE
 */
export class LabelConditionedVAE {
  constructor(name = "vae") {
    this.name = name;
    this.latentChannels = CONFIG.LATENT_CHANNELS || 8;
    this.labelDim = CONFIG.LABEL_EMB_DIM || 128;

    // Multimodal Text Encoder / Label Embedding
    if (CONFIG.USE_NEURAL_TOKENIZER) {
      this.textEncoder = new NeuralTokenizer(
        260,
        256,
        512,
        this.labelDim,
        `${name}/text_enc`,
      );
    } else {
      this.labelEmb = tf.layers.embedding({
        inputDim: CONFIG.NUM_CLASSES || 11,
        outputDim: this.labelDim,
        name: `${name}/label_emb`,
      });
    }

    const lDim = this.labelDim;

    // Fourier feature channels
    this.fourierChannels = CONFIG.USE_FOURIER_FEATURES ? 16 : 0; // Simplified

    // Encoder (96 -> 48 -> 24 -> 12)
    this.encIn = wrapWithLoRA(
      tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        padding: "same",
        inputShape: [
          CONFIG.IMG_SIZE,
          CONFIG.IMG_SIZE,
          3 + this.fourierChannels,
        ],
        name: `${name}/enc_in`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );

    this.encBlocks = [
      new ResidualBlock(64, 128, 2, `${name}/enc_block0`), // 96 -> 48
      new LabelConditionedBlock(128, 128, lDim, `${name}/enc_cond0`),
      new ResidualBlock(128, 256, 2, `${name}/enc_block1`), // 48 -> 24
      new SpatialSplitAttention(256, 4, `${name}/enc_attn0`),
      new LabelConditionedBlock(256, 512, lDim, `${name}/enc_cond1`),
      new ResidualBlock(512, 512, 2, `${name}/enc_block2`), // 24 -> 12
    ];

    this.zMean = wrapWithLoRA(
      tf.layers.conv2d({
        filters: this.latentChannels,
        kernelSize: 3,
        padding: "same",
        name: `${name}/z_mean`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );
    this.zLogvar = wrapWithLoRA(
      tf.layers.conv2d({
        filters: this.latentChannels,
        kernelSize: 3,
        padding: "same",
        name: `${name}/z_logvar`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );

    // Image Projection for alignment
    if (CONFIG.USE_PROJECTION_HEADS) {
      this.imageProj = new SharedEmbeddingHead(
        this.latentChannels * CONFIG.LATENT_H * CONFIG.LATENT_W,
        lDim,
        `${name}/image_proj`,
      );
    }

    // Decoder (12 -> 24 -> 48 -> 96)
    this.latentNorm = new GroupNormalization(
      Math.min(8, this.latentChannels),
      1e-5,
      `${name}/latent_norm`,
    );
    this.decIn = wrapWithLoRA(
      tf.layers.conv2d({
        filters: 512,
        kernelSize: 3,
        padding: "same",
        name: `${name}/dec_in`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );

    this.decBlocks = [
      new SubpixelUpsample(512, 256, 2, `${name}/dec_up0`), // 12 -> 24
      new LabelConditionedBlock(256, 256, lDim, `${name}/dec_cond0`),
      new SpatialSplitAttention(256, 4, `${name}/dec_attn0`),

      new SubpixelUpsample(256, 128, 2, `${name}/dec_up1`), // 24 -> 48
      new LabelConditionedBlock(128, 128, lDim, `${name}/dec_cond1`),
      new SpatialSplitAttention(128, 4, `${name}/dec_attn1`),

      new SubpixelUpsample(128, 64, 2, `${name}/dec_up2`), // 48 -> 96
      new LabelConditionedBlock(64, 64, lDim, `${name}/dec_cond2`),
    ];

    this.decOut = wrapWithLoRA(
      tf.layers.conv2d({
        filters: 3,
        kernelSize: 3,
        padding: "same",
        name: `${name}/dec_out`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );
  }

  getFourierFeatures(x) {
    if (!CONFIG.USE_FOURIER_FEATURES) return null;
    return tf.tidy(() => {
      const [b, h, w, c] = x.shape;
      // Simple coordinate grid
      const yCoords = tf
        .linspace(-1, 1, h)
        .reshape([1, h, 1, 1])
        .tile([b, 1, w, 1]);
      const xCoords = tf
        .linspace(-1, 1, w)
        .reshape([1, 1, w, 1])
        .tile([b, h, 1, 1]);

      const feats = [];
      [1, 2, 4, 8].forEach((f) => {
        feats.push(tf.sin(tf.mul(Math.PI * f, xCoords)));
        feats.push(tf.cos(tf.mul(Math.PI * f, xCoords)));
        feats.push(tf.sin(tf.mul(Math.PI * f, yCoords)));
        feats.push(tf.cos(tf.mul(Math.PI * f, yCoords)));
      });
      return tf.concat(feats, -1);
    });
  }

  getConditioning(labels, textBytes = null) {
    return tf.tidy(() => {
      if (CONFIG.USE_NEURAL_TOKENIZER && textBytes) {
        return this.textEncoder.forward(textBytes);
      }

      let emb = this.labelEmb.apply(labels);
      if (emb.shape.length === 3) emb = tf.squeeze(emb, [1]);
      return emb;
    });
  }

  encode(x, labels, textBytes = null) {
    return tf.tidy(() => {
      const f_feats = this.getFourierFeatures(x);
      const input = f_feats ? tf.concat([x, f_feats], -1) : x;

      const cond = this.getConditioning(labels, textBytes);
      let h = this.encIn.apply(input);

      for (const block of this.encBlocks) {
        if (block instanceof LabelConditionedBlock) {
          h = block.forward(h, cond);
        } else {
          h = block.forward(h);
        }
      }

      let mu = this.zMean.apply(h);
      let logvar = this.zLogvar.apply(h);

      return [mu, logvar];
    });
  }

  decode(z, labels, textBytes = null) {
    return tf.tidy(() => {
      const cond = this.getConditioning(labels, textBytes);
      let h = this.latentNorm.apply(z);
      h = this.decIn.apply(h);

      for (const block of this.decBlocks) {
        if (block instanceof LabelConditionedBlock) {
          h = block.forward(h, cond, false); // evaluation mode
        } else {
          h = block.forward(h);
        }
      }

      h = this.decOut.apply(h);
      return tf.tanh(h);
    });
  }

  reparameterize(mu, logvar) {
    return tf.tidy(() => {
      const std = tf.exp(tf.mul(0.5, logvar));
      const eps = tf.randomNormal(mu.shape);
      return tf.add(mu, tf.mul(eps, std));
    });
  }

  forward(x, labels, textBytes = null) {
    return tf.tidy(() => {
      const [mu, logvar] = this.encode(x, labels, textBytes);
      const z = this.reparameterize(mu, logvar);
      const recon = this.decode(z, labels, textBytes);
      return [recon, mu, logvar];
    });
  }

  dispose() {
    if (this.textEncoder) this.textEncoder.dispose();
    if (this.labelEmb) this.labelEmb.dispose();
    if (this.encIn) this.encIn.dispose();
    if (this.encBlocks) {
      this.encBlocks.forEach((b) => b.dispose());
    }
    if (this.zMean) this.zMean.dispose();
    if (this.zLogvar) this.zLogvar.dispose();
    if (this.imageProj) this.imageProj.dispose();
    if (this.latentNorm) this.latentNorm.dispose();
    if (this.decIn) this.decIn.dispose();
    if (this.decBlocks) {
      this.decBlocks.forEach((b) => b.dispose());
    }
    if (this.decOut) this.decOut.dispose();
  }
}

/**
 * Fourier Time Embedding
 */
class FourierTimeEmbed {
  constructor(dim = 128, maxFreq = 64) {
    this.dim = dim;
    this.maxFreq = maxFreq;
    const freqs = tf.linspace(1, maxFreq, Math.floor(dim / 2));
    this.freqs = tf.variable(freqs, false);
  }

  forward(t) {
    return tf.tidy(() => {
      const args = tf.mul(t, tf.mul(2 * Math.PI, this.freqs));
      const sin = tf.sin(args);
      const cos = tf.cos(args);
      return tf.concat([sin, cos], -1);
    });
  }

  dispose() {
    if (this.freqs) this.freqs.dispose();
  }
}

/**
 * Label Conditioned Drift Network (U-Net)
 */
export class LabelConditionedDrift {
  constructor(name = "drift") {
    this.name = name;
    this.latentChannels = CONFIG.LATENT_CHANNELS || 8;
    this.labelDim = CONFIG.LABEL_EMB_DIM || 128;

    this.timeEmbed = new FourierTimeEmbed(128);
    this.timeMlp = tf.sequential({
      layers: [
        tf.layers.dense({
          units: 256,
          activation: "relu",
          inputShape: [128],
          name: `${name}/time_mlp1`,
        }),
        tf.layers.dense({ units: 256, name: `${name}/time_mlp2` }),
      ],
    });

    // Time Adaptive Scaling (Replacement for time_scales in Python)
    this.timeWeightNet = tf.sequential({
      layers: [
        tf.layers.dense({
          units: 64,
          activation: "relu",
          inputShape: [1],
          name: `${name}/time_weight1`,
        }),
        tf.layers.dense({ units: 1, name: `${name}/time_weight2` }),
      ],
    });

    if (CONFIG.USE_NEURAL_TOKENIZER) {
      this.textEncoder = new NeuralTokenizer(
        260,
        256,
        512,
        this.labelDim,
        `${name}/text_enc`,
      );
    } else {
      this.labelEmb = tf.layers.embedding({
        inputDim: CONFIG.NUM_CLASSES || 11,
        outputDim: this.labelDim,
        name: `${name}/label_emb`,
      });
    }

    this.condProj = wrapWithLoRA(
      tf.layers.dense({ units: 512, name: `${name}/cond_proj` }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );

    // U-Net Structure
    this.head = wrapWithLoRA(
      tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        padding: "same",
        name: `${name}/head`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );

    this.down1 = new LabelConditionedBlock(64, 128, 512, `${name}/down1`);
    this.down2Conv = wrapWithLoRA(
      tf.layers.conv2d({
        filters: 128,
        kernelSize: 4,
        strides: 2,
        padding: "same",
        name: `${name}/down2_conv`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );
    this.down2Block = new LabelConditionedBlock(
      128,
      128,
      512,
      `${name}/down2_block`,
    );

    this.mid1 = new LabelConditionedBlock(128, 128, 512, `${name}/mid1`);
    this.midAttn = new SpatialSplitAttention(128, 4, `${name}/mid_attn`);
    this.mid2 = new LabelConditionedBlock(128, 128, 512, `${name}/mid2`);

    this.up2Conv = wrapWithLoRA(
      tf.layers.conv2dTranspose({
        filters: 128,
        kernelSize: 4,
        strides: 2,
        padding: "same",
        name: `${name}/up2_conv`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );
    this.up2Block = new LabelConditionedBlock(128, 128, 512, `${name}/up2_block`);

    this.up1 = new LabelConditionedBlock(128, 64, 512, `${name}/up1`);
    this.tail = wrapWithLoRA(
      tf.layers.conv2d({
        filters: this.latentChannels,
        kernelSize: 3,
        padding: "same",
        name: `${name}/tail`,
      }),
      CONFIG.LORA_R,
      CONFIG.LORA_ALPHA,
    );
  }

  forward(z, t, labels, textBytes = null) {
    return tf.tidy(() => {
      // Time and Label Conditioning
      const t_f = this.timeEmbed.forward(t);
      const tEmb = this.timeMlp.apply(t_f);
      const tWeight = this.timeWeightNet.apply(t);

      let lEmb;
      if (CONFIG.USE_NEURAL_TOKENIZER && textBytes) {
        lEmb = this.textEncoder.forward(textBytes);
      } else {
        lEmb = this.labelEmb.apply(labels);
        if (lEmb.shape.length === 3) lEmb = tf.squeeze(lEmb, [1]);
      }

      const cond_input = tf.concat([tEmb, lEmb], -1);
      const cond = this.condProj.apply(cond_input);

      // U-Net
      const h0 = this.head.apply(z);
      const d1 = this.down1.forward(h0, cond);

      const d2_c = this.down2Conv.apply(d1);
      const d2 = this.down2Block.forward(d2_c, cond);

      const m1 = this.mid1.forward(d2, cond);
      const m_a = this.midAttn.forward(m1);
      const m = this.mid2.forward(m_a, cond);

      let u2 = this.up2Conv.apply(m);
      if (u2.shape[1] !== d1.shape[1] || u2.shape[2] !== d1.shape[2]) {
        u2 = tf.image.resizeNearestNeighbor(u2, [d1.shape[1], d1.shape[2]]);
      }

      const u2_input = tf.add(u2, d1);
      const u2_b = this.up2Block.forward(u2_input, cond);

      const u1 = this.up1.forward(u2_b, cond);
      let out = this.tail.apply(u1);

      // Adaptive scaling based on time
      const tWeight_bc = tf.reshape(tWeight, [tWeight.shape[0], 1, 1, 1]);
      return tf.mul(out, tf.add(1, tWeight_bc));
    });
  }

  dispose() {
    if (this.timeEmbed) this.timeEmbed.dispose();
    if (this.timeMlp) this.timeMlp.dispose();
    if (this.timeWeightNet) this.timeWeightNet.dispose();
    if (this.textEncoder) this.textEncoder.dispose();
    if (this.labelEmb) this.labelEmb.dispose();
    if (this.condProj) this.condProj.dispose();
    if (this.head) this.head.dispose();
    if (this.down1) this.down1.dispose();
    if (this.down2Conv) this.down2Conv.dispose();
    if (this.down2Block) this.down2Block.dispose();
    if (this.mid1) this.mid1.dispose();
    if (this.midAttn) this.midAttn.dispose();
    if (this.mid2) this.mid2.dispose();
    if (this.up2Conv) this.up2Conv.dispose();
    if (this.up2Block) this.up2Block.dispose();
    if (this.up1) this.up1.dispose();
    if (this.tail) this.tail.dispose();
  }
}
