// TensorFlow.js implementation of Schrödinger Bridge models
// Ported from js-pytorch version for GPU (WebGL/WebGPU) acceleration

import * as tf from '@tensorflow/tfjs';
import { CONFIG } from "../config.js";

/**
 * Residual Block for TF.js using tf.layers for proper variable tracking
 */
class ResidualBlock {
  constructor(in_channels, out_channels, stride = 1) {
    this.stride = stride;
    
    this.conv1 = tf.layers.conv2d({
      filters: out_channels,
      kernelSize: 3,
      strides: stride,
      padding: 'same',
      useBias: false,
      kernelInitializer: 'heNormal'
    });
    this.bn1 = tf.layers.batchNormalization();
    
    this.conv2 = tf.layers.conv2d({
      filters: out_channels,
      kernelSize: 3,
      strides: 1,
      padding: 'same',
      useBias: false,
      kernelInitializer: 'heNormal'
    });
    this.bn2 = tf.layers.batchNormalization();

    if (in_channels !== out_channels || stride !== 1) {
      this.shortcut = tf.layers.conv2d({
        filters: out_channels,
        kernelSize: 1,
        strides: stride,
        padding: 'same',
        useBias: false
      });
      this.bn_shortcut = tf.layers.batchNormalization();
    } else {
      this.shortcut = null;
    }
  }

  apply(x) {
    let identity = x;
    if (this.shortcut) {
      identity = this.bn_shortcut.apply(this.shortcut.apply(x));
    }

    let out = this.conv1.apply(x);
    out = this.bn1.apply(out);
    out = tf.leakyRelu(out, 0.2);

    out = this.conv2.apply(out);
    out = this.bn2.apply(out);

    return tf.add(out, identity);
  }
}

/**
 * Label Conditioned Block for TF.js
 */
class LabelConditionedBlock {
  constructor(c_in, c_out, label_dim = CONFIG.LABEL_EMB_DIM) {
    this.conv1 = tf.layers.conv2d({
      filters: c_out,
      kernelSize: 3,
      padding: 'same',
      useBias: false
    });
    this.bn1 = tf.layers.batchNormalization();
    
    this.conv2 = tf.layers.conv2d({
      filters: c_out,
      kernelSize: 3,
      padding: 'same',
      useBias: false
    });
    this.bn2 = tf.layers.batchNormalization();

    this.label_proj = tf.layers.dense({ units: c_out * 2 });

    if (c_in !== c_out) {
      this.shortcut = tf.layers.conv2d({ filters: c_out, kernelSize: 1, padding: 'same' });
    } else {
      this.shortcut = null;
    }
  }

  apply(x, label_emb = null) {
    let identity = x;
    if (this.shortcut) {
      identity = this.shortcut.apply(x);
    }

    let out = this.conv1.apply(x);
    out = this.bn1.apply(out);

    if (label_emb) {
      const cond = this.label_proj.apply(label_emb);
      const cond_reshaped = tf.reshape(cond, [-1, 1, 1, cond.shape[1]]);
      const scale = tf.slice(cond_reshaped, [0, 0, 0, 0], [-1, -1, -1, out.shape[3]]);
      const shift = tf.slice(cond_reshaped, [0, 0, 0, out.shape[3]], [-1, -1, -1, -1]);
      
      out = tf.add(tf.mul(out, tf.add(1, scale)), shift);
    }

    out = tf.leakyRelu(out, 0.2);
    out = this.conv2.apply(out);
    out = this.bn2.apply(out);

    return tf.add(out, identity);
  }
}

/**
 * Label Conditioned VAE Model
 */
export class LabelConditionedVAE {
  constructor() {
    // Model architecture using Layers API
    this.label_emb = tf.layers.embedding({
      inputDim: CONFIG.NUM_CLASSES,
      outputDim: CONFIG.LABEL_EMB_DIM
    });

    this.enc_in = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same' });
    this.enc_block1 = new LabelConditionedBlock(64, 128);
    this.enc_block2 = new LabelConditionedBlock(128, 256);
    
    this.z_mean = tf.layers.conv2d({ filters: CONFIG.LATENT_CHANNELS, kernelSize: 1 });
    this.z_logvar = tf.layers.conv2d({ filters: CONFIG.LATENT_CHANNELS, kernelSize: 1 });

    this.dec_in = tf.layers.conv2d({ filters: 256, kernelSize: 1 });
    this.dec_block1 = new LabelConditionedBlock(256, 128);
    this.dec_block2 = new LabelConditionedBlock(128, 64);
    this.dec_out = tf.layers.conv2d({ filters: 3, kernelSize: 3, padding: 'same', activation: 'tanh' });

    // Build layers with dummy inputs to initialize weights
    this.initializeWeights();
  }

  initializeWeights() {
    tf.tidy(() => {
      const dummyX = tf.zeros([1, 64, 64, 3]);
      const dummyLabels = tf.zeros([1], 'int32');
      const dummyZ = tf.zeros([1, 8, 8, CONFIG.LATENT_CHANNELS]);
      
      this.forward(dummyX, dummyLabels);
      this.decode(dummyZ, dummyLabels);
    });
  }

  getWeights() {
    const weights = [];
    const layers = [
      this.label_emb, this.enc_in, 
      this.enc_block1.conv1, this.enc_block1.bn1, this.enc_block1.conv2, this.enc_block1.bn2, this.enc_block1.label_proj,
      this.enc_block2.conv1, this.enc_block2.bn1, this.enc_block2.conv2, this.enc_block2.bn2, this.enc_block2.label_proj,
      this.z_mean, this.z_logvar, this.dec_in,
      this.dec_block1.conv1, this.dec_block1.bn1, this.dec_block1.conv2, this.dec_block1.bn2, this.dec_block1.label_proj,
      this.dec_block2.conv1, this.dec_block2.bn1, this.dec_block2.conv2, this.dec_block2.bn2, this.dec_block2.label_proj,
      this.dec_out
    ];
    
    for (const layer of layers) {
      if (layer && layer.weights) {
        weights.push(...layer.weights.map(w => w.val));
      }
    }
    return weights;
  }

  encode(x, labels) {
    const l_emb = this.label_emb.apply(labels);
    let h = this.enc_in.apply(x);
    h = this.enc_block1.apply(h, l_emb);
    h = this.enc_block2.apply(h, l_emb);
    return [this.z_mean.apply(h), this.z_logvar.apply(h)];
  }

  reparameterize(mu, logvar) {
    const std = tf.exp(tf.mul(0.5, logvar));
    const eps = tf.randomNormal(mu.shape);
    return tf.add(mu, tf.mul(eps, std));
  }

  decode(z, labels) {
    const l_emb = this.label_emb.apply(labels);
    let h = this.dec_in.apply(z);
    h = this.dec_block1.apply(h, l_emb);
    h = this.dec_block2.apply(h, l_emb);
    return this.dec_out.apply(h);
  }

  forward(x, labels) {
    const [mu, logvar] = this.encode(x, labels);
    const z = this.reparameterize(mu, logvar);
    return [this.decode(z, labels), mu, logvar];
  }
}

/**
 * Label Conditioned Drift Network
 */
export class LabelConditionedDrift {
  constructor() {
    this.time_mlp_dense1 = tf.layers.dense({ units: 128, activation: 'relu' });
    this.time_mlp_dense2 = tf.layers.dense({ units: 256, activation: 'relu' });

    this.label_emb = tf.layers.embedding({
      inputDim: CONFIG.NUM_CLASSES,
      outputDim: CONFIG.LABEL_EMB_DIM
    });

    this.head = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same' });
    this.down1 = new LabelConditionedBlock(64, 128);
    this.down2 = tf.layers.conv2d({ filters: 256, kernelSize: 4, strides: 2, padding: 'same' });
    this.mid = new LabelConditionedBlock(256, 256);
    this.up_conv = tf.layers.conv2dTranspose({ filters: 128, kernelSize: 4, strides: 2, padding: 'same' });
    this.up_block = new LabelConditionedBlock(128, 64);
    this.tail = tf.layers.conv2d({ filters: CONFIG.LATENT_CHANNELS, kernelSize: 3, padding: 'same' });

    // Initialize weights
    this.initializeWeights();
  }

  initializeWeights() {
    tf.tidy(() => {
      const dummyZ = tf.zeros([1, 8, 8, CONFIG.LATENT_CHANNELS]);
      const dummyT = tf.zeros([1, 1]);
      const dummyLabels = tf.zeros([1], 'int32');
      this.forward(dummyZ, dummyT, dummyLabels);
    });
  }

  getWeights() {
    const weights = [];
    const layers = [
      this.time_mlp_dense1, this.time_mlp_dense2, this.label_emb, this.head,
      this.down1.conv1, this.down1.bn1, this.down1.conv2, this.down1.bn2, this.down1.label_proj,
      this.down2,
      this.mid.conv1, this.mid.bn1, this.mid.conv2, this.mid.bn2, this.mid.label_proj,
      this.up_conv,
      this.up_block.conv1, this.up_block.bn1, this.up_block.conv2, this.up_block.bn2, this.up_block.label_proj,
      this.tail
    ];
    
    for (const layer of layers) {
      if (layer && layer.weights) {
        weights.push(...layer.weights.map(w => w.val));
      }
    }
    return weights;
  }

  forward(z, t, labels) {
    const t_emb = this.time_mlp_dense2.apply(this.time_mlp_dense1.apply(t));
    const l_emb = this.label_emb.apply(labels);
    const cond = tf.concat([t_emb, l_emb], -1);

    let h = this.head.apply(z);
    const d1 = this.down1.apply(h, cond);
    let d2 = this.down2.apply(d1);
    let m = this.mid.apply(d2, cond);
    let u = this.up_conv.apply(m);
    u = this.up_block.apply(u, cond);
    return this.tail.apply(u);
  }
}
