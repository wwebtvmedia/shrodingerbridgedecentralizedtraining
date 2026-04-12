// Enhanced Schrödinger Bridge Model Architectures
// Ported from PyTorch (enhancedoptimaltransport/models.py) to TensorFlow.js
// Optimized for GPU acceleration and high-fidelity generation (96x96)

import * as tf from '@tensorflow/tfjs';
import { CONFIG } from "../config.js";
import { wrapWithLoRA } from "./lora.js";

/**
 * Residual Block with optional stride and bottleneck
 */
class ResidualBlock {
  constructor(inChannels, outChannels, stride = 1, name) {
    this.name = name;
    this.stride = stride;
    this.inChannels = inChannels;
    this.outChannels = outChannels;

    this.conv1 = wrapWithLoRA(tf.layers.conv2d({
      filters: outChannels,
      kernelSize: 3,
      strides: stride,
      padding: 'same',
      useBias: false,
      name: `${name}/conv1`
    }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
    this.bn1 = tf.layers.batchNormalization({ name: `${name}/bn1` });
    
    this.conv2 = wrapWithLoRA(tf.layers.conv2d({
      filters: outChannels,
      kernelSize: 3,
      strides: 1,
      padding: 'same',
      useBias: false,
      name: `${name}/conv2`
    }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
    this.bn2 = tf.layers.batchNormalization({ name: `${name}/bn2` });

    if (stride !== 1 || inChannels !== outChannels) {
      this.shortcut = tf.sequential({
        layers: [
          tf.layers.conv2d({
            filters: outChannels,
            kernelSize: 1,
            strides: stride,
            useBias: false,
            inputShape: [null, null, inChannels], // [H, W, C]
            name: `${name}/shortcut/conv`
          }),
          tf.layers.batchNormalization({ name: `${name}/shortcut/bn` })
        ]
      });
    } else {
      this.shortcut = null;
    }
  }

  forward(x) {
    return tf.tidy(() => {
      let out = this.conv1.apply(x);
      out = this.bn1.apply(out);
      out = tf.leakyRelu(out, 0.1); // Using LeakyReLU as SiLU alternative if needed

      out = this.conv2.apply(out);
      out = this.bn2.apply(out);

      let shortcut = this.shortcut ? this.shortcut.apply(x) : x;
      
      // Ensure shapes match (interpolation in PyTorch)
      if (out.shape[1] !== shortcut.shape[1] || out.shape[2] !== shortcut.shape[2]) {
        shortcut = tf.image.resizeNearestNeighbor(shortcut, [out.shape[1], out.shape[2]]);
      }

      return tf.leakyRelu(tf.add(out, shortcut), 0.1);
    });
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
    
    // In TFJS we can use dense layers for attention projections
    this.qkv_h = tf.layers.dense({ units: channels * 3, name: `${name}/qkv_h` });
    this.qkv_w = tf.layers.dense({ units: channels * 3, name: `${name}/qkv_w` });
    this.proj_h = tf.layers.dense({ units: channels, name: `${name}/proj_h` });
    this.proj_w = tf.layers.dense({ units: channels, name: `${name}/proj_w` });
  }

  forward(x) {
    return tf.tidy(() => {
      const [b, h, w, c] = x.shape;
      
      // 1. Vertical Attention (along H)
      // [B, H, W, C] -> [B, W, H, C] -> [B*W, H, C]
      let v = tf.transpose(x, [0, 2, 1, 3]);
      v = tf.reshape(v, [b * w, h, c]);
      
      let qkv_v = this.qkv_h.apply(v);
      let [q_v, k_v, v_v] = tf.split(qkv_v, 3, -1);
      
      // Simple self-attention
      let attn_v = tf.matMul(q_v, k_v, false, true);
      attn_v = tf.softmax(tf.div(attn_v, tf.sqrt(c / this.numHeads)));
      let out_v = tf.matMul(attn_v, v_v);
      out_v = this.proj_h.apply(out_v);
      
      // [B*W, H, C] -> [B, W, H, C] -> [B, H, W, C]
      out_v = tf.reshape(out_v, [b, w, h, c]);
      out_v = tf.transpose(out_v, [0, 2, 1, 3]);
      let x_res = tf.add(x, out_v);

      // 2. Horizontal Attention (along W)
      // [B, H, W, C] -> [B*H, W, C]
      let h_in = tf.reshape(x_res, [b * h, w, c]);
      
      let qkv_h = this.qkv_w.apply(h_in);
      let [q_h, k_h, v_h] = tf.split(qkv_h, 3, -1);
      
      let attn_h = tf.matMul(q_h, k_h, false, true);
      attn_h = tf.softmax(tf.div(attn_h, tf.sqrt(c / this.numHeads)));
      let out_h = tf.matMul(attn_h, v_h);
      out_h = this.proj_w.apply(out_h);
      
      out_h = tf.reshape(out_h, [b, h, w, c]);
      return tf.add(x_res, out_h);
    });
  }
}

/**
 * Label Conditioned Block with FiLM-like modulation
 */
class LabelConditionedBlock {
  constructor(cIn, cOut, labelDim = 128, name) {
    this.name = name;
    this.conv1 = wrapWithLoRA(tf.layers.conv2d({ filters: cOut, kernelSize: 3, padding: 'same', name: `${name}/conv1` }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
    this.conv2 = wrapWithLoRA(tf.layers.conv2d({ filters: cOut, kernelSize: 3, padding: 'same', name: `${name}/conv2` }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
    
    this.labelProj = wrapWithLoRA(tf.layers.dense({ units: cOut * 2, name: `${name}/label_proj` }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
    
    if (cIn !== cOut) {
      this.skip = wrapWithLoRA(tf.layers.conv2d({ filters: cOut, kernelSize: 1, name: `${name}/skip` }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
    } else {
      this.skip = null;
    }
  }

  forward(x, labels) {
    return tf.tidy(() => {
      let h = tf.leakyRelu(this.conv1.apply(x), 0.1);
      
      if (labels) {
        const scaleShift = this.labelProj.apply(labels);
        const [scale, shift] = tf.split(scaleShift, 2, -1);
        
        // Reshape for broadcasting [B, 1, 1, C]
        const scaleReshaped = tf.reshape(scale, [scale.shape[0], 1, 1, scale.shape[1]]);
        const shiftReshaped = tf.reshape(shift, [shift.shape[0], 1, 1, shift.shape[1]]);
        
        // Use local variable to avoid multiple references
        const modulated = tf.add(tf.mul(h, tf.add(1, scaleReshaped)), shiftReshaped);
        h = modulated;
      }
      
      h = tf.leakyRelu(this.conv2.apply(h), 0.1);
      let skip = this.skip ? this.skip.apply(x) : x;
      
      // Ensure shapes match
      if (h.shape[1] !== skip.shape[1] || h.shape[2] !== skip.shape[2]) {
        skip = tf.image.resizeNearestNeighbor(skip, [h.shape[1], h.shape[2]]);
      }
      
      return tf.add(h, skip);
    });
  }
}

/**
 * Subpixel Upsampling (PixelShuffle)
 */
class SubpixelUpsample {
  constructor(inChannels, outChannels, upscaleFactor = 2, name) {
    this.name = name;
    this.upscaleFactor = upscaleFactor;
    this.upsample = tf.layers.upSampling2d({
      size: [upscaleFactor, upscaleFactor],
      name: `${name}/upsample`
    });
    this.conv = wrapWithLoRA(tf.layers.conv2d({
      filters: outChannels,
      kernelSize: 3,
      padding: 'same',
      name: `${name}/conv`
    }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
  }

  forward(x) {
    return tf.tidy(() => {
      let h = this.upsample.apply(x);
      return this.conv.apply(h);
    });
  }
}

/**
 * Label Conditioned VAE
 */
export class LabelConditionedVAE {
  constructor() {
    this.latentChannels = CONFIG.LATENT_CHANNELS || 8;
    this.labelDim = CONFIG.LABEL_EMB_DIM || 128;
    
    // Layers
    this.labelEmb = tf.layers.embedding({
      inputDim: CONFIG.NUM_CLASSES || 11,
      outputDim: this.labelDim,
      name: 'vae/label_emb'
    });
    
    // Encoder
    this.encIn = wrapWithLoRA(tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: 'same', name: 'vae/enc_in' }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
    this.encBlocks = [
      new ResidualBlock(16, 32, 2, 'vae/enc_block0'),           // 96 -> 48
      new LabelConditionedBlock(32, 32, this.labelDim, 'vae/enc_cond0'),
      new ResidualBlock(32, 64, 2, 'vae/enc_block1'),          // 48 -> 24
      new SpatialSplitAttention(64, 4, 'vae/enc_attn0'),
      new LabelConditionedBlock(64, 64, this.labelDim, 'vae/enc_cond1'),
      new ResidualBlock(64, 64, 2, 'vae/enc_block2'),          // 24 -> 12
    ];
    
    this.zMean = wrapWithLoRA(tf.layers.conv2d({ filters: this.latentChannels, kernelSize: 3, padding: 'same', name: 'vae/z_mean' }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
    this.zLogvar = wrapWithLoRA(tf.layers.conv2d({ filters: this.latentChannels, kernelSize: 3, padding: 'same', name: 'vae/z_logvar' }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);

    // Decoder
    this.decIn = wrapWithLoRA(tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: 'same', name: 'vae/dec_in' }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
    this.decBlocks = [
      new SubpixelUpsample(64, 64, 2, 'vae/dec_up0'),          // 12 -> 24
      new LabelConditionedBlock(64, 64, this.labelDim, 'vae/dec_cond0'),
      new SpatialSplitAttention(64, 4, 'vae/dec_attn0'),
      
      new SubpixelUpsample(64, 32, 2, 'vae/dec_up1'),          // 24 -> 48
      new LabelConditionedBlock(32, 32, this.labelDim, 'vae/dec_cond1'),
      new SpatialSplitAttention(32, 4, 'vae/dec_attn1'),
      
      new SubpixelUpsample(32, 16, 2, 'vae/dec_up2'),           // 48 -> 96
      new LabelConditionedBlock(16, 16, this.labelDim, 'vae/dec_cond2'),
    ];
    
    this.decOut = wrapWithLoRA(tf.layers.conv2d({ filters: 3, kernelSize: 3, padding: 'same', name: 'vae/dec_out' }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
  }

  getConditioning(labels) {
    return tf.tidy(() => {
      let emb = this.labelEmb.apply(labels);
      // If labels is [B, 1], result is [B, 1, D], squeeze to [B, D]
      if (emb.shape.length === 3) {
        emb = tf.squeeze(emb, [1]);
      }
      return emb;
    });
  }

  encode(x, labels) {
    return tf.tidy(() => {
      const cond = this.getConditioning(labels);
      let h = this.encIn.apply(x);
      
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

  decode(z, labels) {
    return tf.tidy(() => {
      const cond = this.getConditioning(labels);
      let h = this.decIn.apply(z);
      
      for (const block of this.decBlocks) {
        if (block instanceof LabelConditionedBlock) {
          h = block.forward(h, cond);
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

  forward(x, labels) {
    const [mu, logvar] = this.encode(x, labels);
    const z = this.reparameterize(mu, logvar);
    const recon = this.decode(z, labels);
    return [recon, mu, logvar];
  }
}

/**
 * Fourier Time Embedding
 */
class FourierTimeEmbed {
  constructor(dim = 128, maxFreq = 64) {
    this.dim = dim;
    this.maxFreq = maxFreq;
    // Precompute frequencies
    const freqs = tf.linspace(1, maxFreq, Math.floor(dim / 2));
    this.freqs = tf.variable(freqs, false);
  }

  forward(t) {
    return tf.tidy(() => {
      // t is [B, 1]
      const args = tf.mul(t, tf.mul(2 * Math.PI, this.freqs));
      const sin = tf.sin(args);
      const cos = tf.cos(args);
      return tf.concat([sin, cos], -1);
    });
  }
}

/**
 * Label Conditioned Drift Network (U-Net)
 */
export class LabelConditionedDrift {
  constructor() {
    this.latentChannels = CONFIG.LATENT_CHANNELS || 8;
    this.labelDim = CONFIG.LABEL_EMB_DIM || 128;
    
    this.timeEmbed = new FourierTimeEmbed(128);
    this.timeMlp = tf.sequential({
      layers: [
        tf.layers.dense({ units: 256, activation: 'relu', inputShape: [128], name: 'drift/time_mlp1' }),
        tf.layers.dense({ units: 256, name: 'drift/time_mlp2' })
      ]
    });
    
    this.labelEmb = tf.layers.embedding({
      inputDim: CONFIG.NUM_CLASSES || 11,
      outputDim: this.labelDim,
      name: 'drift/label_emb'
    });
    
    this.condProj = wrapWithLoRA(tf.layers.dense({ units: 256, name: 'drift/cond_proj' }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
    
    // U-Net Structure
    this.head = wrapWithLoRA(tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', name: 'drift/head' }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
    
    this.down1 = new LabelConditionedBlock(32, 64, 256, 'drift/down1');
    this.down2Conv = wrapWithLoRA(tf.layers.conv2d({ filters: 64, kernelSize: 4, strides: 2, padding: 'same', name: 'drift/down2_conv' }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
    this.down2Block = new LabelConditionedBlock(64, 64, 256, 'drift/down2_block');
    
    this.mid1 = new LabelConditionedBlock(64, 64, 256, 'drift/mid1');
    this.midAttn = new SpatialSplitAttention(64, 4, 'drift/mid_attn');
    this.mid2 = new LabelConditionedBlock(64, 64, 256, 'drift/mid2');
    
    this.up2Conv = wrapWithLoRA(tf.layers.conv2dTranspose({ filters: 64, kernelSize: 4, strides: 2, padding: 'same', name: 'drift/up2_conv' }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
    this.up2Block = new LabelConditionedBlock(64, 64, 256, 'drift/up2_block');
    
    this.up1 = new LabelConditionedBlock(64, 32, 256, 'drift/up1');
    this.tail = wrapWithLoRA(tf.layers.conv2d({ filters: this.latentChannels, kernelSize: 3, padding: 'same', name: 'drift/tail' }), CONFIG.LORA_R, CONFIG.LORA_ALPHA);
  }

  forward(z, t, labels) {
    return tf.tidy(() => {
      // Time and Label Conditioning
      const t_f = this.timeEmbed.forward(t);
      const tEmb = this.timeMlp.apply(t_f);
      let lEmb = this.labelEmb.apply(labels);
      if (lEmb.shape.length === 3) lEmb = tf.squeeze(lEmb, [1]);
      
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
      // Ensure u2 shape matches d1 for skip connection
      if (u2.shape[1] !== d1.shape[1] || u2.shape[2] !== d1.shape[2]) {
        u2 = tf.image.resizeNearestNeighbor(u2, [d1.shape[1], d1.shape[2]]);
      }
      
      const u2_input = tf.add(u2, d1);
      const u2_b = this.up2Block.forward(u2_input, cond);
      
      const u1 = this.up1.forward(u2_b, cond);
      return this.tail.apply(u1);
    });
  }
}
