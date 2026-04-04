// Torch-js implementation of Schrödinger Bridge models
// Based on ../enhancedoptimaltransport/models.py

import { CONFIG } from "../config.js";

// Check if torch is available
let torch;
try {
  torch = await import("js-pytorch");
} catch (error) {
  console.warn("js-pytorch not available, using mock implementation");
  torch = null;
}

// Helper function to create layers
function createLayer(type, ...args) {
  if (!torch) {
    return { type, args };
  }

  switch (type) {
    case "Linear":
      return new torch.nn.Linear(...args);
    case "Conv2d":
      return new torch.nn.Conv2d(...args);
    case "ConvTranspose2d":
      return new torch.nn.ConvTranspose2d(...args);
    case "BatchNorm2d":
      return new torch.nn.BatchNorm2d(...args);
    case "SiLU":
      return new torch.nn.SiLU();
    case "ReLU":
      return new torch.nn.ReLU();
    case "Embedding":
      return new torch.nn.Embedding(...args);
    case "Sequential":
      return new torch.nn.Sequential(...args);
    default:
      throw new Error(`Unknown layer type: ${type}`);
  }
}

// Residual Block
class ResidualBlock {
  constructor(in_channels, out_channels, stride = 1) {
    this.conv1 = createLayer("Conv2d", in_channels, out_channels, 3, stride, 1);
    this.bn1 = createLayer("BatchNorm2d", out_channels);
    this.act1 = createLayer("SiLU");
    this.conv2 = createLayer("Conv2d", out_channels, out_channels, 3, 1, 1);
    this.bn2 = createLayer("BatchNorm2d", out_channels);

    if (in_channels !== out_channels || stride !== 1) {
      this.shortcut = createLayer(
        "Conv2d",
        in_channels,
        out_channels,
        1,
        stride,
        0,
      );
    } else {
      this.shortcut = null;
    }
  }

  forward(x) {
    let identity = x;
    if (this.shortcut) {
      identity = this.shortcut.forward(x);
    }

    let out = this.conv1.forward(x);
    out = this.bn1.forward(out);
    out = this.act1.forward(out);
    out = this.conv2.forward(out);
    out = this.bn2.forward(out);

    return out.add(identity);
  }
}

// Self Attention
class SelfAttention {
  constructor(in_channels) {
    this.query = createLayer(
      "Conv2d",
      in_channels,
      Math.floor(in_channels / 8),
      1,
    );
    this.key = createLayer(
      "Conv2d",
      in_channels,
      Math.floor(in_channels / 8),
      1,
    );
    this.value = createLayer("Conv2d", in_channels, in_channels, 1);
    this.gamma = torch ? new torch.nn.Parameter(torch.zeros(1)) : { data: 0 };
  }

  forward(x) {
    const batch_size = x.shape[0];
    const channels = x.shape[1];
    const height = x.shape[2];
    const width = x.shape[3];

    // Flatten spatial dimensions
    const flat_x = x.reshape([batch_size, channels, height * width]);

    // Compute queries, keys, values
    const q = this.query
      .forward(x)
      .reshape([batch_size, Math.floor(channels / 8), height * width]);
    const k = this.key
      .forward(x)
      .reshape([batch_size, Math.floor(channels / 8), height * width]);
    const v = this.value
      .forward(x)
      .reshape([batch_size, channels, height * width]);

    // Attention
    const attn = q
      .transpose(1, 2)
      .matmul(k)
      .div(Math.sqrt(Math.floor(channels / 8)));
    const attn_softmax = attn.softmax(2);

    // Apply attention to values
    const out = v.matmul(attn_softmax.transpose(1, 2));
    const out_reshaped = out.reshape([batch_size, channels, height, width]);

    return x.add(out_reshaped.mul(this.gamma));
  }
}

// Label Conditioned Block
class LabelConditionedBlock {
  constructor(
    c_in,
    c_out,
    label_dim = CONFIG.LABEL_EMB_DIM,
    use_spectral_norm = false,
  ) {
    this.conv1 = createLayer("Conv2d", c_in, c_out, 3, 1, 1);
    this.bn1 = createLayer("BatchNorm2d", c_out);
    this.act1 = createLayer("SiLU");
    this.conv2 = createLayer("Conv2d", c_out, c_out, 3, 1, 1);
    this.bn2 = createLayer("BatchNorm2d", c_out);

    // Label conditioning
    this.label_proj = createLayer("Linear", label_dim, c_out * 2);

    if (c_in !== c_out) {
      this.shortcut = createLayer("Conv2d", c_in, c_out, 1, 1, 0);
    } else {
      this.shortcut = null;
    }
  }

  forward(x, labels = null) {
    let identity = x;
    if (this.shortcut) {
      identity = this.shortcut.forward(x);
    }

    let out = this.conv1.forward(x);
    out = this.bn1.forward(out);

    // Apply label conditioning if provided
    if (labels !== null) {
      const label_emb = this.label_proj.forward(labels);
      const scale = label_emb.slice([0, 0, out.shape[2]]);
      const shift = label_emb.slice([0, out.shape[2], out.shape[2] * 2]);
      out = out
        .mul(scale.reshape([-1, out.shape[1], 1, 1]))
        .add(shift.reshape([-1, out.shape[1], 1, 1]));
    }

    out = this.act1.forward(out);
    out = this.conv2.forward(out);
    out = this.bn2.forward(out);

    return out.add(identity);
  }
}

// Label Conditioned VAE
export class LabelConditionedVAE {
  constructor(free_bits = CONFIG.FREE_BITS) {
    this.free_bits = free_bits;

    // Label embedding
    this.label_emb = createLayer(
      "Embedding",
      CONFIG.NUM_CLASSES,
      CONFIG.LABEL_EMB_DIM,
    );

    // Context embedding (if enabled)
    if (CONFIG.USE_CONTEXT) {
      this.source_emb = createLayer(
        "Embedding",
        CONFIG.NUM_SOURCES,
        CONFIG.CONTEXT_DIM,
      );
      this.cond_proj = createLayer(
        "Linear",
        CONFIG.LABEL_EMB_DIM + CONFIG.CONTEXT_DIM,
        CONFIG.LABEL_EMB_DIM,
      );
    }

    // Encoder
    this.enc_in = createLayer("Conv2d", 3, 64, 3, 1, 1);
    this.enc_blocks = [
      new LabelConditionedBlock(64, 128),
      new LabelConditionedBlock(128, 256),
      new LabelConditionedBlock(256, 512),
      new LabelConditionedBlock(512, 512),
    ];

    // Latent projection
    this.z_mean = createLayer("Conv2d", 512, CONFIG.LATENT_CHANNELS, 1);
    this.z_logvar = createLayer("Conv2d", 512, CONFIG.LATENT_CHANNELS, 1);

    // Decoder
    this.dec_in = createLayer("Conv2d", CONFIG.LATENT_CHANNELS, 512, 1);
    this.dec_blocks = [
      new LabelConditionedBlock(512, 512),
      new LabelConditionedBlock(512, 256),
      new LabelConditionedBlock(256, 128),
      new LabelConditionedBlock(128, 64),
    ];
    this.dec_out = createLayer("Conv2d", 64, 3, 3, 1, 1);

    // Diversity loss tracking
    this.diversity_loss = null;
    this.mu_noise_scale = CONFIG.MU_NOISE_SCALE;
  }

  _channel_diversity_loss(mu) {
    // Compute channel-wise diversity loss
    const channel_means = mu.mean([0, 2, 3]);
    const channel_stds = mu.std([0, 2, 3]);

    const target_std = CONFIG.DIVERSITY_TARGET_STD;
    const max_std = CONFIG.DIVERSITY_MAX_STD;

    // Penalize channels with too low or too high std
    const low_mask = channel_stds.lt(target_std);
    const high_mask = channel_stds.gt(max_std);

    const low_penalty = channel_stds
      .sub(target_std)
      .abs()
      .mul(CONFIG.DIVERSITY_LOW_PENALTY)
      .mul(low_mask);
    const high_penalty = channel_stds
      .sub(max_std)
      .mul(CONFIG.DIVERSITY_HIGH_PENALTY)
      .mul(high_mask);

    return low_penalty
      .add(high_penalty)
      .mean()
      .mul(CONFIG.DIVERSITY_BALANCE_WEIGHT);
  }

  encode(x, labels, source_id = null) {
    // Label embedding
    let label_emb = this.label_emb.forward(labels);

    // Add context if enabled
    if (CONFIG.USE_CONTEXT && source_id !== null) {
      const s_emb = this.source_emb.forward(source_id);
      label_emb = this.cond_proj.forward(torch.cat([label_emb, s_emb], -1));
    }

    // Encoder forward pass
    let h = this.enc_in.forward(x);
    for (const block of this.enc_blocks) {
      h = block.forward(h, label_emb);
    }

    // Latent parameters
    let mu = this.z_mean.forward(h).mul(CONFIG.LATENT_SCALE);

    // Add noise during training
    if (this.training && this.mu_noise_scale > 0) {
      mu = mu.add(torch.randn_like(mu).mul(this.mu_noise_scale));
    }

    // Channel dropout
    if (this.training && Math.random() < CONFIG.CHANNEL_DROPOUT_PROB) {
      const channel_mask = torch.bernoulli(
        torch.full(
          [mu.shape[0], mu.shape[1], 1, 1],
          CONFIG.CHANNEL_DROPOUT_SURVIVAL,
        ),
      );
      mu = mu.mul(channel_mask).div(CONFIG.CHANNEL_DROPOUT_SURVIVAL);
    }

    let logvar = this.z_logvar.forward(h);
    logvar = logvar.clamp(CONFIG.LOGVAR_CLAMP_MIN, CONFIG.LOGVAR_CLAMP_MAX);

    // Compute diversity loss during training
    if (this.training) {
      this.diversity_loss = this._channel_diversity_loss(mu);
    }

    return [mu, logvar];
  }

  decode(z, labels, source_id = null) {
    // Label embedding
    let label_emb = this.label_emb.forward(labels);

    // Add context if enabled
    if (CONFIG.USE_CONTEXT && source_id !== null) {
      const s_emb = this.source_emb.forward(source_id);
      label_emb = this.cond_proj.forward(torch.cat([label_emb, s_emb], -1));
    }

    // Decoder forward pass
    let h = this.dec_in.forward(z);
    for (const block of this.dec_blocks) {
      h = block.forward(h, label_emb);
    }
    h = this.dec_out.forward(h);

    // Final upsampling if needed
    if (h.shape[2] !== CONFIG.IMG_SIZE || h.shape[3] !== CONFIG.IMG_SIZE) {
      h = torch.nn.functional.interpolate(
        h,
        [CONFIG.IMG_SIZE, CONFIG.IMG_SIZE],
        "bilinear",
      );
    }

    return h.tanh();
  }

  forward(x, labels, source_id = null) {
    const [mu, logvar] = this.encode(x, labels, source_id);

    // Reparameterization
    let z;
    if (this.training) {
      const std = logvar.mul(0.5).exp();
      z = mu.add(std.mul(torch.randn_like(std)));
    } else {
      z = mu;
    }

    const recon = this.decode(z, labels, source_id);
    return [recon, mu, logvar];
  }

  reparameterize(mu, logvar) {
    if (this.training) {
      const std = logvar.mul(0.5).exp();
      const eps = torch.randn_like(std);
      return mu.add(eps.mul(std));
    } else {
      return mu;
    }
  }
}

// Fourier Time Embedding
export class FourierTimeEmbed {
  constructor(dim = 128, max_freq = 64) {
    this.dim = dim;
    this.freqs = torch.linspace(1, max_freq, Math.floor(dim / 2));
  }

  forward(t) {
    // t: (B, 1) in range [0, 1]
    const t_scaled = t.mul(2 * Math.PI);
    const sin_emb = this.freqs.mul(t_scaled).sin();
    const cos_emb = this.freqs.mul(t_scaled).cos();
    return torch.cat([sin_emb, cos_emb], -1);
  }
}

// Label Conditioned Drift Network
export class LabelConditionedDrift {
  constructor() {
    // Time embedding
    this.time_mlp = createLayer(
      "Sequential",
      new FourierTimeEmbed(128),
      createLayer("Linear", 128, 256),
      createLayer("SiLU"),
      createLayer("Linear", 256, 256),
    );

    // Label conditioning
    this.label_emb = createLayer(
      "Embedding",
      CONFIG.NUM_CLASSES,
      CONFIG.LABEL_EMB_DIM,
    );

    // Context embedding
    if (CONFIG.USE_CONTEXT) {
      this.source_emb = createLayer(
        "Embedding",
        CONFIG.NUM_SOURCES,
        CONFIG.CONTEXT_DIM,
      );
      this.cond_proj = createLayer(
        "Linear",
        256 + CONFIG.LABEL_EMB_DIM + CONFIG.CONTEXT_DIM,
        128,
      );
    } else {
      this.cond_proj = createLayer("Linear", 256 + CONFIG.LABEL_EMB_DIM, 128);
    }

    // Time-adaptive scaling
    this.time_weight_net = createLayer(
      "Sequential",
      createLayer("Linear", 1, 32),
      createLayer("SiLU"),
      createLayer("Linear", 32, 1),
      createLayer("Sigmoid"),
    );

    // U-Net architecture
    this.head = createLayer("Conv2d", CONFIG.LATENT_CHANNELS, 64, 3, 1, 1);
    this.down1 = new LabelConditionedBlock(64, 128, 128, true);
    this.down2_conv = createLayer("Conv2d", 128, 256, 4, 2, 1);
    this.down2_block = new LabelConditionedBlock(256, 256, 128, true);

    this.mid1 = new LabelConditionedBlock(256, 256, 128, true);
    this.mid_attn = new SelfAttention(256);
    this.mid2 = new LabelConditionedBlock(256, 256, 128, true);

    this.up2_conv = createLayer(
      "Sequential",
      createLayer("Upsample", 2, "nearest"),
      createLayer("Conv2d", 256, 128, 3, 1, 1),
    );
    this.up2_block = new LabelConditionedBlock(128, 128, 128, true);
    this.up1 = new LabelConditionedBlock(128, 64, 128, true);

    this.tail = createLayer("Conv2d", 64, CONFIG.LATENT_CHANNELS, 3, 1, 1);

    // Learnable scaling
    this.output_scale = torch
      ? new torch.nn.Parameter(torch.tensor(0.1))
      : { data: 0.1 };
    this.time_scales = torch
      ? new torch.nn.Parameter(torch.ones(4).mul(0.1))
      : { data: [0.1, 0.1, 0.1, 0.1] };

    // Running statistics
    this.drift_mean = torch ? torch.zeros(1) : 0;
    this.drift_std = torch ? torch.ones(1) : 1;
    this.n_samples = torch ? torch.zeros(1) : 0;
    this.momentum = 0.99;
  }

  forward(z, t, labels, cfg_scale = 1.0, source_id = null) {
    // Time embedding
    if (t.dim() === 1) {
      t = t.unsqueeze(-1);
    }
    const t_emb = this.time_mlp.forward(t);

    // Classifier-Free Guidance during inference
    if (cfg_scale !== 1.0 && !this.training) {
      const cond_drift = this._forward_internal(z, t, labels, t_emb, source_id);
      const uncond_labels = torch.zeros_like(labels);
      const uncond_drift = this._forward_internal(
        z,
        t,
        uncond_labels,
        t_emb,
        source_id,
      );

      return uncond_drift.add(cond_drift.sub(uncond_drift).mul(cfg_scale));
    }

    return this._forward_internal(z, t, labels, t_emb, source_id);
  }

  _forward_internal(z, t, labels, t_emb, source_id = null) {
    // Label embedding
    const label_emb = this.label_emb.forward(labels);

    // Combine embeddings with optional context
    let cond;
    if (CONFIG.USE_CONTEXT && source_id !== null) {
      const s_emb = this.source_emb.forward(source_id);
      cond = torch.cat([t_emb, label_emb, s_emb], -1);
    } else {
      cond = torch.cat([t_emb, label_emb], -1);
    }

    cond = this.cond_proj.forward(cond);

    // Time-adaptive scaling
    const time_weight = this.time_weight_net.forward(t);

    // U-Net forward pass
    let h = this.head.forward(z);
    const d1 = this.down1.forward(h, cond);
    let d2 = this.down2_conv.forward(d1);
    d2 = this.down2_block.forward(d2, cond);
    let m = this.mid1.forward(d2, cond);
    m = this.mid_attn.forward(m);
    m = this.mid2.forward(m, cond);
    let u2 = this.up2_conv.forward(m);
    u2 = this.up2_block.forward(u2, cond);

    // Skip connection
    if (u2.shape[2] !== d1.shape[2] || u2.shape[3] !== d1.shape[3]) {
      u2 = torch.nn.functional.interpolate(
        u2,
        [d1.shape[2], d1.shape[3]],
        "nearest",
      );
    }

    const u1 = this.up1.forward(u2.add(d1), cond);
    let out = this.tail.forward(u1);

    // Scale output
    out = out
      .mul(this.output_scale)
      .mul(time_weight.reshape([-1, 1, 1, 1]).add(1));

    return out;
  }
}

// Export all models
export default {
  LabelConditionedVAE,
  LabelConditionedDrift,
  ResidualBlock,
  SelfAttention,
  LabelConditionedBlock,
  FourierTimeEmbed,
};
