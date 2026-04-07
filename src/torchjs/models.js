// PyTorch-like implementation of Schrödinger Bridge models using js-pytorch
// MLP-Mixer architecture for improved spatial quality in js-pytorch 0.7.2
// Hardware accelerated (GPU/WebGL)

import { CONFIG } from "../config.js";

/**
 * Robust torch initialization
 * Handles browser global, ESM exports, and potential initialization delays.
 */
async function resolveTorch() {
  if (typeof window !== 'undefined' && window.torch) return window.torch;
  
  try {
    const JSTorch = await import('js-pytorch');
    return JSTorch.torch || (JSTorch.default && JSTorch.default.torch) || JSTorch;
  } catch (e) {
    // If dynamic import fails, try waiting for global if in browser
    if (typeof window !== 'undefined') {
      return new Promise((resolve, reject) => {
        let attempts = 0;
        const interval = setInterval(() => {
          if (window.torch) {
            clearInterval(interval);
            resolve(window.torch);
          }
          if (attempts++ > 50) {
            clearInterval(interval);
            reject(new Error("Timeout waiting for torch to initialize"));
          }
        }, 100);
      });
    }
    throw e;
  }
}

// Top-level await to ensure torch is ready for class definitions
const torch = await resolveTorch();

// Verify torch.nn
if (!torch || !torch.nn) {
  console.error("Torch or Torch.nn is undefined:", torch);
  throw new Error("js-pytorch initialization failed: nn submodule not found");
}

// Global activation instances
const relu_module = new torch.nn.ReLU();
const relu = (x) => relu_module.forward(x);

/**
 * Mixer Block: Performs spatial (token) and feature (channel) mixing.
 */
class MixerBlock extends torch.nn.Module {
  constructor(n_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, device = "gpu") {
    super();
    this.ln1 = new torch.nn.LayerNorm(hidden_dim, device);
    this.ln2 = new torch.nn.LayerNorm(hidden_dim, device);

    this.token_fc1 = new torch.nn.Linear(n_patches, tokens_mlp_dim, device);
    this.token_fc2 = new torch.nn.Linear(tokens_mlp_dim, n_patches, device);

    this.channel_fc1 = new torch.nn.Linear(hidden_dim, channels_mlp_dim, device);
    this.channel_fc2 = new torch.nn.Linear(channels_mlp_dim, hidden_dim, device);
  }

  forward(x) {
    let h = this.ln1.forward(x);
    h = h.transpose(1, 2); 
    h = relu(this.token_fc1.forward(h));
    h = this.token_fc2.forward(h);
    h = h.transpose(1, 2); 
    let out = x.add(h);

    h = this.ln2.forward(out);
    h = relu(this.channel_fc1.forward(h));
    h = this.channel_fc2.forward(h);
    
    return out.add(h);
  }
}

/**
 * Label-Conditioned Mixer Block
 */
class ConditionedMixer extends torch.nn.Module {
  constructor(patches, dim, label_dim = 128, device = "gpu") {
    super();
    this.patches = patches;
    this.dim = dim;
    this.mixer = new MixerBlock(patches, dim, Math.floor(dim / 2), dim * 2, device);
    
    this.label_scale = new torch.nn.Linear(label_dim, patches * dim, device);
    this.label_shift = new torch.nn.Linear(label_dim, patches * dim, device);
  }

  forward(x, label_emb) {
    let out = this.mixer.forward(x);
    
    if (label_emb) {
      const B = x.shape[0];
      const scale = this.label_scale.forward(label_emb).reshape([B, this.patches, this.dim]);
      const shift = this.label_shift.forward(label_emb).reshape([B, this.patches, this.dim]);
      out = out.mul(scale).add(shift);
    }
    
    return out;
  }
}

/**
 * Label Conditioned VAE Model (MLP-Mixer Version)
 */
export class LabelConditionedVAE extends torch.nn.Module {
  constructor(device = "gpu") {
    super();
    const patch_size = 4;
    const n_patches = 64; 
    const hidden_dim = 64;
    const latent_dim = 64;
    const label_dim = 128;
    const input_dim = 3 * 32 * 32; 

    this.n_patches = n_patches;
    this.hidden_dim = hidden_dim;
    this.latent_dim = latent_dim;

    this.label_emb = new torch.nn.Embedding(CONFIG.NUM_CLASSES || 10, label_dim, device);

    this.patch_proj = new torch.nn.Linear(48, hidden_dim, device); 
    this.enc_mixer = new ConditionedMixer(n_patches, hidden_dim, label_dim, device);
    this.z_mean = new torch.nn.Linear(n_patches * hidden_dim, latent_dim, device);
    this.z_logvar = new torch.nn.Linear(n_patches * hidden_dim, latent_dim, device);

    this.z_to_patches = new torch.nn.Linear(latent_dim, n_patches * hidden_dim, device);
    this.dec_mixer = new ConditionedMixer(n_patches, hidden_dim, label_dim, device);
    this.recon_proj = new torch.nn.Linear(hidden_dim, 48, device);
  }

  encode(x, labels) {
    const B = x.shape[0];
    const l_emb_raw = this.label_emb.forward(labels);
    const l_emb = l_emb_raw.reshape([B, l_emb_raw.shape[2]]);
    
    let h = x.reshape([B, 64, 48]);
    h = this.patch_proj.forward(h);
    h = this.enc_mixer.forward(h, l_emb);
    
    h = h.reshape([B, 64 * 64]);
    return [this.z_mean.forward(h), this.z_logvar.forward(h)];
  }

  reparameterize(mu, logvar) {
    const std = torch.exp(logvar.mul(0.5));
    const eps = torch.randn(mu.shape, false, mu.device);
    return mu.add(eps.mul(std));
  }

  decode(z, labels) {
    const B = z.shape[0];
    const l_emb_raw = this.label_emb.forward(labels);
    const l_emb = l_emb_raw.reshape([B, l_emb_raw.shape[2]]);
    
    let h = this.z_to_patches.forward(z).reshape([B, 64, 64]);
    h = this.dec_mixer.forward(h, l_emb);
    h = this.recon_proj.forward(h);
    
    return relu(h.reshape([B, 3072]));
  }

  forward(x, labels) {
    const [mu, logvar] = this.encode(x, labels);
    const z = this.reparameterize(mu, logvar);
    return [this.decode(z, labels), mu, logvar];
  }
}

/**
 * Label Conditioned Drift Network (MLP-Mixer Version)
 */
export class LabelConditionedDrift extends torch.nn.Module {
  constructor(device = "gpu") {
    super();
    const latent_dim = 64;
    const label_dim = 128;
    const hidden_dim = 32;
    const n_tokens = 4; 

    this.n_tokens = n_tokens;
    this.hidden_dim = hidden_dim;

    this.time_fc1 = new torch.nn.Linear(1, 64, device);
    this.time_fc2 = new torch.nn.Linear(64, label_dim, device);

    this.label_emb = new torch.nn.Embedding(CONFIG.NUM_CLASSES || 10, label_dim, device);

    this.head = new torch.nn.Linear(latent_dim, 4 * 32, device);
    this.mixer = new ConditionedMixer(4, 32, label_dim, device);
    this.tail = new torch.nn.Linear(4 * 32, latent_dim, device);
  }

  forward(z, t, labels) {
    const B = z.shape[0];
    const t_emb = relu(this.time_fc2.forward(relu(this.time_fc1.forward(t))));
    const l_emb_raw = this.label_emb.forward(labels);
    const l_emb = l_emb_raw.reshape([B, l_emb_raw.shape[2]]);
    const cond = t_emb.add(l_emb);

    let h = this.head.forward(z).reshape([B, 4, 32]);
    h = this.mixer.forward(h, cond);
    h = h.reshape([B, 128]);
    return this.tail.forward(h);
  }
}
