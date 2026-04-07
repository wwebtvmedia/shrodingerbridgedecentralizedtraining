import {
  CONFIG,
} from "../config.js";
import { LabelConditionedVAE, LabelConditionedDrift } from "./models.js";

/**
 * Robust torch initialization
 */
let torch;
if (typeof window !== 'undefined' && window.torch) {
  torch = window.torch;
} else {
  try {
    const JSTorch = await import('js-pytorch');
    torch = JSTorch.torch || (JSTorch.default && JSTorch.default.torch) || JSTorch;
  } catch (e) {
    if (typeof globalThis !== 'undefined' && globalThis.torch) {
      torch = globalThis.torch;
    }
  }
}

// OU Reference Process
class OUReference {
  constructor(theta = 1.0, sigma = Math.sqrt(2)) {
    this.theta = theta;
    this.sigma = sigma;
  }

  bridgeSample(z0, z1, t) {
    const device = z0.device;
    // Vectorized version using torch
    const exp_neg_theta_t = torch.exp(t.mul(-this.theta));
    const ones_t = torch.ones(t.shape, false, device);
    const exp_neg_theta_1_t = torch.exp(ones_t.sub(t).mul(-this.theta));
    const exp_neg_theta = Math.exp(-this.theta);

    const denominator = 1 - exp_neg_theta ** 2;
    
    const term1 = exp_neg_theta_t.mul(torch.ones(exp_neg_theta_1_t.shape, false, device).sub(exp_neg_theta_1_t.pow(2)));
    const term2 = torch.ones(exp_neg_theta_t.shape, false, device).sub(exp_neg_theta_t.pow(2)).mul(exp_neg_theta_1_t);
    
    const mean = term1.mul(z0).add(term2.mul(z1)).div(denominator);
    
    const var_term = torch.ones(exp_neg_theta_t.shape, false, device).sub(exp_neg_theta_t.pow(2))
      .mul(torch.ones(exp_neg_theta_1_t.shape, false, device).sub(exp_neg_theta_1_t.pow(2)))
      .mul(this.sigma ** 2 / (2 * this.theta))
      .div(denominator);
    
    return [mean, var_term];
  }
}

// Enhanced Label Trainer using js-pytorch
export class EnhancedLabelTrainer {
  constructor(device = "gpu") {
    this.device = device;
    // Initialize models
    this.vae = new LabelConditionedVAE(device);
    this.drift = new LabelConditionedDrift(device);

    // Optimizers (js-pytorch style)
    this.opt_vae = new torch.optim.Adam(this.vae.parameters(), CONFIG.LR || 0.0002);
    this.opt_drift = new torch.optim.Adam(this.drift.parameters(), (CONFIG.LR || 0.0002) * (CONFIG.DRIFT_LR_MULTIPLIER || 1.0));

    // Loss modules (functional ones are missing)
    this.mse_loss = new torch.nn.MSELoss();

    // Training state
    this.epoch = 0;
    this.step = 0;
    this.phase = 1;

    // OU reference process
    this.ou_ref = new OUReference(CONFIG.OU_THETA || 1.0, CONFIG.OU_SIGMA || Math.sqrt(2));

    console.log(`💓 Enhanced Label Trainer initialized (js-pytorch / ${device.toUpperCase()})`);
  }

  setPhase(phase) {
    if (typeof phase === 'string') {
      switch(phase) {
        case 'vae': this.phase = 1; break;
        case 'drift': this.phase = 2; break;
        case 'both': this.phase = 3; break;
      }
    } else {
      this.phase = phase;
    }
  }

  async trainStep(batch, labels) {
    const images = torch.tensor(batch, false, this.device);
    // Embedding expects [B, T]
    const labelsTensor = torch.tensor(labels.map(l => [l]), false, this.device);

    if (this.phase === 1) {
      // Phase 1: VAE
      this.opt_vae.zero_grad();
      const [recon, mu, logvar] = this.vae.forward(images, labelsTensor);
      
      const recon_loss = this.mse_loss.forward(recon, images);
      
      // Simplified KL loss for js-pytorch compatibility
      const mu_sq = mu.pow(2);
      const var_val = torch.exp(logvar);
      const kl_element = mu_sq.add(var_val).add(logvar.add(1).neg());
      const kl_loss = kl_element.sum(-1).mean(-1).mul(0.5);
      
      const total_loss = recon_loss.add(kl_loss.mul(CONFIG.KL_WEIGHT || 0.01));
      total_loss.backward();
      this.opt_vae.step();
      
      return { 
        loss: total_loss._data[0], 
        metrics: { 
          phase: 'vae', 
          recon_loss: recon_loss._data[0], 
          kl_loss: kl_loss._data[0] 
        } 
      };
    } else {
      // Phase 2/3: Drift
      this.opt_drift.zero_grad();
      
      // Get latents
      const [mu, logvar] = this.vae.encode(images, labelsTensor);
      const z1 = this.vae.reparameterize(mu, logvar);
      
      // Sample t and z0
      const t = torch.rand([images.shape[0], 1], false, this.device);
      const z0 = torch.randn(z1.shape, false, this.device);
      
      // Sample zt and target
      const [mean, var_] = this.ou_ref.bridgeSample(z0, z1, t);
      // zt = mean + sqrt(var) * eps
      const zt = mean.add(torch.randn(mean.shape, false, this.device).mul(torch.sqrt(var_)));
      const target = z1.sub(z0);
      
      const pred = this.drift.forward(zt, t, labelsTensor);
      const drift_loss = this.mse_loss.forward(pred, target);
      
      drift_loss.backward();
      this.opt_drift.step();
      
      if (this.phase === 3) {
        // Also update VAE in phase 3
        this.opt_vae.zero_grad();
        const [recon_p3] = this.vae.forward(images, labelsTensor);
        const recon_loss_p3 = this.mse_loss.forward(recon_p3, images);
        recon_loss_p3.backward();
        this.opt_vae.step();
      }
      
      return { 
        loss: drift_loss._data[0], 
        metrics: { 
          phase: this.phase === 2 ? 'drift' : 'both', 
          drift_loss: drift_loss._data[0] 
        } 
      };
    }
  }

  async generateSamples(labels, count = 4) {
    // Embedding expects [B, T]
    const labelsTensor = torch.tensor(labels.slice(0, count).map(l => [l]), false, this.device);
    // Latent dim is 64
    const z = torch.randn([labelsTensor.shape[0], 64], false, this.device);
    const samples = this.vae.decode(z, labelsTensor);
    // js-pytorch tensor to array conversion
    return samples.tolist ? samples.tolist() : samples._data;
  }

  getCheckpoint() {
    return {
      vae_params: this.vae.parameters().map(p => p.tolist ? p.tolist() : p._data),
      drift_params: this.drift.parameters().map(p => p.tolist ? p.tolist() : p._data),
      epoch: this.epoch,
      phase: this.phase
    };
  }

  loadCheckpoint(checkpoint) {
    if (checkpoint.vae_params) {
      const params = this.vae.parameters();
      checkpoint.vae_params.forEach((data, i) => {
        if (params[i]) {
          params[i]._data = data;
        }
      });
    }
    if (checkpoint.drift_params) {
      const params = this.drift.parameters();
      checkpoint.drift_params.forEach((data, i) => {
        if (params[i]) {
          params[i]._data = data;
        }
      });
    }
    this.epoch = checkpoint.epoch || 0;
    this.phase = checkpoint.phase || 1;
  }
}

export default EnhancedLabelTrainer;
