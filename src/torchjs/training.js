// Enhanced Schrödinger Bridge Trainer using TensorFlow.js
// Optimized for GPU acceleration and high-fidelity generation (96x96)

import * as tf from '@tensorflow/tfjs';
import { CONFIG } from "../config.js";
import { LabelConditionedVAE, LabelConditionedDrift } from "./models.js";

// OU Reference Process (Ported to TFJS)
class OUReference {
  constructor(theta = 1.0, sigma = Math.sqrt(2)) {
    this.theta = theta;
    this.sigma = sigma;
  }

  bridgeSample(z0, z1, t) {
    return tf.tidy(() => {
      // Ensure t can broadcast with z0/z1 which are typically [B, H, W, C]
      let t_bc = t;
      if (t.shape.length === 2 && z0.shape.length === 4) {
        t_bc = t.reshape([t.shape[0], 1, 1, 1]);
      }

      const exp_neg_theta_t = tf.exp(tf.mul(t_bc, -this.theta));
      const exp_neg_theta_1_t = tf.exp(tf.mul(tf.sub(1, t_bc), -this.theta));
      const exp_neg_theta = Math.exp(-this.theta);

      const denominator = 1 - exp_neg_theta ** 2;
      
      const term1 = tf.mul(exp_neg_theta_t, tf.sub(1, tf.pow(exp_neg_theta_1_t, 2)));
      const term2 = tf.mul(tf.sub(1, tf.pow(exp_neg_theta_t, 2)), exp_neg_theta_1_t);
      
      const mean = tf.div(tf.add(tf.mul(term1, z0), tf.mul(term2, z1)), denominator);
      
      const var_term = tf.div(
        tf.mul(
          tf.mul(tf.sub(1, tf.pow(exp_neg_theta_t, 2)), tf.sub(1, tf.pow(exp_neg_theta_1_t, 2))),
          this.sigma ** 2 / (2 * this.theta)
        ),
        denominator
      );
      
      return [mean, var_term];
    });
  }
}

// Enhanced Label Trainer using TensorFlow.js
export class EnhancedLabelTrainer {
  constructor(device = "gpu") {
    this.device = device;
    // Initialize models
    this.vae = new LabelConditionedVAE();
    this.drift = new LabelConditionedDrift();

    // Optimizers
    this.opt_vae = tf.train.adam(CONFIG.LR || 0.0002);
    this.opt_drift = tf.train.adam((CONFIG.LR || 0.0002) * (CONFIG.DRIFT_LR_MULTIPLIER || 1.0));

    // Training state
    this.epoch = 0;
    this.step = 0;
    this.phase = 1;

    // OU reference process
    this.ou_ref = new OUReference(CONFIG.OU_THETA || 1.0, CONFIG.OU_SIGMA || Math.sqrt(2));

    console.log(`💓 Enhanced Label Trainer initialized (TensorFlow.js / ${device.toUpperCase()})`);
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
    const numTensorsStart = tf.memory().numTensors;
    const images = tf.tensor(batch).reshape([-1, CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, 3]);
    const labelsTensor = tf.tensor(labels, [labels.length], 'int32');

    try {
      if (this.phase === 1) {
        // Phase 1: VAE
        let lossVal = 0;
        let metrics = {};

        const grads = this.opt_vae.computeGradients(() => {
          const [recon, mu, logvar] = this.vae.forward(images, labelsTensor);
          const recon_loss = tf.losses.meanSquaredError(images, recon);
          
          // KL loss: 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
          const kl_element = tf.add(tf.add(tf.exp(logvar), tf.square(mu)), tf.neg(tf.add(1, logvar)));
          const kl_loss = tf.mul(0.5, tf.mean(tf.sum(kl_element, [1, 2, 3])));
          
          const total_loss = tf.add(recon_loss, tf.mul(CONFIG.KL_WEIGHT || 0.002, kl_loss));
          
          lossVal = total_loss.dataSync()[0];
          metrics = {
            phase: 'vae',
            recon_loss: recon_loss.dataSync()[0],
            kl_loss: kl_loss.dataSync()[0]
          };
          
          return total_loss;
        });

        this.opt_vae.applyGradients(grads.grads);
        
        // Explicitly dispose of all tensors in the grads object
        const total_loss = grads.value;
        tf.dispose(total_loss);
        Object.values(grads.grads).forEach(t => tf.dispose(t));
        
        return { loss: lossVal, metrics };
      } else {
        // Phase 2/3: Drift
        let driftLossVal = 0;

        const grads = this.opt_drift.computeGradients(() => {
          // Get latents (no grad for VAE in phase 2)
          return tf.tidy(() => {
            const [mu, logvar] = this.vae.encode(images, labelsTensor);
            const z1 = this.vae.reparameterize(mu, logvar);
            
            // Sample t and z0
            const t = tf.randomUniform([images.shape[0], 1]);
            const z0 = tf.randomNormal(z1.shape);
            
            // Sample zt and target
            const [mean, var_] = this.ou_ref.bridgeSample(z0, z1, t);
            const zt = tf.add(mean, tf.mul(tf.randomNormal(mean.shape), tf.sqrt(var_)));
            const target = tf.sub(z1, z0);
            
            const pred = this.drift.forward(zt, t, labelsTensor);
            const drift_loss = tf.losses.meanSquaredError(target, pred);
            
            driftLossVal = drift_loss.dataSync()[0];
            return drift_loss;
          });
        });

        this.opt_drift.applyGradients(grads.grads);
        
        // Explicitly dispose of all tensors in the grads object
        const drift_loss = grads.value;
        tf.dispose(drift_loss);
        Object.values(grads.grads).forEach(t => tf.dispose(t));
        
        if (this.phase === 3) {
          const vGrads = this.opt_vae.computeGradients(() => {
            const [recon] = this.vae.forward(images, labelsTensor);
            return tf.losses.meanSquaredError(images, recon);
          });
          this.opt_vae.applyGradients(vGrads.grads);
          
          const vLoss = vGrads.value;
          tf.dispose(vLoss);
          Object.values(vGrads.grads).forEach(t => tf.dispose(t));
        }

        return { 
          loss: driftLossVal, 
          metrics: { 
            phase: this.phase === 2 ? 'drift' : 'both', 
            drift_loss: driftLossVal 
          } 
        };
      }
    } finally {
      tf.dispose([images, labelsTensor]);
      const numTensorsEnd = tf.memory().numTensors;
      if (numTensorsEnd > numTensorsStart) {
        console.warn(`⚠️ Potential memory leak: ${numTensorsEnd - numTensorsStart} tensors not disposed in trainStep.`);
      }
    }
  }

  async generateSamples(labels, count = 4) {
    return tf.tidy(() => {
      const selectedLabels = labels.slice(0, count);
      const labelsTensor = tf.tensor(selectedLabels, [selectedLabels.length], 'int32');
      
      // Latent shape is [B, 12, 12, 8]
      const z = tf.randomNormal([count, CONFIG.LATENT_H, CONFIG.LATENT_W, CONFIG.LATENT_CHANNELS]);
      const samples = this.vae.decode(z, labelsTensor);
      
      // Convert to array [B, H, W, C] -> flat or nested array
      return samples.arraySync();
    });
  }

  getCheckpoint() {
    // Collect all weights from models
    // Since we don't have a single parameters() method, we'll need to collect them manually 
    // or use a naming convention.
    // In a real implementation, we would traverse all layers.
    console.log("📤 Getting checkpoint weights...");
    // This is a simplified version - in a full implementation, you'd iterate over all layers
    return {
      epoch: this.epoch,
      phase: this.phase,
      // We would need to serialize weights here
      weights_serialized: true 
    };
  }

  loadCheckpoint(checkpoint) {
    this.epoch = checkpoint.epoch || 0;
    this.phase = checkpoint.phase || 1;
    console.log("📥 Checkpoint loaded (meta only in this prototype)");
  }
}

export default EnhancedLabelTrainer;
