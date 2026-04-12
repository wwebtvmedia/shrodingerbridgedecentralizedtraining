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
    
    // Convert to tensors outside of tidy/grads
    const images = tf.tensor(batch).reshape([-1, CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, 3]);
    const labelsTensor = tf.tensor(labels, [labels.length], 'int32');

    try {
      if (this.phase === 1) {
        // Phase 1: VAE
        const vaeVars = this.getVaeVariables();
        const gradsObj = tf.variableGrads(() => {
          return tf.tidy(() => {
            const [recon, mu, logvar] = this.vae.forward(images, labelsTensor);
            const recon_loss = tf.losses.meanSquaredError(images, recon);
            
            // KL loss
            const kl_element = tf.add(tf.add(tf.exp(logvar), tf.square(mu)), tf.neg(tf.add(1, logvar)));
            const kl_loss = tf.mul(0.5, tf.mean(tf.sum(kl_element, [1, 2, 3])));
            
            return tf.add(recon_loss, tf.mul(CONFIG.KL_WEIGHT || 0.002, kl_loss));
          });
        }, vaeVars);

        this.opt_vae.applyGradients(gradsObj.grads);
        
        const lossVal = gradsObj.value.dataSync()[0];
        tf.dispose(gradsObj);
        
        return { 
          loss: lossVal, 
          metrics: { phase: 'vae' } 
        };
      } else {
        // Phase 2/3: Drift
        const driftVars = this.getDriftVariables();
        const gradsObj = tf.variableGrads(() => {
          return tf.tidy(() => {
            // Encapsulate non-gradient parts in nested tidy
            const [z1, t, z0] = tf.tidy(() => {
              const [mu, logvar] = this.vae.encode(images, labelsTensor);
              const z_1 = this.vae.reparameterize(mu, logvar);
              const _t = tf.randomUniform([images.shape[0], 1]);
              const _z0 = tf.randomNormal(z_1.shape);
              return [z_1, _t, _z0];
            });

            const [zt, target] = tf.tidy(() => {
              const [mean, var_] = this.ou_ref.bridgeSample(z0, z1, t);
              const _zt = tf.add(mean, tf.mul(tf.randomNormal(mean.shape), tf.sqrt(var_)));
              const _target = tf.sub(z1, z0);
              return [_zt, _target];
            });
            
            const pred = this.drift.forward(zt, t, labelsTensor);
            return tf.losses.meanSquaredError(target, pred);
          });
        }, driftVars);

        this.opt_drift.applyGradients(gradsObj.grads);
        const lossVal = gradsObj.value.dataSync()[0];
        tf.dispose(gradsObj);
        
        if (this.phase === 3) {
          const vVars = this.getVaeVariables();
          const vGrads = tf.variableGrads(() => {
            return tf.tidy(() => {
              const [recon] = this.vae.forward(images, labelsTensor);
              return tf.losses.meanSquaredError(images, recon);
            });
          }, vVars);
          this.opt_vae.applyGradients(vGrads.grads);
          tf.dispose(vGrads);
        }

        return { 
          loss: lossVal, 
          metrics: { phase: this.phase === 2 ? 'drift' : 'both' } 
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

  // Helper to collect all trainable variables from models
  getVaeVariables() {
    const vars = [];
    // VAE variables
    [this.vae.labelEmb, this.vae.encIn, this.vae.zMean, this.vae.zLogvar, this.vae.decIn, this.vae.decOut].forEach(layer => {
      if (layer && layer.trainableWeights) {
        layer.trainableWeights.forEach(w => vars.push(w.val));
      }
    });
    
    // Complex blocks
    this.vae.encBlocks.forEach(block => {
      if (block.conv1 && block.conv1.trainableWeights) block.conv1.trainableWeights.forEach(w => vars.push(w.val));
      if (block.conv2 && block.conv2.trainableWeights) block.conv2.trainableWeights.forEach(w => vars.push(w.val));
      if (block.labelProj && block.labelProj.trainableWeights) block.labelProj.trainableWeights.forEach(w => vars.push(w.val));
      if (block.skip && block.skip.trainableWeights) block.skip.trainableWeights.forEach(w => vars.push(w.val));
    });

    this.vae.decBlocks.forEach(block => {
      if (block.conv && block.conv.trainableWeights) block.conv.trainableWeights.forEach(w => vars.push(w.val));
      if (block.conv1 && block.conv1.trainableWeights) block.conv1.trainableWeights.forEach(w => vars.push(w.val));
      if (block.conv2 && block.conv2.trainableWeights) block.conv2.trainableWeights.forEach(w => vars.push(w.val));
      if (block.labelProj && block.labelProj.trainableWeights) block.labelProj.trainableWeights.forEach(w => vars.push(w.val));
    });

    return vars;
  }

  getDriftVariables() {
    const vars = [];
    // Drift variables
    if (this.drift.timeMlp) {
      this.drift.timeMlp.layers.forEach(l => {
        if (l.trainableWeights) l.trainableWeights.forEach(w => vars.push(w.val));
      });
    }
    
    [this.drift.labelEmb, this.drift.condProj, this.drift.head, this.drift.down2Conv, this.drift.up2Conv, this.drift.tail].forEach(layer => {
      if (layer && layer.trainableWeights) {
        layer.trainableWeights.forEach(w => vars.push(w.val));
      }
    });

    [this.drift.down1, this.drift.down2Block, this.drift.mid1, this.drift.mid2, this.drift.up2Block, this.drift.up1].forEach(block => {
      if (block.conv1 && block.conv1.trainableWeights) block.conv1.trainableWeights.forEach(w => vars.push(w.val));
      if (block.conv2 && block.conv2.trainableWeights) block.conv2.trainableWeights.forEach(w => vars.push(w.val));
      if (block.labelProj && block.labelProj.trainableWeights) block.labelProj.trainableWeights.forEach(w => vars.push(w.val));
    });

    return vars;
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
