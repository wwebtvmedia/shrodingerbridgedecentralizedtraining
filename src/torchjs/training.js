// Enhanced Schrödinger Bridge Trainer using TensorFlow.js
// Optimized for GPU acceleration and high-fidelity generation (96x96)
// Aligned with enhancedoptimaltransport/training.py

import * as tf from "@tensorflow/tfjs";
import { CONFIG, klDivergenceSpatial, calcSNR } from "../config.js";
import { LabelConditionedVAE, LabelConditionedDrift } from "./models.js";

/**
 * Huber Loss for robust regression
 */
function huberLoss(yTrue, yPred, delta = 1.0) {
  return tf.tidy(() => {
    const error = tf.sub(yTrue, yPred);
    const absError = tf.abs(error);
    const quadratic = tf.minimum(absError, delta);
    const linear = tf.sub(absError, quadratic);
    return tf.mean(
      tf.add(tf.mul(0.5, tf.square(quadratic)), tf.mul(delta, linear)),
    );
  });
}

/**
 * Simplified SSIM for structural integrity
 */
function ssimLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const muX = tf.mean(yTrue, [1, 2]).reshape([-1, 1, 1, 3]);
    const muY = tf.mean(yPred, [1, 2]).reshape([-1, 1, 1, 3]);
    const sigmaX = tf
      .mean(tf.square(tf.sub(yTrue, muX)), [1, 2])
      .reshape([-1, 1, 1, 3]);
    const sigmaY = tf
      .mean(tf.square(tf.sub(yPred, muY)), [1, 2])
      .reshape([-1, 1, 1, 3]);
    const sigmaXY = tf
      .mean(tf.mul(tf.sub(yTrue, muX), tf.sub(yPred, muY)), [1, 2])
      .reshape([-1, 1, 1, 3]);

    const c1 = 0.01 ** 2;
    const c2 = 0.03 ** 2;

    const numerator = tf.mul(
      tf.add(tf.mul(2, tf.mul(muX, muY)), c1),
      tf.add(tf.mul(2, sigmaXY), c2),
    );
    const denominator = tf.mul(
      tf.add(tf.add(tf.square(muX), tf.square(muY)), c1),
      tf.add(tf.add(sigmaX, sigmaY), c2),
    );

    const ssim = tf.div(numerator, denominator);
    return tf.sub(1, tf.mean(ssim));
  });
}
// OU Reference Process (Aligned with Python)
class OUReference {
  constructor(theta = 1.0, sigma = Math.sqrt(2)) {
    this.theta = theta;
    this.sigma = sigma;
  }

  bridgeSample(z0, z1, t) {
    return tf.tidy(() => {
      let t_bc = t;
      if (t.shape.length === 2 && z0.shape.length === 4) {
        t_bc = t.reshape([t.shape[0], 1, 1, 1]);
      }

      const exp_neg_theta_t = tf.exp(tf.mul(t_bc, -this.theta));
      const exp_neg_theta_1_t = tf.exp(tf.mul(tf.sub(1, t_bc), -this.theta));
      const exp_neg_theta = Math.exp(-this.theta);

      const denominator = 1 - exp_neg_theta ** 2;

      const term1 = tf.mul(
        exp_neg_theta_t,
        tf.sub(1, tf.pow(exp_neg_theta_1_t, 2)),
      );
      const term2 = tf.mul(
        tf.sub(1, tf.pow(exp_neg_theta_t, 2)),
        exp_neg_theta_1_t,
      );

      const mean = tf.div(
        tf.add(tf.mul(term1, z0), tf.mul(term2, z1)),
        denominator,
      );

      const var_term = tf.div(
        tf.mul(
          tf.mul(
            tf.sub(1, tf.pow(exp_neg_theta_t, 2)),
            tf.sub(1, tf.pow(exp_neg_theta_1_t, 2)),
          ),
          this.sigma ** 2 / (2 * this.theta),
        ),
        denominator,
      );

      return [mean, var_term];
    });
  }

  bridgeVelocity(z0, z1, t) {
    // Exact velocity d/dt mean(t)
    return tf.tidy(() => {
      const dt = 1e-4;
      const [m_plus] = this.bridgeSample(z0, z1, tf.add(t, dt));
      const [m_minus] = this.bridgeSample(z0, z1, tf.maximum(0, tf.sub(t, dt)));
      return tf.div(tf.sub(m_plus, m_minus), 2 * dt);
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

    // Anchor model for consistency
    this.vae_ref = null;

    // Optimizers
    this.opt_vae = tf.train.adam(CONFIG.LR || 0.0002);
    this.opt_drift = tf.train.adam(
      (CONFIG.LR || 0.0002) * (CONFIG.DRIFT_LR_MULTIPLIER || 1.0),
    );

    // Training state
    this.epoch = 0;
    this.step = 0;
    this.phase = 1;

    // OU reference process
    this.ou_ref = new OUReference(
      CONFIG.OU_THETA || 1.0,
      CONFIG.OU_SIGMA || Math.sqrt(2),
    );

    console.log(
      `💓 Enhanced Label Trainer initialized (TensorFlow.js / ${device.toUpperCase()})`,
    );
  }

  setPhase(phase) {
    if (typeof phase === "string") {
      switch (phase) {
        case "vae":
          this.phase = 1;
          break;
        case "drift":
          this.phase = 2;
          break;
        case "both":
          this.phase = 3;
          break;
      }
    } else {
      this.phase = phase;
    }

    // Create vae_ref if moving to drift phase
    if (this.phase >= 2 && !this.vae_ref) {
      this.updateVaeRef();
    }
  }

  async updateVaeRef() {
    console.log("⚓ Creating VAE anchor for consistency...");
    if (this.vae_ref) {
      this.vae_ref.dispose();
    }
    this.vae_ref = new LabelConditionedVAE("vae_ref");

    const checkpoint = await this.saveCheckpoint();
    if (checkpoint.vae_params) {
      const refVars = this.collectVariables(this.vae_ref);
      refVars.forEach((v, i) => {
        if (checkpoint.vae_params[i]) {
          try {
            const data = checkpoint.vae_params[i];
            const expectedSize = v.shape.reduce((a, b) => a * b, 1);
            const actualSize = Array.isArray(data)
              ? data.flat(Infinity).length
              : 0;

            if (expectedSize !== actualSize) {
              return; // Skip mismatch
            }

            tf.tidy(() => {
              const tensor = tf.tensor(data, v.shape);
              v.assign(tensor);
            });
          } catch (e) {
            console.warn(
              `⚠️ Failed to sync reference variable ${i}: ${e.message}`,
            );
          }
        }
      });
    }
  }

  dispose() {
    if (this.vae) this.vae.dispose();
    if (this.drift) this.drift.dispose();
    if (this.vae_ref) this.vae_ref.dispose();
    if (this.opt_vae) this.opt_vae.dispose();
    if (this.opt_drift) this.opt_drift.dispose();
  }

  // Robustly collect all variables from a model or object
  collectVariables(obj, vars = [], visited = new Set()) {
    if (!obj || typeof obj !== "object" || visited.has(obj)) return vars;
    visited.add(obj);

    // 1. If it's a Layer, get its weights and then its sub-components
    if (obj.trainableWeights) {
      obj.trainableWeights.forEach((w) => {
        const v = w.val || w;
        if (v instanceof tf.Variable && !vars.includes(v)) {
          vars.push(v);
        }
      });
    }

    // 2. If it's a Sequential or Model, it has a layers property
    if (obj.layers && Array.isArray(obj.layers)) {
      obj.layers.forEach((l) => this.collectVariables(l, vars, visited));
    }

    // 3. Recursively check custom properties (like encBlocks, decBlocks, etc.)
    const keys = Object.keys(obj);
    for (const key of keys) {
      // Skip internal properties, already visited ones, and known non-model properties
      if (
        key.startsWith("_") ||
        key === "layers" ||
        key === "trainableWeights" ||
        key === "vae_ref" ||
        key === "ou_ref" ||
        key === "opt_vae" ||
        key === "opt_drift"
      )
        continue;

      const prop = obj[key];
      if (prop && typeof prop === "object") {
        this.collectVariables(prop, vars, visited);
      }
    }

    return vars;
  }

  getVaeVariables() {
    return this.collectVariables(this.vae);
  }

  getDriftVariables() {
    return this.collectVariables(this.drift);
  }

  async trainStep(batch, labels, textBytes = null) {
    // console.log("DEBUG: labels =", labels, "type =", typeof labels, "isArray =", Array.isArray(labels));
    // Convert to tensors outside of tidy/grads
    const images = tf
      .tensor(batch)
      .reshape([-1, CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, 3]);

    // Safety check for labels shape
    const labelsArray = Array.isArray(labels) ? labels : [labels];
    const labelsTensor = tf.tensor(labelsArray, [labelsArray.length], "int32");

    let textBytesTensor = null;
    if (textBytes) {
      // console.log("DEBUG: textBytes =", textBytes);
      try {
        textBytesTensor = tf.tensor(
          textBytes,
          [textBytes.length, textBytes[0].length],
          "int32",
        );
      } catch (e) {
        console.error(
          "❌ Failed to create textBytesTensor:",
          e.message,
          "shape info:",
          textBytes.length,
          textBytes[0]?.length,
        );
        throw e;
      }
    }

    try {
      if (this.phase === 1) {
        const vaeVars = this.getVaeVariables();
        const gradsObj = tf.variableGrads(() => {
          return tf.tidy(() => {
            const [recon, mu, logvar] = this.vae.forward(
              images,
              labelsTensor,
              textBytesTensor,
            );

            const raw_l1 = tf.losses.absoluteDifference(images, recon);
            const recon_loss = tf.mul(raw_l1, CONFIG.RECON_WEIGHT || 5.0);

            const kl_loss = tf.mul(
              klDivergenceSpatial(mu, logvar),
              CONFIG.KL_WEIGHT || 0.002,
            );

            let ssim_loss = tf.scalar(0);
            if (CONFIG.SSIM_WEIGHT > 0) {
              ssim_loss = tf.mul(ssimLoss(images, recon), CONFIG.SSIM_WEIGHT);
            }

            return tf.add(tf.add(recon_loss, kl_loss), ssim_loss);
          });
        }, vaeVars);

        this.opt_vae.applyGradients(gradsObj.grads);
        const lossVal = gradsObj.value.dataSync()[0];
        tf.dispose(gradsObj.value);
        tf.dispose(gradsObj.grads);
        return { loss: lossVal, metrics: { phase: "vae" } };
      } else {
        const driftVars = this.getDriftVariables();
        const gradsObj = tf.variableGrads(() => {
          return tf.tidy(() => {
            // Temperature annealing
            const temp =
              CONFIG.TEMPERATURE_START +
              (CONFIG.TEMPERATURE_END - CONFIG.TEMPERATURE_START) *
                (this.epoch / CONFIG.EPOCHS);

            const [z1, t, z0, mu, mu_ref] = tf.tidy(() => {
              const [mu_curr, logvar_curr] = this.vae.encode(
                images,
                labelsTensor,
                textBytesTensor,
              );
              let mu_r = mu_curr;
              if (this.vae_ref) {
                [mu_r] = this.vae_ref.encode(
                  images,
                  labelsTensor,
                  textBytesTensor,
                );
              }

              const noise = tf.mul(
                tf.randomNormal(mu_curr.shape),
                tf.mul(tf.exp(tf.mul(0.5, logvar_curr)), temp),
              );
              const _z1 = tf.add(mu_curr, noise);
              const _t = tf.randomUniform([images.shape[0], 1]);
              const _z0 = tf.randomNormal(
                _z1.shape,
                0,
                CONFIG.CST_COEF_GAUSSIAN_PRIO || 1.0,
              );
              return [_z1, _t, _z0, mu_curr, mu_r];
            });

            // Bridge sampling
            const [zt, target] = tf.tidy(() => {
              if (CONFIG.USE_OU_BRIDGE && this.ou_ref) {
                const [mean, var_] = this.ou_ref.bridgeSample(z0, z1, t);
                const _zt = tf.add(
                  mean,
                  tf.mul(
                    tf.randomNormal(mean.shape),
                    tf.sqrt(tf.add(var_, 1e-8)),
                  ),
                );
                const _target = this.ou_ref.bridgeVelocity(z0, z1, t);
                return [_zt, _target];
              } else {
                const t_bc = t.reshape([-1, 1, 1, 1]);
                const _zt = tf.add(
                  tf.mul(tf.sub(1, t_bc), z0),
                  tf.mul(t_bc, z1),
                );
                const _target = tf.sub(z1, z0);
                return [_zt, _target];
              }
            });

            // Classifier-Free Guidance Dropout
            let trainLabels = labelsTensor;
            let trainText = textBytesTensor;
            if (Math.random() < (CONFIG.LABEL_DROPOUT_PROB || 0.1)) {
              trainLabels = tf.fill(
                labelsTensor.shape,
                CONFIG.NUM_CLASSES - 1,
                "int32",
              );
              trainText = null;
            }

            const pred = this.drift.forward(zt, t, trainLabels, trainText);

            // Time-weighted Huber loss
            const t_bc = t.reshape([-1, 1, 1, 1]);
            const timeWeights = tf.add(
              1.0,
              tf.mul(CONFIG.TIME_WEIGHT_FACTOR || 2.0, t_bc),
            );
            const drift_loss = tf.mul(
              huberLoss(tf.mul(target, timeWeights), tf.mul(pred, timeWeights)),
              CONFIG.DRIFT_WEIGHT || 1.0,
            );

            // Consistency loss
            const consistency_loss = tf.mul(
              tf.losses.meanSquaredError(mu, mu_ref),
              CONFIG.CONSISTENCY_WEIGHT || 1.0,
            );

            let total_loss = tf.add(drift_loss, consistency_loss);

            // Phase 3 Reconstruction Enhancement
            if (this.phase === 3) {
              const recon_p3 = this.vae.decode(
                mu,
                labelsTensor,
                textBytesTensor,
              );
              const p3_recon_loss = tf.mul(
                tf.losses.absoluteDifference(images, recon_p3),
                (CONFIG.RECON_WEIGHT || 5.0) *
                  (CONFIG.PHASE3_RECON_SCALE || 0.1),
              );
              total_loss = tf.add(total_loss, p3_recon_loss);
            }

            return total_loss;
          });
        }, driftVars);

        this.opt_drift.applyGradients(gradsObj.grads);
        const lossVal = gradsObj.value.dataSync()[0];
        tf.dispose(gradsObj.value);
        tf.dispose(gradsObj.grads);

        // Phase 3: Also update VAE if needed
        if (this.phase === 3) {
          const vVars = this.getVaeVariables();
          const vGrads = tf.variableGrads(() => {
            return tf.tidy(() => {
              const [recon] = this.vae.forward(
                images,
                labelsTensor,
                textBytesTensor,
              );
              return tf.losses.absoluteDifference(images, recon);
            });
          }, vVars);
          this.opt_vae.applyGradients(vGrads.grads);
          tf.dispose(vGrads.value);
          tf.dispose(vGrads.grads);
        }

        return {
          loss: lossVal,
          metrics: { phase: this.phase === 2 ? "drift" : "both" },
        };
      }
    } finally {
      tf.dispose([images, labelsTensor]);
      if (textBytesTensor) tf.dispose(textBytesTensor);
    }
  }

  async generateSamples(labels, count = 4, textBytes = null) {
    return tf.tidy(() => {
      const selectedLabels = labels.slice(0, count);
      const labelsTensor = tf.tensor(
        selectedLabels,
        [selectedLabels.length],
        "int32",
      );
      const textBytesTensor = textBytes
        ? tf.tensor(
            textBytes.slice(0, count),
            [count, textBytes[0].length],
            "int32",
          )
        : null;

      const z = tf.randomNormal(
        [count, CONFIG.LATENT_H, CONFIG.LATENT_W, CONFIG.LATENT_CHANNELS],
        0,
        CONFIG.CST_COEF_GAUSSIAN_PRIO || 1.0,
      );
      const samples = this.vae.decode(z, labelsTensor, textBytesTensor);

      return samples.arraySync();
    });
  }

  async getCheckpoint() {
    return this.saveCheckpoint();
  }

  async getHashSample() {
    // Only download a few small weights for hashing
    const vaeVars = this.getVaeVariables();
    if (vaeVars.length > 0) {
      return await vaeVars[0].array();
    }
    return [Math.random()];
  }

  async saveCheckpoint() {
    const vaeVars = this.getVaeVariables();
    const driftVars = this.getDriftVariables();

    // Read weights in batches to avoid memory spikes and shader compilation issues
    const batchSize = 10;

    const vae_params = [];
    for (let i = 0; i < vaeVars.length; i += batchSize) {
      const batch = vaeVars.slice(i, i + batchSize);
      const results = await Promise.all(batch.map((v) => v.array()));
      vae_params.push(...results);
    }

    const drift_params = [];
    for (let i = 0; i < driftVars.length; i += batchSize) {
      const batch = driftVars.slice(i, i + batchSize);
      const results = await Promise.all(batch.map((v) => v.array()));
      drift_params.push(...results);
    }

    return {
      epoch: this.epoch,
      phase: this.phase,
      vae_params,
      drift_params,
    };
  }

  async loadCheckpoint(checkpoint) {
    if (!checkpoint) return;

    this.epoch = checkpoint.epoch || 0;
    this.phase = checkpoint.phase || 1;

    if (checkpoint.vae_params) {
      const vaeVars = this.getVaeVariables();
      vaeVars.forEach((v, i) => {
        if (checkpoint.vae_params[i]) {
          try {
            const data = checkpoint.vae_params[i];
            const expectedSize = v.shape.reduce((a, b) => a * b, 1);
            const actualSize = Array.isArray(data)
              ? data.flat(Infinity).length
              : 0;

            if (expectedSize !== actualSize) {
              console.warn(
                `⚠️ Shape mismatch for VAE variable ${i} (${v.name}): Expected ${v.shape} (${expectedSize} values) but got ${actualSize}`,
              );
              return;
            }

            tf.tidy(() => {
              const tensor = tf.tensor(data, v.shape);
              v.assign(tensor);
            });
          } catch (e) {
            console.warn(
              `⚠️ Failed to load VAE variable ${i} (${v.name}): ${e.message}`,
            );
          }
        }
      });
    }

    if (checkpoint.drift_params) {
      const driftVars = this.getDriftVariables();
      driftVars.forEach((v, i) => {
        if (checkpoint.drift_params[i]) {
          try {
            const data = checkpoint.drift_params[i];
            const expectedSize = v.shape.reduce((a, b) => a * b, 1);
            const actualSize = Array.isArray(data)
              ? data.flat(Infinity).length
              : 0;

            if (expectedSize !== actualSize) {
              console.warn(
                `⚠️ Shape mismatch for Drift variable ${i} (${v.name}): Expected ${v.shape} (${expectedSize} values) but got ${actualSize}`,
              );
              return;
            }

            tf.tidy(() => {
              const tensor = tf.tensor(data, v.shape);
              v.assign(tensor);
            });
          } catch (e) {
            console.warn(
              `⚠️ Failed to load Drift variable ${i} (${v.name}): ${e.message}`,
            );
          }
        }
      });
    }

    console.log(`📥 Checkpoint loaded for epoch ${this.epoch}`);
  }
}

export default EnhancedLabelTrainer;
