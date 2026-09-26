// Enhanced Schrödinger Bridge Trainer using TensorFlow.js
// Optimized for WebGPU acceleration and high-fidelity generation (96x96)
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
 * Total Variation Loss for spatial smoothness
 */
function totalVariationLoss(img) {
  return tf.tidy(() => {
    // img shape [B, H, W, C]
    const hDiff = tf.square(
      tf.sub(
        tf.slice(img, [0, 1, 0, 0], [-1, -1, -1, -1]),
        tf.slice(img, [0, 0, 0, 0], [-1, img.shape[1] - 1, -1, -1]),
      ),
    );
    const wDiff = tf.square(
      tf.sub(
        tf.slice(img, [0, 0, 1, 0], [-1, -1, -1, -1]),
        tf.slice(img, [0, 0, 0, 0], [-1, -1, img.shape[2] - 1, -1]),
      ),
    );
    return tf.add(tf.mean(hDiff), tf.mean(wDiff));
  });
}

/**
 * Edge gradient match loss
 */
function edgeLoss(recon, target) {
  return tf.tidy(() => {
    const reconDy = tf.abs(
      tf.sub(
        tf.slice(recon, [0, 1, 0, 0], [-1, -1, -1, -1]),
        tf.slice(recon, [0, 0, 0, 0], [-1, recon.shape[1] - 1, -1, -1]),
      ),
    );
    const targetDy = tf.abs(
      tf.sub(
        tf.slice(target, [0, 1, 0, 0], [-1, -1, -1, -1]),
        tf.slice(target, [0, 0, 0, 0], [-1, target.shape[1] - 1, -1, -1]),
      ),
    );
    const reconDx = tf.abs(
      tf.sub(
        tf.slice(recon, [0, 0, 1, 0], [-1, -1, -1, -1]),
        tf.slice(recon, [0, 0, 0, 0], [-1, -1, recon.shape[2] - 1, -1]),
      ),
    );
    const targetDx = tf.abs(
      tf.sub(
        tf.slice(target, [0, 0, 1, 0], [-1, -1, -1, -1]),
        tf.slice(target, [0, 0, 0, 0], [-1, -1, target.shape[2] - 1, -1]),
      ),
    );
    const lossY = tf.losses.meanSquaredError(targetDy, reconDy);
    const lossX = tf.losses.meanSquaredError(targetDx, reconDx);
    return tf.add(lossY, lossX);
  });
}

/**
 * Structural Similarity (SSIM) Loss
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

/**
 * InfoNCE Contrastive Loss for multimodal alignment
 */
function contrastiveLoss(imageEmb, textEmb, temperature = 0.07) {
  return tf.tidy(() => {
    const normImg = tf.div(
      imageEmb,
      tf.maximum(tf.norm(imageEmb, 2, -1, true), 1e-8),
    );
    const normTxt = tf.div(
      textEmb,
      tf.maximum(tf.norm(textEmb, 2, -1, true), 1e-8),
    );
    const logits = tf.div(
      tf.matMul(normImg, normTxt, false, true),
      temperature,
    );
    const b = imageEmb.shape[0];
    const labels = tf.oneHot(tf.range(0, b, 1, "int32"), b);
    const lossI2T = tf.losses.softmaxCrossEntropy(labels, logits);
    const lossT2I = tf.losses.softmaxCrossEntropy(labels, tf.transpose(logits));
    return tf.div(tf.add(lossI2T, lossT2I), 2.0);
  });
}

/**
 * Clip gradients by L2 norm
 */
function clipGradients(grads, maxNorm = 1.0) {
  return tf.tidy(() => {
    let sumSq = tf.scalar(0);
    const keys = Object.keys(grads);
    for (const k of keys) {
      if (grads[k]) {
        sumSq = tf.add(sumSq, tf.sum(tf.square(grads[k])));
      }
    }
    const totalNorm = tf.sqrt(sumSq);
    const scale = tf.minimum(
      tf.scalar(1.0),
      tf.div(tf.scalar(maxNorm), tf.add(totalNorm, 1e-8)),
    );

    const clipped = {};
    for (const k of keys) {
      if (grads[k]) {
        clipped[k] = tf.mul(grads[k], scale);
      }
    }
    return clipped;
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
  constructor(device = "webgpu") {
    this.device = device;
    // Initialize models
    this.vae = new LabelConditionedVAE();
    this.drift = new LabelConditionedDrift();

    // Anchor model for consistency
    this.vae_ref = null;

    // Optimizers
    this.opt_vae = tf.train.adam(CONFIG.LR || 0.0002);
    this.opt_drift = tf.train.adam(
      (CONFIG.LR || 0.0002) * (CONFIG.DRIFT_LR_MULTIPLIER || 0.5),
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

    // 3. Recursively check custom properties
    const keys = Object.keys(obj);
    for (const key of keys) {
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

  computeGradNorm(grads) {
    return tf.tidy(() => {
      let sumSq = tf.scalar(0);
      for (const k of Object.keys(grads)) {
        const g = grads[k];
        if (g) sumSq = tf.add(sumSq, tf.sum(tf.square(g)));
      }
      return Math.sqrt(sumSq.dataSync()[0]);
    });
  }

  _setEpochLrs() {
    const e = Math.min(this.epoch, (CONFIG.EPOCHS || 600) - 1);
    const etaFrac = 0.01;
    const decay =
      etaFrac +
      (1.0 - etaFrac) *
        0.5 *
        (1.0 + Math.cos((Math.PI * e) / (CONFIG.EPOCHS || 600)));
    const vaeFactor =
      this.phase === 1
        ? 1.0
        : this.phase === 2
          ? CONFIG.PHASE2_VAE_LR_FACTOR || 0.1
          : CONFIG.PHASE3_VAE_LR_FACTOR || 0.05;

    const vaeLr = (CONFIG.LR || 2e-4) * decay * vaeFactor;
    const driftLr =
      (CONFIG.LR || 2e-4) * (CONFIG.DRIFT_LR_MULTIPLIER || 0.5) * decay;

    if (this.opt_vae && this.opt_vae.learningRate !== undefined) {
      this.opt_vae.learningRate = vaeLr;
    }
    if (this.opt_drift && this.opt_drift.learningRate !== undefined) {
      this.opt_drift.learningRate = driftLr;
    }
  }

  async trainStep(batch, labels, textBytes = null) {
    this._setEpochLrs();

    const images = tf
      .tensor(batch)
      .reshape([-1, CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, 3]);

    const labelsArray = Array.isArray(labels) ? labels : [labels];
    const labelsTensor = tf.tensor(labelsArray, [labelsArray.length], "int32");

    let textBytesTensor = null;
    if (textBytes) {
      try {
        textBytesTensor = tf.tensor(
          textBytes,
          [textBytes.length, textBytes[0].length],
          "int32",
        );
      } catch (e) {
        console.error("❌ Failed to create textBytesTensor:", e.message);
        throw e;
      }
    }

    if (this.vae) {
      this.vae.currentEpoch = this.epoch;
    }

    try {
      if (this.phase === 1) {
        // --- Phase 1: VAE Training ---
        if (!this._vaeWarmed) {
          tf.tidy(() =>
            this.vae.forward(images, labelsTensor, textBytesTensor),
          );
          this._vaeWarmed = true;
        }
        const vaeVars = this.getVaeVariables();

        let metricsOut = {};
        const gradsObj = tf.variableGrads(() => {
          return tf.tidy(() => {
            const [recon, mu, logvar] = this.vae.forward(
              images,
              labelsTensor,
              textBytesTensor,
            );

            // 1. Reconstruction loss
            const rawL1 = tf.losses.absoluteDifference(images, recon);
            const reconLoss = tf.mul(rawL1, CONFIG.RECON_WEIGHT || 5.0);

            // 2. KL Divergence (with annealing)
            const klProgress = Math.min(
              1.0,
              this.epoch / (CONFIG.KL_ANNEALING_EPOCHS || 40),
            );
            const klLoss = tf.mul(
              klDivergenceSpatial(mu, logvar),
              (CONFIG.KL_WEIGHT || 0.004) * klProgress,
            );

            // 3. SSIM Loss
            let ssim = tf.scalar(0);
            if ((CONFIG.SSIM_WEIGHT || 0) > 0) {
              ssim = tf.mul(
                ssimLoss(images, recon),
                CONFIG.SSIM_WEIGHT || 3.0,
              );
            }

            // 4. Edge gradient loss
            let edge = tf.scalar(0);
            if ((CONFIG.EDGE_WEIGHT || 0) > 0) {
              edge = tf.mul(
                edgeLoss(recon, images),
                CONFIG.EDGE_WEIGHT || 0.5,
              );
            }

            // 5. Total variation loss
            let tv = tf.scalar(0);
            if ((CONFIG.TV_WEIGHT || 0) > 0) {
              tv = tf.mul(
                totalVariationLoss(recon),
                CONFIG.TV_WEIGHT || 0.01,
              );
            }

            // 6. Channel diversity loss
            const divLoss = tf.mul(
              this.vae._channelDiversityLoss(mu),
              CONFIG.DIVERSITY_WEIGHT || 0.7,
            );

            // 7. Multimodal contrastive alignment (if active)
            let cLoss = tf.scalar(0);
            if (
              CONFIG.USE_PROJECTION_HEADS &&
              this.vae.imageProj &&
              textBytesTensor
            ) {
              const textEmb = this.vae.getConditioning(
                labelsTensor,
                textBytesTensor,
              );
              const imgFlat = tf.reshape(mu, [mu.shape[0], -1]);
              const imgEmb = this.vae.imageProj.forward(imgFlat);
              cLoss = tf.mul(
                contrastiveLoss(
                  imgEmb,
                  textEmb,
                  CONFIG.CONTRASTIVE_TEMPERATURE || 0.07,
                ),
                CONFIG.CONTRASTIVE_WEIGHT || 0.1,
              );
            }

            const total = tf.add(
              tf.add(
                tf.add(tf.add(reconLoss, klLoss), ssim),
                tf.add(edge, tv),
              ),
              tf.add(divLoss, cLoss),
            );

            metricsOut = {
              recon: reconLoss.dataSync()[0],
              kl: klLoss.dataSync()[0],
              snr: calcSNR(images, recon),
            };

            return total;
          });
        }, vaeVars);

        const gradNorm = this.computeGradNorm(gradsObj.grads);
        const clippedGrads = clipGradients(
          gradsObj.grads,
          CONFIG.GRAD_CLIP || 1.0,
        );
        this.opt_vae.applyGradients(clippedGrads);

        const lossVal = gradsObj.value.dataSync()[0];
        tf.dispose(gradsObj.value);
        tf.dispose(gradsObj.grads);
        tf.dispose(clippedGrads);

        return {
          loss: lossVal,
          metrics: {
            phase: "vae",
            gradientNorm: gradNorm,
            ...metricsOut,
          },
        };
      } else {
        // --- Phase 2 & 3: Drift + Consistency / Joint Fine-tuning ---
        if (!this._driftWarmed) {
          tf.tidy(() => {
            const [mu_w] = this.vae.encode(
              images,
              labelsTensor,
              textBytesTensor,
            );
            const t_w = tf.randomUniform([images.shape[0], 1]);
            const z0_w = tf.randomNormal(mu_w.shape);
            this.drift.forward(z0_w, t_w, labelsTensor);
          });
          this._driftWarmed = true;
        }

        const driftVars = this.getDriftVariables();
        const vaeVars = this.getVaeVariables();

        // 1. Prepare Target Latents (z1, z0, t, zt, target) outside of drift backprop tape
        const temp =
          CONFIG.TEMPERATURE_START +
          (CONFIG.TEMPERATURE_END - CONFIG.TEMPERATURE_START) *
            (this.epoch / (CONFIG.EPOCHS || 600));

        const { z1, t, z0, zt, target, muRef } = tf.tidy(() => {
          const [muCurr, logvarCurr] = this.vae.encode(
            images,
            labelsTensor,
            textBytesTensor,
          );

          let muR = muCurr;
          if (this.vae_ref) {
            [muR] = this.vae_ref.encode(
              images,
              labelsTensor,
              textBytesTensor,
            );
          }

          const noise = tf.mul(
            tf.randomNormal(muCurr.shape),
            tf.mul(tf.exp(tf.mul(0.5, logvarCurr)), temp),
          );
          const _z1 = tf.add(muCurr, noise);
          const _t = tf.randomUniform([images.shape[0], 1]);
          const _z0 = tf.mul(
            tf.randomNormal(_z1.shape),
            CONFIG.CST_COEF_GAUSSIAN_PRIO || 0.8,
          );

          let _zt, _target;
          if (CONFIG.USE_OU_BRIDGE && this.ou_ref) {
            const [mean, var_] = this.ou_ref.bridgeSample(_z0, _z1, _t);
            _zt = tf.add(
              mean,
              tf.mul(
                tf.randomNormal(mean.shape),
                tf.sqrt(tf.add(var_, 1e-8)),
              ),
            );
            _target = this.ou_ref.bridgeVelocity(_z0, _z1, _t);
          } else {
            const t_bc = _t.reshape([-1, 1, 1, 1]);
            _zt = tf.add(tf.mul(tf.sub(1, t_bc), _z0), tf.mul(t_bc, _z1));
            _target = tf.sub(_z1, _z0);
          }

          return {
            z1: _z1,
            t: _t,
            z0: _z0,
            zt: _zt,
            target: _target,
            muRef: muR,
          };
        });

        // 2. CFG Label Dropout for Drift
        let trainLabels = labelsTensor;
        let trainText = textBytesTensor;
        if (Math.random() < (CONFIG.LABEL_DROPOUT_PROB || 0.1)) {
          trainLabels = tf.fill(
            labelsTensor.shape,
            (CONFIG.NUM_CLASSES || 11) - 1,
            "int32",
          );
          trainText = null;
        }

        // 3. Drift Optimization Step
        let driftLossVal = 0;
        const driftGradsObj = tf.variableGrads(() => {
          return tf.tidy(() => {
            const pred = this.drift.forward(zt, t, trainLabels, trainText);
            const t_bc = t.reshape([-1, 1, 1, 1]);
            const timeWeights = tf.add(
              1.0,
              tf.mul(CONFIG.TIME_WEIGHT_FACTOR || 3.0, t_bc),
            );
            const dLoss = tf.mul(
              huberLoss(
                tf.mul(target, timeWeights),
                tf.mul(pred, timeWeights),
              ),
              CONFIG.DRIFT_WEIGHT || 3.0,
            );
            return dLoss;
          });
        }, driftVars);

        const driftGradNorm = this.computeGradNorm(driftGradsObj.grads);
        const clippedDriftGrads = clipGradients(
          driftGradsObj.grads,
          (CONFIG.GRAD_CLIP || 1.0) * (CONFIG.DRIFT_GRAD_CLIP_FACTOR || 0.5) * 2,
        );
        this.opt_drift.applyGradients(clippedDriftGrads);
        driftLossVal = driftGradsObj.value.dataSync()[0];
        tf.dispose(driftGradsObj.value);
        tf.dispose(driftGradsObj.grads);
        tf.dispose(clippedDriftGrads);

        // 4. VAE Encoder & Consistency Step (Phase 2 & Phase 3)
        let vaeLossVal = 0;
        const vaeGradsObj = tf.variableGrads(() => {
          return tf.tidy(() => {
            const [muCurr] = this.vae.encode(
              images,
              labelsTensor,
              textBytesTensor,
            );

            // Consistency anchor loss
            const consistencyLoss = tf.mul(
              tf.losses.meanSquaredError(muCurr, muRef),
              CONFIG.CONSISTENCY_WEIGHT || 1.5,
            );

            // Channel diversity loss
            const divLoss = tf.mul(
              this.vae._channelDiversityLoss(muCurr),
              CONFIG.DIVERSITY_WEIGHT || 0.7,
            );

            // Global variance floor loss
            const muFloorLoss = this.vae.muFloorLoss(muCurr);

            let totalVae = tf.add(
              tf.add(consistencyLoss, divLoss),
              muFloorLoss,
            );

            // Phase 3: Joint Decoder Reconstruction Loss
            if (this.phase === 3) {
              const reconP3 = this.vae.decode(
                muCurr,
                labelsTensor,
                textBytesTensor,
              );
              const p3Recon = tf.mul(
                tf.losses.absoluteDifference(images, reconP3),
                (CONFIG.RECON_WEIGHT || 5.0) *
                  (CONFIG.PHASE3_RECON_SCALE || 0.5),
              );
              totalVae = tf.add(totalVae, p3Recon);
            }

            return totalVae;
          });
        }, vaeVars);

        const vaeGradNorm = this.computeGradNorm(vaeGradsObj.grads);
        const clippedVaeGrads = clipGradients(
          vaeGradsObj.grads,
          CONFIG.GRAD_CLIP || 1.0,
        );
        this.opt_vae.applyGradients(clippedVaeGrads);
        vaeLossVal = vaeGradsObj.value.dataSync()[0];
        tf.dispose(vaeGradsObj.value);
        tf.dispose(vaeGradsObj.grads);
        tf.dispose(clippedVaeGrads);

        // Dispose target latents
        tf.dispose([z1, t, z0, zt, target, muRef]);
        if (trainLabels !== labelsTensor) tf.dispose(trainLabels);

        return {
          loss: driftLossVal + vaeLossVal,
          metrics: {
            phase: this.phase === 2 ? "drift" : "both",
            driftLoss: driftLossVal,
            vaeLoss: vaeLossVal,
            gradientNorm: (driftGradNorm + vaeGradNorm) / 2,
          },
        };
      }
    } finally {
      tf.dispose([images, labelsTensor]);
      if (textBytesTensor) tf.dispose(textBytesTensor);
    }
  }

  async generateSamples(
    labels,
    count = 4,
    textBytes = null,
    options = {},
  ) {
    const selectedLabels = labels.slice(0, count);
    const numSamples = selectedLabels.length;
    const steps = options.steps || CONFIG.DEFAULT_STEPS || 50;
    const cfgScale =
      options.cfgScale !== undefined
        ? options.cfgScale
        : CONFIG.CFG_SCALE || 6.5;
    const method = options.method || "heun"; // 'euler', 'heun', or 'rk4'
    const langevinSteps =
      options.langevinSteps !== undefined
        ? options.langevinSteps
        : CONFIG.DEFAULT_LANGEVIN_STEPS || 0;
    const nullClass = (CONFIG.NUM_CLASSES || 11) - 1;

    const labelsTensor = tf.tensor(selectedLabels, [numSamples], "int32");
    const nullTensor = tf.fill([numSamples], nullClass, "int32");
    const textBytesTensor = textBytes
      ? tf.tensor(
          textBytes.slice(0, numSamples),
          [numSamples, textBytes[0].length],
          "int32",
        )
      : null;

    try {
      // 1. Start from prior standard deviation z0
      let z = tf.mul(
        tf.randomNormal([
          numSamples,
          CONFIG.LATENT_H || 12,
          CONFIG.LATENT_W || 12,
          CONFIG.LATENT_CHANNELS || 8,
        ]),
        CONFIG.CST_COEF_GAUSSIAN_PRIO || 0.8,
      );

      const dt = 1.0 / steps;

      // 2. Numerical ODE integration
      for (let i = 0; i < steps; i++) {
        const nextZ = tf.tidy(() => {
          const tCur = tf.fill([numSamples, 1], i * dt);

          const evalDrift = (zIn, tIn) => {
            const condDrift = this.drift.forward(
              zIn,
              tIn,
              labelsTensor,
              textBytesTensor,
            );
            if (cfgScale > 1.0) {
              const uncondDrift = this.drift.forward(
                zIn,
                tIn,
                nullTensor,
                null,
              );
              return tf.add(
                uncondDrift,
                tf.mul(tf.sub(condDrift, uncondDrift), cfgScale),
              );
            }
            return condDrift;
          };

          let zOut;
          if (method === "euler") {
            const k1 = evalDrift(z, tCur);
            zOut = tf.add(z, tf.mul(k1, dt));
          } else if (method === "rk4") {
            const k1 = evalDrift(z, tCur);
            const tHalf = tf.fill([numSamples, 1], (i + 0.5) * dt);
            const zHalf1 = tf.add(z, tf.mul(k1, 0.5 * dt));
            const k2 = evalDrift(zHalf1, tHalf);
            const zHalf2 = tf.add(z, tf.mul(k2, 0.5 * dt));
            const k3 = evalDrift(zHalf2, tHalf);
            const tNext = tf.fill([numSamples, 1], (i + 1) * dt);
            const zNext = tf.add(z, tf.mul(k3, dt));
            const k4 = evalDrift(zNext, tNext);

            const rkSum = tf.add(
              tf.add(k1, tf.mul(2.0, k2)),
              tf.add(tf.mul(2.0, k3), k4),
            );
            zOut = tf.add(z, tf.mul(rkSum, dt / 6.0));
          } else {
            // Heun (default)
            const k1 = evalDrift(z, tCur);
            const tNext = tf.fill([numSamples, 1], (i + 1) * dt);
            const zPred = tf.add(z, tf.mul(k1, dt));
            const k2 = evalDrift(zPred, tNext);
            zOut = tf.add(z, tf.mul(tf.add(k1, k2), dt / 2.0));
          }

          // Gentle ODE clamping
          const clampLimit = CONFIG.ODE_CLAMP_MAX || 10.0;
          return tf.clipByValue(zOut, -clampLimit, clampLimit);
        });

        z.dispose();
        z = nextZ;
      }

      // 3. Optional Langevin refinement at t=1
      if (langevinSteps > 0) {
        const stepSize = CONFIG.LANGEVIN_STEP_SIZE || 0.01;
        const scoreScale = CONFIG.LANGEVIN_SCORE_SCALE || 1.2;
        const tOne = tf.fill([numSamples, 1], 1.0);

        for (let s = 0; s < langevinSteps; s++) {
          const nextZ = tf.tidy(() => {
            const driftScore = this.drift.forward(
              z,
              tOne,
              labelsTensor,
              textBytesTensor,
            );
            const noise = tf.randomNormal(z.shape);
            const step = tf.mul(driftScore, stepSize * scoreScale);
            const noiseStep = tf.mul(noise, Math.sqrt(2 * stepSize));
            return tf.add(tf.add(z, step), noiseStep);
          });
          z.dispose();
          z = nextZ;
        }
        tOne.dispose();
      }

      // 4. Decode final trajectory end-state
      const decoded = tf.tidy(() =>
        this.vae.decode(z, labelsTensor, textBytesTensor),
      );
      const result = decoded.arraySync();
      decoded.dispose();
      z.dispose();
      return result;
    } finally {
      labelsTensor.dispose();
      nullTensor.dispose();
      if (textBytesTensor) textBytesTensor.dispose();
    }
  }

  async getCheckpoint() {
    return this.saveCheckpoint();
  }

  async getHashSample() {
    const vaeVars = this.getVaeVariables();
    if (vaeVars.length > 0) {
      return await vaeVars[0].array();
    }
    return [Math.random()];
  }

  async saveCheckpoint() {
    const vaeVars = this.getVaeVariables();
    const driftVars = this.getDriftVariables();

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
