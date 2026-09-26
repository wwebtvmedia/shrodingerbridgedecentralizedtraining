// Configuration constants for Schrödinger Bridge training
// Adapted from ../enhancedoptimaltransport/config.py

import * as tf from "@tensorflow/tfjs";

export const CONFIG = {
  // Dataset configuration
  DATASET_NAME: "STL10",
  NUM_CLASSES: 11, // 10 real classes + 1 NULL class (Index 10) for CFG
  BATCH_SIZE: 8,
  LABEL_EMB_DIM: 128,
  USE_CONTEXT: true,
  CONTEXT_DIM: 64,
  NUM_SOURCES: 2,
  IMG_SIZE: 96,
  GEN_SIZE: 96,
  LATENT_CHANNELS: 8,
  LATENT_H: 12, // IMG_SIZE // 8 (was 6)
  LATENT_W: 12,

  // Training hyperparameters
  LR: 2e-4,
  EPOCHS: 600,
  WEIGHT_DECAY: 1e-4,
  GRAD_CLIP: 1.0,

  // Loss weights (Aligned with enhancedoptimaltransport)
  KL_WEIGHT: 0.004,
  RECON_WEIGHT: 5.0,
  DRIFT_WEIGHT: 3.0,
  DIVERSITY_WEIGHT: 0.7,
  CONSISTENCY_WEIGHT: 1.5,
  PHASE3_RECON_SCALE: 0.5,
  PERCEPTUAL_WEIGHT: 2.0,
  SSIM_WEIGHT: 3.0,
  EDGE_WEIGHT: 0.5,
  TV_WEIGHT: 0.01,
  CONTRASTIVE_WEIGHT: 0.1,
  PHASE2_CONTRASTIVE_FACTOR: 1.0,
  CONTRASTIVE_TEMPERATURE: 0.07,

  // VAE specific (Aligned with enhancedoptimaltransport)
  LATENT_SCALE: 1.0,
  FREE_BITS: 2.0,
  USE_NEURAL_TOKENIZER: false,
  USE_PROJECTION_HEADS: true,
  USE_FOURIER_FEATURES: false,
  USE_SUBPIXEL_CONV: true,
  DIVERSITY_ADAPTIVE: true,
  DIVERSITY_TARGET_START: 0.3,
  DIVERSITY_TARGET_END: 0.8,
  DIVERSITY_MAX_STD: 2.0,
  DIVERSITY_LOW_PENALTY: 2.0,
  DIVERSITY_HIGH_PENALTY: 0.5,
  DIVERSITY_BALANCE_WEIGHT: 0.4,
  DIVERSITY_ADAPT_EPOCHS: 100,
  KL_ANNEALING_EPOCHS: 40,
  LOGVAR_CLAMP_MIN: -4,
  LOGVAR_CLAMP_MAX: 4,
  MU_NOISE_SCALE: 0.01,
  MU_STD_FLOOR: 0.84,
  MU_STD_FLOOR_WEIGHT: 20.0,
  CST_COEF_GAUSSIAN_PRIO: 0.8,

  // Channel dropout
  CHANNEL_DROPOUT_PROB: 0.2,
  CHANNEL_DROPOUT_SURVIVAL: 0.8,

  // Classifier-Free Guidance (CFG)
  LABEL_DROPOUT_PROB: 0.1,
  CFG_SCALE: 6.5,

  // Drift network specific
  DRIFT_LR_MULTIPLIER: 0.5,
  DRIFT_GRAD_CLIP_FACTOR: 0.5,
  PHASE2_VAE_LR_FACTOR: 0.1,
  PHASE3_VAE_LR_FACTOR: 0.05,

  // LoRA configuration
  USE_LORA: true,
  LORA_R: 8,
  LORA_ALPHA: 16,
  LORA_DROPOUT: 0.05,

  // Temperature annealing
  TEMPERATURE_START: 1.0,
  TEMPERATURE_END: 0.4,

  // Target noise for drift training
  DRIFT_TARGET_NOISE_SCALE: 0.01,

  // Time weighting factor
  TIME_WEIGHT_FACTOR: 3.0,

  // ODE / Inference numerics
  ODE_CLAMP_MAX: 10.0,
  DEFAULT_STEPS: 100,
  DEFAULT_SEED: 42,
  INFERENCE_TEMPERATURE: 0.4,
  DEFAULT_LANGEVIN_STEPS: 10,
  LANGEVIN_STEP_SIZE: 0.01,
  LANGEVIN_SCORE_SCALE: 1.2,

  // Enhanced features
  USE_PERCENTILE: true,
  USE_SNAPSHOTS: true,
  USE_KPI_TRACKING: true,
  TARGET_SNR: 30.0,
  SNAPSHOT_INTERVAL: 20,
  CHECKPOINT_INTERVAL: 10,
  SNAPSHOT_KEEP: 5,
  KPI_WINDOW_SIZE: 100,
  EARLY_STOP_PATIENCE: 15,

  // EMA
  USE_EMA: true,
  EMA_DECAY: 0.999,

  // OU Bridge
  USE_OU_BRIDGE: false,
  OU_THETA: 1.0,
  OU_SIGMA: Math.sqrt(2),

  // Three-phase training schedule
  PHASE1_EPOCHS: 150,
  PHASE2_EPOCHS: 400,

  // Training schedule
  TRAINING_SCHEDULE: {
    mode: "auto",
    force_phase: null,
    custom_schedule: {},
    switch_epoch: 150,
    switch_epoch_1: 150,
    switch_epoch_2: 400,
    alternate_freq: 5,
  },
};

// Helper functions
export function setTrainingPhase(epoch) {
  const mode = CONFIG.TRAINING_SCHEDULE.mode;

  if (mode === "manual") {
    return CONFIG.TRAINING_SCHEDULE.force_phase || 1;
  } else if (mode === "custom") {
    return CONFIG.TRAINING_SCHEDULE.custom_schedule[epoch] || 1;
  } else if (mode === "alternate") {
    const alt_freq = CONFIG.TRAINING_SCHEDULE.alternate_freq || 5;
    return Math.floor(epoch / alt_freq) % 2 === 0 ? 1 : 2;
  } else if (mode === "three_phase") {
    const e1 = CONFIG.TRAINING_SCHEDULE.switch_epoch_1;
    const e2 = CONFIG.TRAINING_SCHEDULE.switch_epoch_2;
    if (epoch < e1) return 1;
    else if (epoch < e2) return 2;
    else return 3;
  } else {
    // 'auto' mode
    return epoch < CONFIG.TRAINING_SCHEDULE.switch_epoch ? 1 : 2;
  }
}

export function klDivergenceSpatial(mu, logvar) {
  // KL divergence: 0.5 * (exp(logvar) + mu^2 - 1 - logvar)
  const kl = tf.mul(
    0.5,
    tf.sub(tf.add(tf.exp(logvar), tf.square(mu)), tf.add(1, logvar)),
  );
  const kl_sum = tf.sum(kl, [1, 2, 3]); // sum over spatial dimensions
  const kl_clamped = tf.maximum(kl_sum, CONFIG.FREE_BITS);
  return tf.mean(kl_clamped);
}

export function calcSNR(real, recon) {
  // Calculate Signal-to-Noise Ratio.
  // `.data` is an async method; we need the synchronous value here, so use
  // dataSync() and dispose the scalar to avoid leaking a tensor each call.
  const mse = real.sub(recon).pow(2).mean();
  const v = mse.dataSync()[0];
  mse.dispose();
  if (v === 0) return 100.0;
  return 10 * Math.log10(1.0 / (v + 1e-8));
}
