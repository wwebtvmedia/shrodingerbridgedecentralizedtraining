// Configuration constants for Schrödinger Bridge training
// Adapted from ../enhancedoptimaltransport/config.py

import * as tf from "@tensorflow/tfjs";

export const CONFIG = {
  // Dataset configuration
  DATASET_NAME: "STL10",
  NUM_CLASSES: 11, // 10 real classes + 1 NULL class (Index 10) for CFG
  BATCH_SIZE: 16,
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

  // Loss weights (Updated for quality)
  KL_WEIGHT: 0.002,
  RECON_WEIGHT: 5.0,
  DRIFT_WEIGHT: 1.0,
  DIVERSITY_WEIGHT: 1.5,
  CONSISTENCY_WEIGHT: 1.0,
  PHASE3_RECON_SCALE: 0.8,
  PERCEPTUAL_WEIGHT: 1.5,
  SSIM_WEIGHT: 1.0,
  LPIPS_WEIGHT: 0.3,
  EDGE_WEIGHT: 0.3,
  TV_WEIGHT: 0.02,

  // VAE specific
  LATENT_SCALE: 1.0,
  FREE_BITS: 1.0,
  USE_NEURAL_TOKENIZER: false,
  USE_PROJECTION_HEADS: true,
  USE_FOURIER_FEATURES: false,
  USE_SUBPIXEL_CONV: true,
  USE_CONTEXT: true,
  DIVERSITY_ADAPTIVE: true,
  DIVERSITY_TARGET_START: 0.3,
  DIVERSITY_TARGET_END: 0.8,
  DIVERSITY_TARGET_STD: 0.8,
  DIVERSITY_MAX_STD: 2.0,
  DIVERSITY_LOW_PENALTY: 2.0,
  DIVERSITY_HIGH_PENALTY: 0.5,
  DIVERSITY_BALANCE_WEIGHT: 0.4,
  DIVERSITY_ADAPT_EPOCHS: 50,
  LOGVAR_CLAMP_MIN: -4,
  LOGVAR_CLAMP_MAX: 4,
  MU_NOISE_SCALE: 0.01,

  // Channel dropout
  CHANNEL_DROPOUT_PROB: 0.2,
  CHANNEL_DROPOUT_SURVIVAL: 0.8,

  // Classifier-Free Guidance
  LABEL_DROPOUT_PROB: 0.1,
  CFG_SCALE: 3.0,

  // Drift network specific
  DRIFT_LR_MULTIPLIER: 1.0,
  DRIFT_GRAD_CLIP_FACTOR: 1.0,
  PHASE2_VAE_LR_FACTOR: 0.1,
  PHASE3_VAE_LR_FACTOR: 0.05,

  // LoRA configuration
  USE_LORA: true,
  LORA_R: 8,
  LORA_ALPHA: 16,
  LORA_DROPOUT: 0.05,

  // Temperature annealing
  TEMPERATURE_START: 1.0,
  TEMPERATURE_END: 0.3,

  // Target noise for drift training
  DRIFT_TARGET_NOISE_SCALE: 0.01,

  // Time weighting factor
  TIME_WEIGHT_FACTOR: 2.0,

  // Enhanced features
  USE_PERCENTILE: true,
  USE_SNAPSHOTS: true,
  USE_KPI_TRACKING: true,
  TARGET_SNR: 30.0,
  SNAPSHOT_INTERVAL: 20,
  CHECKPOINT_INTERVAL: 5,
  SNAPSHOT_KEEP: 5,
  KPI_WINDOW_SIZE: 100,
  EARLY_STOP_PATIENCE: 15,

  // OU Bridge
  USE_OU_BRIDGE: false,
  OU_THETA: 1.0,
  OU_SIGMA: Math.sqrt(2),

  // Three-phase training schedule
  PHASE1_EPOCHS: Math.max(50, Math.floor(600 / 6)),
  PHASE2_EPOCHS: Math.max(50, Math.floor(600 / 2)),

  // Training schedule
  TRAINING_SCHEDULE: {
    mode: "auto",
    force_phase: null,
    custom_schedule: {},
    switch_epoch: 50,
    switch_epoch_1: Math.max(50, Math.floor(600 / 6)),
    switch_epoch_2: Math.max(50, Math.floor(600 / 2)),
    vae_epochs: Array.from({ length: 50 }, (_, i) => i),
    drift_epochs: Array.from({ length: 150 }, (_, i) => i + 50),
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
  const kl = tf.mul(0.5, tf.sub(tf.add(tf.exp(logvar), tf.square(mu)), tf.add(1, logvar)));
  const kl_sum = tf.sum(kl, [1, 2, 3]); // sum over spatial dimensions
  const kl_clamped = tf.maximum(kl_sum, CONFIG.FREE_BITS);
  return tf.mean(kl_clamped);
}

export function calcSNR(real, recon) {
  // Calculate Signal-to-Noise Ratio
  const mse = real.sub(recon).pow(2).mean();
  if (mse.data[0] === 0) return 100.0;
  return 10 * Math.log10(1.0 / (mse.data[0] + 1e-8));
}
