// Torch-js training implementation for Schrödinger Bridge
// Based on ../enhancedoptimaltransport/training.py

import { CONFIG, setTrainingPhase, klDivergenceSpatial, calcSNR } from '../config.js';
import { LabelConditionedVAE, LabelConditionedDrift } from './models.js';

// OU Reference Process
class OUReference {
    constructor(theta = 1.0, sigma = Math.sqrt(2)) {
        this.theta = theta;
        this.sigma = sigma;
    }
    
    stationaryVariance() {
        return this.sigma ** 2 / (2 * this.theta);
    }
    
    transitionKernel(z0, t, dt) {
        const exp_neg_theta_dt = Math.exp(-this.theta * dt);
        const mean = z0 * exp_neg_theta_dt;
        const var_ = (this.sigma ** 2 / (2 * this.theta)) * (1 - exp_neg_theta_dt ** 2);
        return [mean, var_];
    }
    
    bridgeSample(z0, z1, t) {
        const exp_neg_theta_t = Math.exp(-this.theta * t);
        const exp_neg_theta_1_t = Math.exp(-this.theta * (1 - t));
        const exp_neg_theta = Math.exp(-this.theta);
        
        // Mean
        const numerator = (exp_neg_theta_t * (1 - exp_neg_theta_1_t ** 2) * z0 + 
                         (1 - exp_neg_theta_t ** 2) * exp_neg_theta_1_t * z1);
        const denominator = 1 - exp_neg_theta ** 2;
        const mean = numerator / denominator;
        
        // Variance
        let var_ = (this.sigma ** 2 / (2 * this.theta)) * 
                  ((1 - exp_neg_theta_t ** 2) * (1 - exp_neg_theta_1_t ** 2)) / (1 - exp_neg_theta ** 2);
        var_ = Math.max(0, var_);
        
        return [mean, var_];
    }
}

// KPI Tracker
class KPITracker {
    constructor(windowSize = 100) {
        this.windowSize = windowSize;
        this.metrics = new Map();
        this.compositeScores = [];
    }
    
    update(metricsDict) {
        for (const [key, value] of Object.entries(metricsDict)) {
            if (value !== null && value !== undefined) {
                if (!this.metrics.has(key)) {
                    this.metrics.set(key, []);
                }
                const values = this.metrics.get(key);
                values.push(value);
                if (values.length > this.windowSize) {
                    values.shift();
                }
            }
        }
        
        // Track composite score if available
        if (metricsDict.composite_score !== undefined) {
            this.compositeScores.push(metricsDict.composite_score);
            if (this.compositeScores.length > this.windowSize) {
                this.compositeScores.shift();
            }
        }
    }
    
    computeConvergence() {
        const convergence = {};
        
        for (const [metricName, values] of this.metrics.entries()) {
            if (values.length >= 10) {
                const window = Math.min(20, values.length);
                const ma = values.slice(-window).reduce((a, b) => a + b, 0) / window;
                convergence[`${metricName}_ma`] = ma;
                
                const std = values.length >= window ? 
                    Math.sqrt(values.slice(-window).reduce((sum, val) => sum + (val - ma) ** 2, 0) / window) : 0;
                convergence[`${metricName}_std`] = std;
                
                if (values.length >= 5) {
                    const recent = values.slice(-10);
                    const x = recent.map((_, i) => i);
                    const y = recent;
                    const slope = this.calculateSlope(x, y);
                    convergence[`${metricName}_trend`] = slope;
                }
            }
        }
        
        if (this.metrics.has('loss') && this.metrics.get('loss').length >= 10) {
            const lossValues = this.metrics.get('loss').slice(-20);
            const lossStd = Math.sqrt(lossValues.reduce((sum, val) => {
                const mean = lossValues.reduce((a, b) => a + b, 0) / lossValues.length;
                return sum + (val - mean) ** 2;
            }, 0) / lossValues.length);
            convergence.convergence_score = 1.0 / (1.0 + lossStd);
        }
        
        return convergence;
    }
    
    calculateSlope(x, y) {
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        
        const numerator = n * sumXY - sumX * sumY;
        const denominator = n * sumX2 - sumX * sumX;
        
        return denominator !== 0 ? numerator / denominator : 0;
    }
}

// Enhanced Label Trainer
export class EnhancedLabelTrainer {
    constructor(loader) {
        this.loader = loader;
        
        // Initialize models
        this.vae = new LabelConditionedVAE();
        this.drift = new LabelConditionedDrift();
        
        // Optimizers
        this.opt_vae = {
            parameters: this.vae.parameters(),
            lr: CONFIG.LR,
            weight_decay: CONFIG.WEIGHT_DECAY
        };
        
        this.opt_drift = {
            parameters: this.drift.parameters(),
            lr: CONFIG.LR * CONFIG.DRIFT_LR_MULTIPLIER,
            weight_decay: CONFIG.WEIGHT_DECAY
        };
        
        // KPI tracking
        this.kpi_tracker = new KPITracker(CONFIG.KPI_WINDOW_SIZE);
        
        // Training state
        this.epoch = 0;
        this.step = 0;
        this.phase = 1;
        this.best_loss = Infinity;
        this.best_composite_score = -Infinity;
        
        // Reference for Phase 2
        this.vae_ref = null;
        
        // OU reference process
        this.ou_ref = CONFIG.USE_OU_BRIDGE ? new OUReference(CONFIG.OU_THETA, CONFIG.OU_SIGMA) : null;
        
        console.log('💓 Enhanced Label Trainer initialized');
        console.log(`  Phase: ${this.phase}, Epoch: ${this.epoch}`);
    }
    
    _switchToPhase(newPhase) {
        console.log(`🔄 Switching to phase ${newPhase}`);
        
        if (newPhase === 1) {
            // Phase 1: VAE only
            this.vae.training = true;
            this.drift.training = false;
            this.vae_ref = null;
            
        } else if (newPhase === 2) {
            // Phase 2: Drift only, freeze decoder
            this.vae_ref = new LabelConditionedVAE();
            // Copy weights from current VAE
            this.vae_ref.loadStateDict(this.vae.stateDict());
            this.vae_ref.training = false;
            
            // Freeze decoder parameters
            for (const param of this.vae.parameters()) {
                if (param.name.includes('dec_')) {
                    param.requires_grad = false;
                }
            }
            
            this.vae.training = true;
            this.drift.training = true;
            this.phase2_start_epoch = this.epoch;
            
        } else if (newPhase === 3) {
            // Phase 3: Both trainable
            // Unfreeze all VAE parameters
            for (const param of this.vae.parameters()) {
                param.requires_grad = true;
            }
            
            // Ensure reference anchor exists
            if (!this.vae_ref) {
                this.vae_ref = new LabelConditionedVAE();
                this.vae_ref.loadStateDict(this.vae.stateDict());
                this.vae_ref.training = false;
            }
            
            this.vae.training = true;
            this.drift.training = true;
        }
        
        this.phase = newPhase;
    }
    
    getTrainingPhase(epoch) {
        const phase = setTrainingPhase(epoch);
        if (phase !== this.phase) {
            this._switchToPhase(phase);
        }
        return phase;
    }
    
    computeLoss(batch, phase = 1, batchIdx = 0) {
        // Extract images and labels
        let images, labels, source_id;
        
        if (batch && typeof batch === 'object') {
            images = batch.image;
            labels = batch.label;
            source_id = batch.source_id || null;
        } else if (Array.isArray(batch) && batch.length >= 2) {
            images = batch[0];
            labels = batch[1];
            source_id = null;
        } else {
            throw new Error(`Unexpected batch type: ${typeof batch}`);
        }
        
        if (phase === 1) {
            // Phase 1: Train VAE
            const [recon, mu, logvar] = this.vae.forward(images, labels, source_id);
            
            // Compute metrics
            const latent_std = logvar.mul(0.5).exp().mean().data[0];
            
            // Adaptive KL weight
            const raw_l1 = images.sub(recon).abs().mean();
            const raw_mse = images.sub(recon).pow(2).mean();
            const raw_kl = klDivergenceSpatial(mu, logvar);
            
            let current_kl_weight = CONFIG.KL_WEIGHT;
            if (latent_std < 0.3) {
                current_kl_weight = CONFIG.KL_WEIGHT * 10.0;
            }
            
            const diversity_loss = this.vae.diversity_loss || 0;
            const kl_annealing = Math.min(1.0, this.epoch / CONFIG.KL_ANNEALING_EPOCHS);
            
            const kl_loss = raw_kl * current_kl_weight * kl_annealing;
            const recon_loss = raw_l1 * CONFIG.RECON_WEIGHT;
            
            // Add edge preservation loss
            let edge_loss = 0;
            if (CONFIG.EDGE_WEIGHT > 0) {
                const grad_x_recon = recon.slice([0, 0, 0, 1], [recon.shape[0], recon.shape[1], recon.shape[2], recon.shape[3] - 1])
                    .sub(recon.slice([0, 0, 0, 0], [recon.shape[0], recon.shape[1], recon.shape[2], recon.shape[3] - 1])).abs().mean();
                const grad_y_recon = recon.slice([0, 0, 1, 0], [recon.shape[0], recon.shape[1], recon.shape[2] - 1, recon.shape[3]])
                    .sub(recon.slice([0, 0, 0, 0], [recon.shape[0], recon.shape[1], recon.shape[2] - 1, recon.shape[3]])).abs().mean();
                const edge_strength = (grad_x_recon + grad_y_recon) / 2;
                
                const grad_x_real = images.slice([0, 0, 0, 1], [images.shape[0], images.shape[1], images.shape[2], images.shape[3] - 1])
                    .sub(images.slice([0, 0, 0, 0], [images.shape[0], images.shape[1], images.shape[2], images.shape[3] - 1])).abs().mean();
                const grad_y_real = images.slice([0, 0, 1, 0], [images.shape[0], images.shape[1], images.shape[2] - 1, images.shape[3]])
                    .sub(images.slice([0, 0, 0, 0], [images.shape[0], images.shape[1], images.shape[2] - 1, images.shape[3]])).abs().mean();
                const real_edge_strength = (grad_x_real + grad_y_real) / 2;
                
                edge_loss = (edge_strength - real_edge_strength).pow(2).mul(CONFIG.EDGE_WEIGHT);
                recon_loss = recon_loss.add(edge_loss);
            }
            
            const total_loss = recon_loss.add(kl_loss).add(diversity_loss * CONFIG.DIVERSITY_WEIGHT);
            const snr = calcSNR(images, recon);
            
            return {
                total: total_loss,
                recon: recon_loss.data[0],
                kl: kl_loss.data[0],
                diversity: diversity_loss,
                edge_loss: edge_loss.data[0] || 0,
                snr: snr,
                raw_mse: raw_mse.data[0],
                raw_kl: raw_kl.data[0],
                latent_std: latent_std
            };
            
        } else {
            // Phase 2 or 3: Drift training
            // Ensure vae_ref exists
            if (!this.vae_ref) {
                this.vae_ref = new LabelConditionedVAE();
                this.vae_ref.loadStateDict(this.vae.stateDict());
                this.vae_ref.training = false;
            }
            
            // Get latent representations
            const mu_ref = this.vae_ref.encode(images, labels, source_id)[0];
            const [mu, logvar] = this.vae.encode(images, labels, source_id);
            
            const consistency_loss = mu.sub(mu_ref).pow(2).mean();
            
            // Temperature annealing
            const temperature = CONFIG.TEMPERATURE_START + 
                (CONFIG.TEMPERATURE_END - CONFIG.TEMPERATURE_START) * (this.epoch / CONFIG.EPOCHS);
            
            // Sample z1 with noise
            const z1_noise = logvar.mul(0.5).exp().mul(Math.random() * temperature);
            const z1 = mu.add(z1_noise);
            
            // Sample time
            let t;
            if (this.epoch > CONFIG.TRAINING_SCHEDULE.switch_epoch + CONFIG.EPOCHS / 6) {
                // Beta distribution
                t = Array.from({length: images.shape[0]}, () => {
                    const a = 2, b = 2;
                    const u = Math.random(), v = Math.random();
                    return Math.log(u / (1 - u)) / a < Math.log(v / (1 - v)) / b ? 
                        u ** (1 / a) / (u ** (1 / a) + v ** (1 / b)) : 
                        v ** (1 / b) / (u ** (1 / a) + v ** (1 / b));
                });
            } else {
                t = Array.from({length: images.shape[0]}, () => Math.random());
            }
            
            // Start from noise
            const z0 = Array(z1.shape).fill(0).map(() => Math.random() * CONFIG.CST_COEF_GAUSSIAN_PRIO);
            
            // Sample intermediate latent
            let zt, target;
            if (CONFIG.USE_OU_BRIDGE && this.ou_ref) {
                const [mean, var_] = this.ou_ref.bridgeSample(z0, z1, t);
                zt = mean + Math.sqrt(var_ + 1e-8) * Math.random();
                target = this.ou_ref.bridgeVelocity(z0, z1, t);
            } else {
                const t_reshaped = t.map(val => [val, val, val, val]);
                zt = z0.map((z0i, i) => (1 - t[i]) * z0i + t[i] * z1[i]);
                target = z1.map((z1i, i) => z1i - z0[i]);
            }
            
            // Classifier-Free Guidance
            let train_labels = labels;
            if (this.drift.training && Math.random() < CONFIG.LABEL_DROPOUT_PROB) {
                train_labels = labels.map(() => 0);
            }
            
            const pred = this.drift.forward(zt, t, train_labels, source_id);
            
            // Time-weighted loss
            const t_reshaped = t.map(val => [val, val, val, val]);
            const time_weights = t_reshaped.map(t_val => 1.0 + CONFIG.TIME_WEIGHT_FACTOR * t_val);
            
            const drift_loss_base = pred.mul(time_weights).sub(target.mul(time_weights)).abs().mean().mul(CONFIG.DRIFT_WEIGHT);
            
            const drift_start_epoch = this.phase2_start_epoch || CONFIG.TRAINING_SCHEDULE.switch_epoch;
            const consistency_decay = Math.max(0.1, 1.0 - (this.epoch - drift_start_epoch) / (CONFIG.EPOCHS - drift_start_epoch));
            
            if (phase === 3) {
                // Phase 3: Also train VAE reconstruction
                const recon_p3 = this.vae.decode(mu, labels, source_id);
                const recon_loss_p3 = images.sub(recon_p3).abs().mean().mul(CONFIG.RECON_WEIGHT * CONFIG.PHASE3_RECON_SCALE);
                
                const total_loss = drift_loss_base.add(consistency_loss.mul(CONFIG.CONSISTENCY_WEIGHT * consistency_decay)).add(recon_loss_p3);
                
                return {
                    total: total_loss,
                    drift: drift_loss_base.data[0],
-e 
                    consistency: consistency_loss.data[0] * CONFIG.CONSISTENCY_WEIGHT * consistency_decay,
                    recon_p3: recon_loss_p3.data[0]
                };
            } else {
                // Phase 2
                const total_loss = drift_loss_base.add(consistency_loss.mul(CONFIG.CONSISTENCY_WEIGHT * consistency_decay));
                
                return {
                    total: total_loss,
                    drift: drift_loss_base.data[0],
                    consistency: consistency_loss.data[0] * CONFIG.CONSISTENCY_WEIGHT * consistency_decay
                };
            }
        }
    }
    
    async trainStep(batch, batchIdx = 0) {
        const phase = this.getTrainingPhase(this.epoch);
        const loss_dict = this.computeLoss(batch, phase, batchIdx);
        
        // Backward pass
        loss_dict.total.backward();
        
        // Optimizer step
        if (phase === 1 || phase === 3) {
            // Update VAE optimizer
            for (const param of this.opt_vae.parameters) {
                if (param.requires_grad) {
                    param.data = param.data.sub(param.grad.mul(this.opt_vae.lr));
                    // Weight decay
                    if (this.opt_vae.weight_decay > 0) {
                        param.data = param.data.mul(1 - this.opt_vae.weight_decay);
                    }
                }
            }
        }
        
        if (phase === 2 || phase === 3) {
            // Update drift optimizer
            for (const param of this.opt_drift.parameters) {
                if (param.requires_grad) {
                    param.data = param.data.sub(param.grad.mul(this.opt_drift.lr));
                    // Weight decay
                    if (this.opt_drift.weight_decay > 0) {
                        param.data = param.data.mul(1 - this.opt_drift.weight_decay);
                    }
                }
            }
        }
        
        // Zero gradients
        this.vae.zeroGrad();
        this.drift.zeroGrad();
        
        // Update KPI tracker
        this.kpi_tracker.update(loss_dict);
        
        this.step++;
        return loss_dict;
    }
    
    async trainEpoch() {
        console.log();
        
        let total_loss = 0;
        let batch_count = 0;
        
        // Simulated training loop
        for (let i = 0; i < 10; i++) { // Reduced for testing
            const batch = this.loader.getBatch();
            if (!batch) break;
            
            const loss_dict = await this.trainStep(batch, i);
            total_loss += loss_dict.total.data[0];
            batch_count++;
            
            if (i % 5 === 0) {
                console.log();
            }
        }
        
        const avg_loss = total_loss / batch_count;
        console.log();
        
        this.epoch++;
        return avg_loss;
    }
    
    async train(num_epochs = CONFIG.EPOCHS) {
        console.log();
        
        for (let epoch = 0; epoch < num_epochs; epoch++) {
            await this.trainEpoch();
            
            // Check for early stopping
            if (this.kpi_tracker.shouldStop()) {
                console.log('🛑 Early stopping triggered');
                break;
            }
        }
        
        console.log('🏁 Training completed');
        return this.getCheckpoint();
    }
    
    getCheckpoint() {
        return {
            vae_state: this.vae.stateDict(),
            drift_state: this.drift.stateDict(),
            optimizer_vae: this.opt_vae,
            optimizer_drift: this.opt_drift,
            epoch: this.epoch,
            phase: this.phase,
            metrics: this.kpi_tracker.computeConvergence()
        };
    }
    
    loadCheckpoint(checkpoint) {
        this.vae.loadStateDict(checkpoint.vae_state);
        this.drift.loadStateDict(checkpoint.drift_state);
        this.epoch = checkpoint.epoch;
        this.phase = checkpoint.phase;
        console.log();
    }
}

// Export the trainer
export default EnhancedLabelTrainer;
