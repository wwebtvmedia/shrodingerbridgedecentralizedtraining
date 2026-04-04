import { v4 as uuidv4 } from 'uuid';
import { ModelManager } from './models.js';

class SwarmTrainer {
    constructor(network, phaseManager) {
        this.id = uuidv4();
        this.network = network;
        this.phaseManager = phaseManager;
        this.modelManager = new ModelManager();
        
        // Training state
        this.currentEpoch = 0;
        this.currentPhase = 'auto';
        this.isTraining = false;
        this.explorationRate = 0.3;
        
        // Metrics
        this.lossHistory = [];
        this.metricsHistory = [];
        this.modelsEvaluated = 0;
        this.syncCount = 0;
        
        // Callbacks
        this.onEpochComplete = null;
        this.onModelShared = null;
        this.onModelAdopted = null;
        
        // Training interval
        this.trainingInterval = null;
    }
    
    async start() {
        if (this.isTraining) return;
        
        console.log('🚀 Starting swarm trainer...');
        this.isTraining = true;
        
        // Initialize model
        await this.modelManager.initialize();
        
        // Start training loop
        this.trainingLoop();
        
        // Start gossip for results
        this.startGossip();
    }
    
    async stop() {
        if (!this.isTraining) return;
        
        console.log('🛑 Stopping swarm trainer...');
        this.isTraining = false;
        
        // Clear intervals
        if (this.trainingInterval) {
            clearInterval(this.trainingInterval);
            this.trainingInterval = null;
        }
        
        // Save checkpoint
        await this.saveCheckpoint();
    }
    
    async trainingLoop() {
        while (this.isTraining) {
            // Determine phase
            if (this.currentPhase === 'auto') {
                this.currentPhase = this.phaseManager.determinePhase(this.currentEpoch);
            }
            
            // Train one epoch
            const { loss, metrics } = await this.trainEpoch();
            
            // Update history
            this.lossHistory.push({ epoch: this.currentEpoch, loss });
            this.metricsHistory.push({ epoch: this.currentEpoch, ...metrics });
            
            // Notify UI
            if (this.onEpochComplete) {
                this.onEpochComplete(this.currentEpoch, loss, metrics);
            }
            
            // Share results with swarm (gossip)
            await this.shareResults(loss, metrics);
            
            // Check for model synchronization
            await this.checkForSynchronization();
            
            // Increment epoch
            this.currentEpoch++;
            
            // Save checkpoint periodically
            if (this.currentEpoch % 10 === 0) {
                await this.saveCheckpoint();
            }
            
            // Generate samples periodically
            if (this.currentEpoch % 20 === 0) {
                await this.generateAndShareSamples();
            }
            
            // Small delay to prevent blocking
            await this.sleep(100);
        }
    }
    
    async trainEpoch() {
        const phaseParams = this.phaseManager.getPhaseParameters(this.currentPhase);
        
        // Simulate training (replace with actual WebTorch training)
        const loss = 0.5 + Math.random() * 0.3 * Math.exp(-this.currentEpoch / 100);
        const metrics = {
            diversity: 0.7 + Math.random() * 0.2,
            reconstruction: 0.3 + Math.random() * 0.2,
            kl_divergence: 0.1 + Math.random() * 0.05
        };
        
        // Apply phase-specific adjustments
        if (this.currentPhase === 'vae') {
            metrics.reconstruction *= 0.8; // Better reconstruction in VAE phase
        } else if (this.currentPhase === 'drift') {
            metrics.diversity *= 1.2; // Better diversity in drift phase
        }
        
        return { loss, metrics };
    }
    
    async shareResults(loss, metrics) {
        const result = {
            type: 'TRAINING_RESULT',
            peerId: this.id,
            epoch: this.currentEpoch,
            phase: this.currentPhase,
            loss,
            metrics,
            timestamp: Date.now(),
            modelHash: await this.modelManager.getModelHash()
        };
        
        // Share via gossip protocol
        await this.network.gossip(result);
        
        if (this.onModelShared) {
            this.onModelShared(result);
        }
    }
    
    async checkForSynchronization() {
        // Get best model from network
        const bestModel = await this.network.getBestModel();
        
        if (!bestModel) return;
        
        // Decide whether to synchronize
        const shouldSync = this.shouldSynchronize(bestModel);
        
        if (shouldSync) {
            await this.synchronizeTo(bestModel);
        }
    }
    
    shouldSynchronize(bestModel) {
        // Exploration vs exploitation
        if (Math.random() < this.explorationRate) {
            // Exploration: sometimes sync randomly
            return Math.random() < 0.3;
        }
        
        // Exploitation: sync only if significantly better
        const myLoss = this.lossHistory.length > 0 ? 
            this.lossHistory[this.lossHistory.length - 1].loss : 1.0;
        
        const improvement = (myLoss - bestModel.loss) / myLoss;
        return improvement > 0.15; // 15% improvement threshold
    }
    
    async synchronizeTo(bestModel) {
        console.log(`🔄 Synchronizing to model from ${bestModel.peerId}`);
        
        // Request model from peer
        const modelData = await this.network.requestModel(bestModel.peerId, bestModel.modelHash);
        
        if (!modelData) {
            console.warn('Failed to get model from peer');
            return;
        }
        
        // Load the model
        await this.modelManager.loadModel(modelData);
        
        // Random epoch jump (0 to bestModel.epoch)
        const randomEpoch = Math.floor(Math.random() * bestModel.epoch);
        this.currentEpoch = randomEpoch;
        
        // Update metrics
        this.syncCount++;
        this.modelsEvaluated++;
        
        // Notify
        if (this.onModelAdopted) {
            this.onModelAdopted(bestModel.peerId, randomEpoch);
        }
        
        console.log(`✅ Synchronized to epoch ${randomEpoch}`);
    }
    
    evaluateIncomingModel(modelData) {
        this.modelsEvaluated++;
        
        // Simple evaluation: compare loss
        const myLoss = this.lossHistory.length > 0 ? 
            this.lossHistory[this.lossHistory.length - 1].loss : Infinity;
        
        if (modelData.loss < myLoss * 0.9) {
            // 10% better, consider adopting
            if (Math.random() < 0.5) { // 50% chance to adopt
                this.synchronizeTo(modelData).catch(console.error);
            }
        }
    }
    
    async generateSamples(count = 4) {
        // Generate sample images (simulated for now)
        const samples = [];
        for (let i = 0; i < count; i++) {
            // Create a simple colored canvas
            const canvas = document.createElement('canvas');
            canvas.width = 64;
            canvas.height = 64;
            const ctx = canvas.getContext('2d');
            
            // Generate random color
            const hue = Math.random() * 360;
            ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
            ctx.fillRect(0, 0, 64, 64);
            
            // Add some random shapes
            ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            for (let j = 0; j < 5; j++) {
                const x = Math.random() * 64;
                const y = Math.random() * 64;
                const size = 5 + Math.random() * 15;
                ctx.beginPath();
                ctx.arc(x, y, size, 0, Math.PI * 2);
                ctx.fill();
            }
            
            samples.push(canvas.toDataURL());
        }
        
        return samples;
    }
    
    async generateAndShareSamples() {
        const samples = await this.generateSamples(2);
        
        // Share samples with network
        await this.network.shareSamples({
            peerId: this.id,
            epoch: this.currentEpoch,
            samples,
            timestamp: Date.now()
        });
    }
    
    async saveCheckpoint() {
        const checkpoint = {
            epoch: this.currentEpoch,
            phase: this.currentPhase,
            lossHistory: this.lossHistory,
            metricsHistory: this.metricsHistory,
            modelState: await this.modelManager.getState(),
            timestamp: Date.now()
        };
        
        // Save to IndexedDB
        await this.saveToStorage('checkpoint', checkpoint);
        
        console.log(`💾 Checkpoint saved at epoch ${this.currentEpoch}`);
    }
    
    async loadCheckpoint() {
        const checkpoint = await this.loadFromStorage('checkpoint');
        
        if (checkpoint) {
            this.currentEpoch = checkpoint.epoch;
            this.currentPhase = checkpoint.phase;
            this.lossHistory = checkpoint.lossHistory || [];
            this.metricsHistory = checkpoint.metricsHistory || [];
            
            if (checkpoint.modelState) {
                await this.modelManager.setState(checkpoint.modelState);
            }
            
            console.log(`📂 Checkpoint loaded from epoch ${this.currentEpoch}`);
            return true;
        }
        
        return false;
    }
    
    startGossip() {
        // Start periodic gossip
        setInterval(async () => {
            if (this.isTraining && this.network.peers.size > 0) {
                const latestResult = this.lossHistory.length > 0 ? 
                    this.lossHistory[this.lossHistory.length - 1] : null;
                
                if (latestResult) {
                    await this.shareResults(latestResult.loss, 
                        this.metricsHistory[this.metricsHistory.length - 1] || {});
                }
            }
        }, 5000); // Every 5 seconds
    }
    
    setPhase(phase) {
        if (['vae', 'drift', 'both', 'auto'].includes(phase)) {
            this.currentPhase = phase;
        }
    }
    
    setExplorationRate(rate) {
        this.explorationRate = Math.max(0, Math.min(1, rate));
    }
    
    // Utility methods
    async saveToStorage(key, value) {
        try {
            localStorage.setItem(`swarm_${key}`, JSON.stringify(value));
        } catch (error) {
            console.warn('Failed to save to storage:', error);
        }
    }
    
    async loadFromStorage(key) {
        try {
            const data = localStorage.getItem(`swarm_${key}`);
            return data ? JSON.parse(data) : null;
        } catch (error) {
            console.warn('Failed to load from storage:', error);
            return null;
        }
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

export { SwarmTrainer };