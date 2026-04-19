#!/usr/bin/env python3
"""
Inspect the latest.pt PyTorch checkpoint and convert to web-friendly format.
Based on ../enhancedoptimaltransport/read_ptrfile.py
"""

import torch
import json
import sys
from pathlib import Path

def inspect_checkpoint(path='checkpoints/latest.pt'):
    """Inspect a PyTorch checkpoint and print its structure."""
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"❌ Error: File not found at {path}")
        return None

    print(f"🔍 --- Inspecting Checkpoint: {path} ---\n")

    try:
        # Using weights_only=False to allow loading the 'defaultdict' in kpi_metrics
        # map_location='cpu' ensures we can read it even without a GPU
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        
        # 1. Basic Metadata
        print("📊 [ METADATA ]")
        print(f"  - Current Epoch:      {ckpt.get('epoch', 'N/A')}")
        print(f"  - Total Steps:        {ckpt.get('step', 'N/A')}")
        print(f"  - Training Phase:     {ckpt.get('phase', 'N/A')}")
        
        best_loss = ckpt.get('best_loss', 'N/A')
        if isinstance(best_loss, float):
            print(f"  - Best Loss:          {best_loss:.6f}")
        else:
            print(f"  - Best Loss:          {best_loss}")
            
        print(f"  - Best Composite:     {ckpt.get('best_composite_score', 'N/A')}")
        
        # 2. Config Snapshot
        if 'config' in ckpt:
            print("\n⚙️ [ SAVED CONFIG ]")
            for k, v in ckpt['config'].items():
                print(f"  - {k:15}: {v}")

        # 3. Model Weights Summary
        print("\n🧠 [ MODEL WEIGHTS ]")
        vae_weights = ckpt.get('vae_state', {})
        drift_weights = ckpt.get('drift_state', {})
        print(f"  - VAE layers:         {len(vae_weights)}")
        print(f"  - Drift layers:       {len(drift_weights)}")
        
        if 'vae_ref_state' in ckpt:
            print(f"  - VAE Anchor layers:  {len(ckpt['vae_ref_state'])}")

        # 4. Metrics History
        if 'kpi_metrics' in ckpt:
            print("\n📈 [ KPI METRICS ]")
            metrics = ckpt['kpi_metrics']
            for m_name in sorted(metrics.keys()):
                data = metrics[m_name]
                if isinstance(data, list):
                    count = len(data)
                    last = data[-1] if count > 0 else 'N/A'
                    if isinstance(last, float):
                        print(f"  - {m_name:18} ({count:4} pts) Last: {last:.4f}")
                    else:
                        print(f"  - {m_name:18} ({count:4} pts) Last: {last}")
                else:
                    print(f"  - {m_name:18}: {data}")

        # 5. Optimizer and Scheduler Status
        print("\n🔄 [ OPTIMIZERS & SCHEDULERS ]")
        opt_vae = "✅ Present" if 'opt_vae_state' in ckpt else "❌ Missing"
        opt_drift = "✅ Present" if 'opt_drift_state' in ckpt else "❌ Missing"
        sch_vae = "✅ Present" if 'scheduler_vae_state' in ckpt else "❌ Missing"
        sch_drift = "✅ Present" if 'scheduler_drift_state' in ckpt else "❌ Missing"
        
        print(f"  - VAE Optimizer:      {opt_vae}")
        print(f"  - Drift Optimizer:    {opt_drift}")
        print(f"  - VAE Scheduler:      {sch_vae}")
        print(f"  - Drift Scheduler:    {sch_drift}")

        print("\n✅ --- Inspection Complete ---")
        return ckpt

    except Exception as e:
        print(f"❌ Failed to read checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_to_web_format(ckpt, output_path='checkpoint_web.json'):
    """Convert PyTorch checkpoint to a web-friendly JSON format."""
    if ckpt is None:
        return
    
    # Extract only the model weights (simplified for prototype)
    web_checkpoint = {
        'metadata': {
            'epoch': ckpt.get('epoch', 0),
            'step': ckpt.get('step', 0),
            'phase': ckpt.get('phase', 1),
            'best_loss': ckpt.get('best_loss', float('inf')),
            'best_composite_score': ckpt.get('best_composite_score', float('-inf'))
        },
        'config': ckpt.get('config', {}),
        'note': 'This is a simplified version of the PyTorch checkpoint for web usage. Actual weights are not included in this prototype.'
    }
    
    # Ensure directory exists
    from pathlib import Path
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(web_checkpoint, f, indent=2)
    
    print(f"✅ Web checkpoint saved to {output_path}")
    return web_checkpoint

if __name__ == "__main__":
    # Inspect the checkpoint
    checkpoint = inspect_checkpoint('latest.pt')
    
    # Convert to web format
    if checkpoint is not None:
        convert_to_web_format(checkpoint, 'public/models/checkpoint_web.json')
        print("\n📝 Note: The 'latest.pt' file is a PyTorch checkpoint from the enhanced optimal transport training.")
        print("   It can be used as a starting point for swarm training when real WebTorch integration is implemented.")