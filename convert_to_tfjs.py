import torch
import numpy as np
import json
import os
import struct

def convert_pt_to_tfjs(pt_path='checkpoints/latest.pt', output_dir='public/models/tfjs_weights'):
    print(f"📦 Loading PyTorch checkpoint: {pt_path}")
    if not os.path.exists(pt_path):
        print(f"❌ Error: {pt_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint (CPU map for compatibility)
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    
    weights_manifest = []
    weight_data = bytearray()
    
    # We'll process VAE and Drift states
    states = {
        'vae': ckpt.get('vae_state', {}),
        'drift': ckpt.get('drift_state', {})
    }

    for model_name, state_dict in states.items():
        print(f"Processing {model_name} weights ({len(state_dict)} tensors)...")
        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
                
            # Convert to numpy
            arr = tensor.numpy()
            
            # --- IMPORTANT: Transpose PyTorch weights to TensorFlow format ---
            # PyTorch Conv2d: [out, in, h, w] -> TF.js: [h, w, in, out]
            if len(arr.shape) == 4:
                arr = arr.transpose(2, 3, 1, 0)
            # PyTorch Linear: [out, in] -> TF.js: [in, out]
            elif len(arr.shape) == 2:
                arr = arr.transpose(1, 0)
            
            # Flatten and prepare for binary storage
            arr_flat = arr.flatten().astype(np.float32)
            bytes_buffer = arr_flat.tobytes()
            
            # Update manifest
            weights_manifest.append({
                "name": f"{model_name}/{name.replace('.', '/')}",
                "shape": list(arr.shape),
                "dtype": "float32",
                "offset": len(weight_data),
                "length": len(bytes_buffer)
            })
            
            weight_data.extend(bytes_buffer)

    # Save manifest
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
        json.dump(weights_manifest, f, indent=2)
        
    # Save binary data
    with open(os.path.join(output_dir, 'weights.bin'), 'wb') as f:
        f.write(weight_data)

    print(f"✅ Conversion complete! Files saved to {output_dir}")
    print(f"📊 Total binary size: {len(weight_data) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    convert_pt_to_tfjs()
