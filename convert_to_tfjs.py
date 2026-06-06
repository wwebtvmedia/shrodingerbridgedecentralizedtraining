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

    # Load checkpoint (CPU map for compatibility).
    # weights_only=True avoids unpickling/executing arbitrary code embedded in an
    # untrusted .pt file. We only read tensors from vae_state/drift_state below,
    # so the restricted loader is sufficient.
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=True)
    
    weights_manifest = []
    weight_data = bytearray()
    
    # We'll process VAE and Drift states
    states = {
        'vae': ckpt.get('vae_state', {}),
        'drift': ckpt.get('drift_state', {})
    }

    if not states['vae'] and not states['drift']:
        print("⚠️  Warning: checkpoint has no 'vae_state'/'drift_state' keys — "
              "nothing to convert. Check the checkpoint format.")
        return

    for model_name, state_dict in states.items():
        print(f"Processing {model_name} weights ({len(state_dict)} tensors)...")
        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue

            # Skip non-float bookkeeping tensors (e.g. BatchNorm num_batches_tracked
            # is int64 and must not be force-cast to a float weight).
            if not torch.is_floating_point(tensor):
                continue

            # Normalize dtype/device first: .numpy() fails for bf16/fp16-on-device
            # and grad-requiring tensors.
            arr = tensor.detach().to(torch.float32).cpu().numpy()

            # --- Transpose PyTorch weights to TensorFlow layout ---
            # NOTE: this rank-based heuristic assumes standard Conv2d/Linear.
            # nn.ConvTranspose2d ([in, out, h, w]), grouped/depthwise convs, and
            # Conv1d (rank 3) need different permutations — convert those by layer
            # type if/when the model uses them.
            if arr.ndim == 4:
                # Conv2d: [out, in, h, w] -> [h, w, in, out]
                arr = arr.transpose(2, 3, 1, 0)
            elif arr.ndim == 2:
                # Linear: [out, in] -> [in, out]
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
