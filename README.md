# SmolVLA-RGB-D-Extended

A modification of the SmolVLA model to support robot 3D perception via cross-attention between RGB and depth maps (RGB-D).

## Architecture
- **Queries (Q):** Original SmolVLM visual tokens (RGB, top view) compressed via Pixel Shuffle (64 tokens, 960 dim).
- **Keys / Values ​​(K/V):** 1-channel depth map processed through a SigLIP-compliant Conv2D layer (16x16 patch) and similar Pixel Shuffle to preserve accurate spatial coordinates (8x8 grid).
- **Merge:** `nn.MultiheadAttention` for injecting geometric features into the main VLM transformer pipeline.
