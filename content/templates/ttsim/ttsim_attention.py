"""
ttsim_attention.py

Transformer self-attention forward pass running on the ttsim hardware simulator.
No model download required. Verifies output against a PyTorch reference using PCC.

Usage:
    export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export TT_METAL_DISABLE_SFPLOADMACRO=1
    python3 ttsim_attention.py
"""

import torch
import ttnn

SEQ_LEN = 32
D_HEAD = 64
SCALE = D_HEAD ** -0.5


def attention_pytorch(q, k, v):
    """Reference implementation in float32."""
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * SCALE
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v.float()).bfloat16()


def main():
    torch.manual_seed(42)

    q_pt = torch.randn(1, SEQ_LEN, D_HEAD, dtype=torch.bfloat16)
    k_pt = torch.randn(1, SEQ_LEN, D_HEAD, dtype=torch.bfloat16)
    v_pt = torch.randn(1, SEQ_LEN, D_HEAD, dtype=torch.bfloat16)

    ref = attention_pytorch(q_pt, k_pt, v_pt)

    device = ttnn.open_device(device_id=0)
    try:
        q = ttnn.from_torch(q_pt, layout=ttnn.TILE_LAYOUT, device=device)
        k = ttnn.from_torch(k_pt, layout=ttnn.TILE_LAYOUT, device=device)
        v = ttnn.from_torch(v_pt, layout=ttnn.TILE_LAYOUT, device=device)

        scores = ttnn.matmul(q, ttnn.permute(k, (0, 2, 1))) * SCALE
        attn = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(attn, v)

        result = ttnn.to_torch(ttnn.from_device(out))
    finally:
        ttnn.close_device(device)

    print(f"Attention output shape: {result.shape}")

    # PCC check
    pcc = torch.corrcoef(
        torch.stack([result.float().flatten(), ref.float().flatten()])
    )[0, 1].item()
    print(f"PCC vs PyTorch reference: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc:.6f}"
    print("PASSED")


if __name__ == "__main__":
    main()
