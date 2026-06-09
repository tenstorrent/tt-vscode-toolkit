"""
ttsim_n300_mesh.py

Demonstrates N300 (2-chip Wormhole) mesh simulation with libttsim_wh_x2.so.
Runs an element-wise add across a 1x2 MeshDevice — the same API used on real N300
hardware, no silicon required.

This example shows the core pattern for multi-chip TTNN workloads:
  1. Open a MeshDevice spanning N chips
  2. Shard a tensor across all chips with ShardTensorToMesh
  3. Apply a TTNN op — it dispatches to all chips in parallel
  4. Read back and concatenate results with ConcatMeshToTensor

Usage:
    export TT_METAL_SIMULATOR=~/sim/libttsim_wh_x2.so
    export TT_METAL_MOCK_CLUSTER_DESC_PATH=~/tt-metal/tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/n300_cluster_desc.yaml
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export TT_METAL_DISABLE_SFPLOADMACRO=1
    python3 ttsim_n300_mesh.py
"""

import torch
import ttnn

# Tensor shape: sharded across 2 chips on dim=0 → each chip sees (32, 64)
ROWS = 64
COLS = 64


def main():
    torch.manual_seed(0)

    # Full tensor that will be split across both chips
    a_full = torch.randn(ROWS, COLS, dtype=torch.bfloat16)
    b_full = torch.randn(ROWS, COLS, dtype=torch.bfloat16)
    expected = a_full + b_full

    # Open a 1×2 mesh — two Wormhole chips connected via Ethernet (N300 topology)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
    print(f"Opened mesh: {mesh}")  # MeshDevice(1x2 grid, 2 devices)

    try:
        # Shard each tensor: top half → chip 0, bottom half → chip 1
        a = ttnn.from_torch(
            a_full,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        b = ttnn.from_torch(
            b_full,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )

        # Element-wise add dispatches to both chips simultaneously
        c = ttnn.add(a, b)

        # Read back and reconstruct the full tensor
        c_full = ttnn.to_torch(c, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

    finally:
        ttnn.close_mesh_device(mesh)

    # bfloat16 has ~0.4% relative error (1 ulp at magnitude 1); max abs error ~0.1 is normal
    max_err = (c_full.float() - expected.float()).abs().max().item()
    ok = max_err < 0.1
    print(f"Max error vs reference: {max_err:.6f}")
    print(f"{'✅ PASS' if ok else '❌ FAIL'} — N300 mesh add ({ROWS}x{COLS}, sharded across 2 chips)")


if __name__ == "__main__":
    main()
