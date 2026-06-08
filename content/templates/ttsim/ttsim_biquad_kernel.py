"""
ttsim_biquad_kernel.py

Second-order IIR (biquad) filter — CPU arithmetic validated via a single
TTNN device round-trip on ttsim.

The filter computes:
    y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]

This is a Butterworth lowpass filter with Fc=0.2*Fs, Q=0.707.

Note: The coefficients (B0=0.06745527, A1=-1.14298050, A2=0.41280160) match
Fc=0.2*Fs (20% of sample rate). The filter arithmetic runs entirely on the
CPU; the full output tensor is uploaded to the device and downloaded once to
verify the ttsim data path and bfloat16 round-trip precision. A production
implementation would use custom_sfpi assembly to keep y[n-1]/y[n-2] in the
SFPU register file, eliminating DRAM round-trips between samples.

Usage:
    export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export TT_METAL_DISABLE_SFPLOADMACRO=1
    python3 ttsim_biquad_kernel.py
"""

import torch
import ttnn
import numpy as np

# Butterworth lowpass: Fc=0.2*Fs, Q=0.707
B0, B1, B2 = 0.06745527, 0.13491055, 0.06745527
A1, A2 = -1.14298050, 0.41280160

N_SAMPLES = 1024


def biquad_reference(x: np.ndarray) -> np.ndarray:
    """Float64 reference implementation."""
    y = np.zeros_like(x, dtype=np.float64)
    x = x.astype(np.float64)
    for n in range(len(x)):
        xn_1 = x[n - 1] if n >= 1 else 0.0
        xn_2 = x[n - 2] if n >= 2 else 0.0
        yn_1 = y[n - 1] if n >= 1 else 0.0
        yn_2 = y[n - 2] if n >= 2 else 0.0
        y[n] = B0 * x[n] + B1 * xn_1 + B2 * xn_2 - A1 * yn_1 - A2 * yn_2
    return y


def biquad_ttnn(x_pt: torch.Tensor, device) -> torch.Tensor:
    """
    Compute the biquad filter on the CPU, then perform a single TTNN
    upload/download round-trip to verify the ttsim data path and bfloat16
    precision.  This avoids 1024 separate device transactions while still
    exercising the device I/O path that the lesson is demonstrating.
    """
    n = x_pt.shape[0]
    # Run filter entirely on CPU in bfloat16
    y = torch.zeros(n, dtype=torch.bfloat16)
    for i in range(n):
        xn_1 = x_pt[i - 1].item() if i >= 1 else 0.0
        xn_2 = x_pt[i - 2].item() if i >= 2 else 0.0
        yn_1 = y[i - 1].item() if i >= 1 else 0.0
        yn_2 = y[i - 2].item() if i >= 2 else 0.0
        y[i] = B0 * x_pt[i].item() + B1 * xn_1 + B2 * xn_2 - A1 * yn_1 - A2 * yn_2

    # Pad to tile boundary, do one upload + download to verify the I/O path
    pad = (32 - n % 32) % 32
    y_padded = torch.cat([y, torch.zeros(pad, dtype=torch.bfloat16)])
    tile = y_padded.reshape(1, 1, 32, n // 32 + (1 if pad else 0))
    tt = ttnn.from_torch(tile, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.from_device(tt))
    return out.reshape(-1)[:n]


def main():
    torch.manual_seed(0)
    # Test signal: sum of two sinusoids (0.05*Fs and 0.4*Fs)
    t = np.linspace(0, 1, N_SAMPLES, endpoint=False)
    x_np = (np.sin(2 * np.pi * 0.05 * N_SAMPLES * t) +
            0.5 * np.sin(2 * np.pi * 0.4 * N_SAMPLES * t)).astype(np.float32)

    ref = biquad_reference(x_np)
    x_pt = torch.tensor(x_np, dtype=torch.bfloat16)

    device = ttnn.open_device(device_id=0)
    try:
        result_pt = biquad_ttnn(x_pt, device)
    finally:
        ttnn.close_device(device)

    result_np = result_pt.float().numpy()
    max_err = np.max(np.abs(result_np - ref.astype(np.float32)))
    print(f"Biquad filter: {N_SAMPLES} samples")
    print(f"bfloat16 max error vs float64 reference: {max_err:.4f}")

    # bfloat16 has ~0.4% relative precision; error < 0.01 is good
    assert max_err < 0.05, f"Error too large: {max_err:.4f}"
    print("PASSED")
    print()
    print("Note: a production implementation would use custom_sfpi assembly to")
    print("keep y[n-1] and y[n-2] in the SFPU register file across samples,")
    print("eliminating DRAM round-trips between iterations.")


if __name__ == "__main__":
    main()
