# Tensix Grid Playground

Explore the Tenstorrent Tensix chip architecture interactively. Each scene below
animates a different concept — click **▶** to play or **⏭** to step through
frame by frame.

---

## Parallelism: Amdahl's Law in Action

Watch how adding more Tensix cores increases throughput. Even with a 10% serial
fraction, 64 cores yield 21× speedup. The colors show which cores are actively
computing vs. idle.

```tensix_viz arch=wormhole
[
  { "step": "highlight", "cores": [[1,1]], "color": "pink", "label": "1 core · 1.0x", "ms": 500 },
  { "step": "pause", "ms": 900 },
  { "step": "unhighlight" },
  { "step": "highlight", "cores": [[1,1],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1]], "color": "tensixActive", "label": "8 cores · 5.3x", "ms": 500 },
  { "step": "pause", "ms": 900 },
  { "step": "unhighlight" },
  { "step": "highlight", "cores": [[1,1],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1],[1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[8,2],[1,3],[2,3],[3,3],[4,3],[5,3],[6,3],[7,3],[8,3],[1,4],[2,4],[3,4],[4,4],[5,4],[6,4],[7,4],[8,4],[1,5],[2,5],[3,5],[4,5],[5,5],[6,5],[7,5],[8,5],[1,6],[2,6],[3,6],[4,6],[5,6],[6,6],[7,6],[8,6],[1,7],[2,7],[3,7],[4,7],[5,7],[6,7],[7,7],[8,7],[1,8],[2,8],[3,8],[4,8],[5,8],[6,8],[7,8],[8,8]], "color": "tensixActive", "label": "64 cores · 21.4x", "ms": 600 },
  { "step": "pause", "ms": 1500 },
  { "step": "clear" }
]
```

**Key insight:** Amdahl's Law: `Speedup = 1 / (S + P/N)` where S = serial fraction,
P = parallel fraction, N = number of processors. Even 10% serial work caps
maximum speedup at 10×.

---

## NOC Routing: Data Travels the Mesh

The Network-on-Chip (NOC) is a 2D torus mesh. Data tiles travel from source
to destination using row-first routing — horizontal first, then vertical.

```tensix_viz arch=wormhole
[
  { "step": "highlight", "cores": [[1,1]], "color": "teal",  "label": "source", "ms": 400 },
  { "step": "highlight", "cores": [[7,7]], "color": "pink",  "label": "dest",   "ms": 400 },
  { "step": "pause", "ms": 500 },
  { "step": "transfer", "from": [1,1], "to": [7,7], "ms": 1200 },
  { "step": "pause", "ms": 400 },
  { "step": "transfer", "from": [7,7], "to": [3,2], "ms": 900 },
  { "step": "pause", "ms": 400 },
  { "step": "transfer", "from": [3,2], "to": [6,5], "ms": 900 },
  { "step": "pause", "ms": 800 },
  { "step": "clear" }
]
```

**Key insight:** Two independent NOC planes (NOC0 and NOC1) allow simultaneous
bidirectional transfers — no contention for opposite-direction traffic.

---

## Kernel Dispatch: Compute + Data Movement Threads

Each Tensix core runs two independent RISC-V thread groups simultaneously:
**compute** (yellow) handles math operations while **data movement** (pink) handles
NOC reads/writes. They communicate via lock-free circular buffers (Dataflow Buffers).

```tensix_viz arch=wormhole
[
  { "step": "highlight", "cores": [[1,1],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1]], "color": "tensixActive", "label": "compute threads", "ms": 500 },
  { "step": "pause", "ms": 600 },
  { "step": "highlight", "cores": [[1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[8,2]], "color": "pink", "label": "data movement threads", "ms": 500 },
  { "step": "pause", "ms": 600 },
  { "step": "transfer", "from": [1,2], "to": [1,1], "ms": 500 },
  { "step": "transfer", "from": [4,2], "to": [4,1], "ms": 500 },
  { "step": "transfer", "from": [7,2], "to": [7,1], "ms": 500 },
  { "step": "pause", "ms": 600 },
  { "step": "transfer", "from": [3,1], "to": [6,1], "ms": 700 },
  { "step": "transfer", "from": [3,1], "to": [6,4], "ms": 900 },
  { "step": "pause", "ms": 1000 },
  { "step": "clear" }
]
```

---

## Blackhole Architecture

The P100/P150/P300c (Blackhole) chip has a wider grid than Wormhole.
The same programming model applies — compute cores in the center,
DRAM controllers on the edges, Ethernet links for multi-chip connectivity.

```tensix_viz arch=blackhole
[
  { "step": "highlight", "cores": [[1,1],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1],[8,1],[9,1],[10,1],[11,1],[12,1],[13,1],[14,1],[15,1]], "color": "tensixActive", "label": "15 compute cols", "ms": 600 },
  { "step": "pause", "ms": 800 },
  { "step": "unhighlight" },
  { "step": "transfer", "from": [1,5], "to": [15,5], "ms": 1200 },
  { "step": "transfer", "from": [8,1], "to": [8,10], "ms": 1200 },
  { "step": "pause", "ms": 1000 },
  { "step": "clear" }
]
```

---

## How to Use the Visualizer in Your Own Lessons

Add a `tensix_viz` code block to any lesson markdown:

````markdown
```tensix_viz arch=wormhole
[
  { "step": "highlight", "cores": [[1,1]], "label": "Start here" },
  { "step": "transfer", "from": [1,1], "to": [4,4] },
  { "step": "pause", "ms": 500 }
]
```
````

**Available step types:**

| Step | Parameters | Description |
|------|-----------|-------------|
| `highlight` | `cores`, `color`, `label`, `ms` | Pulse-animate a set of cores |
| `unhighlight` | `cores` (optional) | Remove highlights |
| `transfer` | `from`, `to`, `ms` | Animate a tile packet along the NOC |
| `heatmap` | `data` (2D array), `ms` | Per-core utilization overlay |
| `label` | `core`, `text` | Place a persistent label on a core |
| `clear` | `what` (optional) | Clear highlights / labels / heatmap |
| `pause` | `ms` | Wait before next step |

**Color names:** `tensixActive` (teal), `pink`, `teal`, `gold`, `green`, `red`

**Architectures:** `wormhole` (8×8 compute grid), `blackhole` (15×10 compute grid)
