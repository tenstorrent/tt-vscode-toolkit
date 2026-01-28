# Mermaid.js Diagram Examples

The extension now supports [Mermaid.js](https://mermaid.js.org/) diagrams in lesson content!

## Usage

In any lesson markdown file, use a code fence with `mermaid` language:

````markdown
```mermaid
graph TD
    A[Start] --> B{Is it working?}
    B -->|Yes| C[Great!]
    B -->|No| D[Debug]
    D --> A
```
````

## Example Diagrams

### Flowchart - TT-Metal Stack
```mermaid
graph TB
    App[Your Application]
    TTNN[TTNN API Layer]
    TTMetal[TT-Metal Runtime]
    Hardware[Tenstorrent Hardware]

    App --> TTNN
    TTNN --> TTMetal
    TTMetal --> Hardware

    style App fill:#3293b2,color:#fff
    style TTNN fill:#5347a4,color:#fff
    style TTMetal fill:#499c8d,color:#fff
    style Hardware fill:#ffb71b,color:#000
```

### Sequence Diagram - Model Inference
```mermaid
sequenceDiagram
    participant User
    participant vLLM
    participant TTMetal
    participant Hardware

    User->>vLLM: Send prompt
    vLLM->>TTMetal: Forward pass
    TTMetal->>Hardware: Execute on Tensix cores
    Hardware-->>TTMetal: Return results
    TTMetal-->>vLLM: Aggregated output
    vLLM-->>User: Generated text
```

### Architecture Diagram
```mermaid
graph LR
    subgraph N150 Chip
        A[DRAM] --> B[NoC]
        B --> C[Tensix Core 1]
        B --> D[Tensix Core 2]
        B --> E[Tensix Core N]
    end
```

### State Diagram - Model Lifecycle
```mermaid
stateDiagram-v2
    [*] --> Downloaded: Download model
    Downloaded --> Loaded: Load to DRAM
    Loaded --> Ready: Compile kernels
    Ready --> Inference: Process tokens
    Inference --> Ready: Next token
    Ready --> [*]: Shutdown
```

## Styling

Mermaid diagrams automatically:
- Use dark theme to match VSCode
- Inherit font from VSCode settings
- Render centered with proper spacing
- Display in bordered containers

## Supported Diagram Types

- Flowchart / Graph
- Sequence Diagram
- Class Diagram
- State Diagram
- Entity Relationship Diagram
- Gantt Chart
- Pie Chart
- Git Graph
- And more! See [Mermaid docs](https://mermaid.js.org/intro/)
