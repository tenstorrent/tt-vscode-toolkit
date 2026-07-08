---
id: lfs-04-block-and-model
title: "The Transformer Block & the Model"
description: >-
  Assemble MLP, RMSNorm, and residuals into a full transformer block, stack it
  into a nano GPT, and see TT-Lang kernels drop in as ttnn ops. Scale to 80M.
category: llm-from-scratch
tags: [transformer, mlp, rmsnorm, matmul, tt-lang]
supportedHardware: [n150, n300, t3k, p100, p150, p300c, galaxy, simulator]
status: draft
estimatedMinutes: 40
---

# The Transformer Block & the Model

<!-- Body authored in Task 7 -->
