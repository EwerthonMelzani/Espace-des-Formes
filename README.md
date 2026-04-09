# Diffeomorphic Learning for Nonlinear Classification

## Overview

This repository explores the implementation of Diffeomorphic Learning, a mathematical framework within shape analysis and optimal transport that models an alternative approach to conventional Deep Neural Networks. Rather than relying on discrete linear mappings followed by non-linear activations, this architecture establishes a continuous, dynamic geometric flow that topologically transforms non-separable datasets into a linearly separable space.

## Mathematical and Geometric Foundations

Diffeomorphic Learning trains a smooth, time-dependent, and topology-preserving velocity field $v(t, \cdot)$. This continuous flow operator dynamically advects data points over a temporal domain, ensuring that the inherent topological structure remains intact without singularities or folding. Following the continuous transformation sequence, a canonical terminal classifier acts at the final time $t=1$.

The underlying architecture relies on two critical mathematical phenomena:

1. **Dimensional Lifting (The Dummy Feature):** Preserving structural boundaries severely limits traditional $2$-D spatial diffeomorphisms, producing inherent restrictions on crossing boundaries of enclosed objects. We resolve this by orthogonally injecting a single $(d+1)$ spatial parameter, effectively granting the deformation an extra degree of freedom to navigate structures across intersecting geometric topologies.
2. **Affine Subspace Injections ($g$):** Standard Reproducing Kernel Hilbert Space (RKHS) fields naturally vanish at spatial boundaries and lack expressivity for executing uniform scale, translation, and rotational transformations. Explicitly incorporating an affine linear group sequence mapping $g(t, x) = A(t)x + b(t)$ greatly mitigates optimization vanishing points globally.

## Codebase Implementation

We transition the exact theoretical structure of the continuous Pontryagin Maximum Principle (PMP) dynamics into a natively functional backend leveraging robust back-propagation dynamics over finite spatial distributions.

The structured framework within the `/code` directory contains the following implementations:
- **`code/Diffeomorphic_Learning_pytorch.ipynb`**: The fundamental layout of the framework built as `DiffeomorphicLearnerTorch(nn.Module)`. It addresses numerical topological evaluation constraints computing discrete Eulerian representations via dynamic unrolled autograd mechanics.
- **`code/generate_best_seed_visualizations.py`**: A computational workflow for iterating model configurations and generating results mapping transformation pathways specifically across non-linear testing scenarios (`circles`, `moons`, `blobs`, `gaussian_quantiles`).

## Presentation and Further Readings

Complementary rigorous mathematical analyses (detailing RKHS optimality criteria, affine energy constraints, computation optimizations around $\mathcal{O}(N^2)$ control variables, and finite state formulation) along with global comparative results, have been explicitly formulated.

View the compiled presentation analysis mapping out the complete theoretical boundaries here: **[`final_presentation.pdf`](final_presentation.pdf)**.
