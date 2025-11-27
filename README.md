# Loss Landscape Geometry & Optimization Dynamics Framework

This repository contains the theoretical framework, efficient probing methods, and empirical validation for analyzing the geometry of neural network loss landscapes and their relationship to optimization dynamics and generalization.

---

## Project Goal

The primary goal of this project is to develop a rigorous framework for understanding how the topology of the loss function—specifically its local curvature, or **sharpness**—influences:

- the training process  
- the type of minimum the model converges to  
- the final generalization performance of deep neural networks  

---

## Technical Background

Neural network optimization is a complex, high-dimensional problem.  
This framework addresses questions such as:

- How does the local curvature of a minimum correlate with a model's ability to generalize?
- Can we efficiently measure curvature in massive weight spaces?
- How do optimization strategies (e.g., SGD vs. Adam) and hyperparameters (LR, WD) influence the geometry of minima?

---

## Theoretical Framework

The framework uses second-order optimization theory, particularly the **Hessian matrix** **H**, which quantifies curvature around minima.

### Sharpness and Generalization

The **sharpness** of a minimum $ \mathbf{w}^* $ is governed by the maximum Hessian eigenvalue:

$$
\mathcal{S}(\mathbf{w}^*) \propto \lambda_{\max}(\mathbf{H}(\mathbf{w}^*))
$$

**Theoretical Result:**  
A smaller $ \lambda_{\max} $ (a flatter minimum) implies:

- lower generalization error  
- increased robustness to perturbations  

---

## Optimization Dynamics

SGD behaves like a stochastic process.  
The steady-state distribution of weights satisfies:

$$
\Sigma \propto \mathbf{H}^{-1}
$$

This means:

- SGD naturally prefers **flat minima**  
- Noise helps avoid sharp or poor-quality basins  

---

## Efficient Landscape Probing

Because computing the full Hessian is intractable in modern networks, we estimate sharpness using two methods:

1. **Hessian–Vector Products (HVPs)**  
   Allows computing $ \mathbf{H} \mathbf{v} $ efficiently without forming **H**.

2. **Power Iteration**  
   Repeated HVPs converge to:

   - the maximum Hessian eigenvalue $ \lambda_{\max} $  
   - its principal eigenvector  

These methods are implemented in:

- `landscape_prober.py`

---

## Empirical Validation

We train three models on MNIST using different optimization settings and then probe the geometry of the resulting minima.

### Key Findings

| Configuration | Optimization Strategy | Test Accuracy (%) | Sharpness $ \lambda_{\max} $ | Observation |
|--------------|------------------------|-------------------|-------------------------------|-------------|
| **Adam\_Default** | Adam, default LR | **98.90** | **6.26** (Flattest) | **Best generalization** — Flatness hypothesis holds |
| **Sharp\_SGD\_HighLR** | SGD, high LR | 98.14 | 9.14 | Good generalization despite moderate sharpness |
| **Flat\_SGD\_LowLR** | SGD, low LR, high WD | 91.72 | **246.19** (Sharpest) | **Worst generalization** — optimization failure |

### Interpretation

- **High sharpness = poor generalization** (always true in experiment)  
- **Low sharpness = necessary for good generalization** (but not sufficient)  
- Hyperparameters **dictate** which basin the model converges to  
- SGD can find **sharp** minima if configured poorly  

---

## Repository Structure

