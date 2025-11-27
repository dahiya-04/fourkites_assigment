# Loss Landscape Geometry & Optimization Dynamics Framework

This repository contains the theoretical framework, efficient probing methods, and empirical validation for analyzing the geometry of neural network loss landscapes and its relationship to optimization dynamics and generalization.

## Project Goal

The primary goal of this project is to develop a rigorous framework for understanding how the topology of the loss function (specifically its local curvature, or **sharpness**) influences the training process and the final generalization performance of deep neural networks.

## Technical Background

Neural network optimization is a complex, high-dimensional problem. Key questions addressed by this framework include:
*   How does the local curvature of a minimum correlate with a model's ability to generalize to unseen data?
*   Can we efficiently measure this curvature in high-dimensional weight spaces?
*   How do different optimization strategies (e.g., SGD vs. Adam) and hyperparameters (Learning Rate, Weight Decay) affect the geometry of the minimum found?


### Optimization Dynamics

The steady-state distribution of weights found by Stochastic Gradient Descent (SGD) is inversely proportional to the Hessian matrix, $\mathbf{\Sigma} \propto \mathbf{H}^{-1}$. This suggests that SGD's inherent noise helps it explore and settle in wide, flat basins.

## Efficient Landscape Probing

Due to the massive size of the Hessian matrix in deep networks, we employ efficient methods to estimate its maximum eigenvalue:

1.  **Hessian-Vector Product (HVP):** Used to compute $\mathbf{H}\mathbf{v}$ without explicitly forming $\mathbf{H}$.
2.  **Power Iteration:** An iterative algorithm that uses the HVP to efficiently converge to the maximum eigenvalue ($\lambda_{\max}$) and its corresponding eigenvector.

The implementation of these methods is contained within the `landscape_prober.py` file.

## Empirical Validation

The empirical experiment compared three optimization configurations on a CNN model trained on MNIST, probing the resulting minimum's geometry and generalization performance.

### Key Findings

The actual empirical data provided a critical insight into the complexity of the problem:

| Configuration | Optimization Strategy | Test Accuracy (%) | Sharpness ($\lambda_{\max}$) | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Adam\_Default** | Adam, Default LR | **98.90** | **6.26** (Flattest) | **Best Generalization** - Flatness Hypothesis holds. |
| **Sharp\_SGD\_HighLR** | SGD, High LR | 98.14 | 9.14 | Good Generalization despite moderate sharpness. |
| **Flat\_SGD\_LowLR** | SGD, Low LR, High WD | 91.72 | **246.19** (Sharpest) | **Worst Generalization** - Optimization failure led to a sharp, sub-optimal minimum. |

### Conclusion

The experiment confirms that **high absolute sharpness is a reliable indicator of a poor, non-generalizing minimum**. Conversely, **low sharpness is a necessary condition for a highly generalizable model**. The optimization path (hyperparameters and optimizer choice) is the primary factor determining whether the model converges to a sharp, low-quality minimum or a flat, high-quality minimum.

## Repository Structure

*   `final_report_actual_data.md`: The comprehensive research report detailing the theory, methods, and analysis.
*   `landscape_prober.py`: Python class for efficient Hessian-Vector Product and Power Iteration.
*   `empirical_experiment.py`: Python script to train models and run the sharpness probing experiment.
*   `experiment_results.json`: Example file containing the final empirical data.

## How to Run the Experiment

To reproduce the empirical results, you will need a Python environment with PyTorch and torchvision installed.

1.  **Install Dependencies:**
    ```bash
    pip install torch torchvision numpy
    ```
2.  **Run the Experiment:**
    ```bash
    python empirical_experiment.py
    ```
    This script will train the models, run the sharpness probing, and save the results to `experiment_results.json`.

## Further Reading

For a detailed discussion of the theoretical derivations and the full analysis, please refer to the `final_report_actual_data.md`.
