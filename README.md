
# Frank-Wolfe and KL Property for Optimization

This repository contains implementations and experimental scripts exploring the **Greedy Frank-Wolfe (GFW)** algorithm based on the **Kurdyka-Łojasiewicz (KL)** property. The focus is on two primary problem domains: **Max-Cut SDP** and **Reweighted L1 (RWL1)** minimization.

Contact aktasselimfatih@gmail.com for any feedback and questions.

## 📁 Repository Structure

- [**`max_cut/`**](./max_cut/): Optimization for the Maximum Cut problem.
    - [`matlab/`](./max_cut/matlab/): Implementations of BCM, ADMM-BM, RGD, and RTR. Use `comparison_final.m` to run the benchmark suite.
    - [`python/`](./max_cut/python/): GFW implementation for analysis of the shifting parameter. Includes visualization of MATLAB and Python outputs.
    - [`gpu/`](./max_cut/gpu/): Jupyter Notebooks for float16 and float32 precision testing via Google Colab.
        
-   [**`rwl1/`**](./rwl1/): Reweighted L1 minimization scripts.
    
    -   Includes noisy code simulations and result polishing scripts (`rwl1_polish_result.py`).
        

## 🚀 Getting Started

### Prerequisites

**Python Environment:**

-   Python 3.8+
    
-   NumPy, SciPy, Matplotlib, Pandas

-   Optionally; Pymanopt for CPU execution
    
-   For GPU experiments; CuPy, JAX, Pymanopt

-   For visualizations; simulation result of [LoRADS](https://github.com/COPT-Public/LoRADS) externally
    

**MATLAB Environment:**

-   Requires MATLAB (tested on R2023b or later).
    
-   [SketchyCGAL](https://github.com/alpyurtsever/SketchyCGAL), with minor adaptations as declared in the paper
    

### Running the Experiments

#### 1. Max-Cut Benchmark (MATLAB)

To compare different solvers (BCM, ADMM, RGD) for the Max-Cut problem:

1.  Navigate to `max_cut/matlab/`.
    
2.  Run the main script:
    
    Matlab
    
    ```
    comparison_final
    
    ```
    

#### 2. GPU Precision Testing (Python/Jupyter)

To reproduce the precision-based performance results (float16 vs. float32):

1.  Navigate to `max_cut/gpu/`.
    
2.  Open `maxcut_test_float16.ipynb` or `maxcut_test_float32.ipynb` in a Jupyter environment.
    
3.  Ensure your GPU drivers are configured for CuPy.
    

#### 3. Reweighted L1

To run the RWL1 experiments execute the main script:

Bash

```
python rwl1/rwl1_codes.py

```

## 📝 Note on Results

Numerical results and convergence plots are generated locally upon execution. This repository provides the source code and algorithm implementations to reproduce the experimental data discussed in the associated research.

## 📚 Citation
The research paper associated with this repository is currently under review. If you use this code in your work, please check back—a formal citation will be provided here upon publication.

