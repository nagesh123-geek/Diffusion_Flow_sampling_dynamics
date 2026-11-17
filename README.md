# Diffusion vs Rectified Flow: Molecular Sampling Dynamics

This repository contains the implementation of **Diffusion Models** and **Rectified Flow Models** developed for the research project:

**"Diffusion vs Rectified-Flow Generative Models for Molecular Conformational Sampling: Architecture Dependence and Sampling Dynamics."**

The repository provides training and sampling implementations for both generative modeling frameworks, with three neural network architectures:

- Multi-Layer Perceptron (MLP)
- Residual Multi-Layer Perceptron (MLP with residual connections)
- Transformer

The implementations are developed and evaluated using molecular dynamics data from the Trp-cage protein.

---

## Research Publication

The work associated with this repository has been published in the *Journal of Chemical Theory and Computation (JCTC)*.

**Title:** Convergence Mechanisms of Generative Models in Molecular Conformational Sampling

**Authors:** Nagesh B. E. and Jagannath Mondal

**Journal:** Journal of Chemical Theory and Computation

**Volume:** 22, Issue 18, Pages 9343–9359 (2026)

**DOI:** [10.1021/acs.jctc.6c01009](https://doi.org/10.1021/acs.jctc.6c01009)

---

## Repository Structure

```text
Diffusion_Flow_sampling_dynamics/
│
├── trpcage_10p_psiPhi.npy
│
├── Diffusion_trpcage_training.py
├── diffusion_sampling.py
│
├── flow_trpcage_train.py
└── flow_sampling.py
```

---

## 1. Dataset

The repository provides the Trp-cage molecular dynamics dataset:

**File:** `trpcage_10p_psiPhi.npy`

- Number of samples: 50,000
- Number of features: 38
- Features: Backbone ψ–φ dihedral angles
- Training data: 10% of the available Trp-cage molecular dynamics data.

**Data shape:**

```python
(50000, 38)
```

---

## 2. Diffusion Models

The diffusion model implementation consists of training and sampling scripts.

### Training

**`Diffusion_trpcage_training.py`**
### Diffusion Model Training Configuration

The diffusion models are trained using the following hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Batch size | 4096 |
| Data dimension | 38 |
| Hidden dimension | 256 |
| Training epochs | 50,000 |
| Learning rate | 5 × 10⁻⁵ |
| Dropout | 0.2 |
| Diffusion timesteps (T) | 1000 |

These settings are used for training the MLP, Residual MLP, and Transformer architectures.

Implements diffusion model training using three neural network architectures:

1. MLP
2. MLP with residual connections
3. Transformer

### Sampling

**`diffusion_sampling.py`**

Performs molecular conformational sampling using the trained diffusion models.

The script generates samples from the learned molecular conformational distribution.

---

## 3. Rectified Flow Models

The repository also implements Rectified Flow models for molecular conformational sampling.

### Training

**`flow_trpcage_train.py`**
### Flow Model Training Configuration

The flow models are trained using the following hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Batch size | 4096 |
| Data dimension | 38 |
| Hidden dimension | 256 |
| Training epochs | 50,000 |
| Learning rate | 5 × 10⁻⁵ |
| Dropout | 0.2 |


These settings are used for training the MLP, Residual MLP, and Transformer architectures.

Implements Rectified Flow training using three neural network architectures:

1. MLP
2. MLP with residual connections
3. Transformer

### Sampling

**`flow_sampling.py`**

Performs molecular conformational sampling using trained Rectified Flow models.

---

## 4. Neural Network Architectures

Both generative modeling frameworks are implemented using the same three architectural families:

| Architecture | Description |
|---|---|
| MLP | Standard multilayer perceptron |
| Residual MLP | MLP incorporating residual connections |
| Transformer | Transformer-based architecture |

This enables a comparative investigation of the dependence of generative sampling behavior on neural network architecture.

---

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Diffusion_Flow_sampling_dynamics.git
cd Diffusion_Flow_sampling_dynamics
```

Install the required dependencies:

```bash
pip install numpy torch matplotlib scipy
```

Additional dependencies may be required depending on the implementation.

---

## 6. Research Scope

This repository supports the comparative study of diffusion and rectified flow generative models for molecular conformational sampling.

The primary research questions concern:

- Differences between diffusion and rectified flow sampling dynamics.
- Dependence of generative convergence mechanisms on neural network architecture.
- Molecular conformational distribution learning.
- Sampling efficiency and convergence behavior.
- Evolution of generated molecular conformations during the sampling process.

---

## Citation

If you use this code or build upon this research, please cite the following publication:

```bibtex
@article{10.1021/acs.jctc.6c01009,
    author = {B E, Nagesh and Mondal, Jagannath},
    title = {Convergence Mechanisms
of Generative Models in Molecular
Conformational Sampling},
    journal = {Journal of Chemical Theory and Computation},
    volume = {22},
    number = {18},
    pages = {9343-9359},
    year = {2026},
    month = {09},
    issn = {1549-9618},
    doi = {10.1021/acs.jctc.6c01009},
    url = {https://doi.org/10.1021/acs.jctc.6c01009},
    eprint = {https://pubs.acs.org/jctcce/article-pdf/22/18/9343/68401504/acs.jctc.6c01009.pdf},
}

```

---

## Acknowledgements

This work was conducted at the Tata Institute of Fundamental Research (TIFR), Hyderabad.
