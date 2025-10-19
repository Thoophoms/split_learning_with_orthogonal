# Split Learning with Orthogonal Projection

This repository explores **Split Learning (SL)** enhanced by **Orthogonal Subspace Projection** — a technique inspired by *Federated Orthogonal Training (FOT)* — to improve knowledge retention and mitigate **catastrophic forgetting** in distributed and sequential learning environments.


## Overview

Traditional Split Learning separates the model between **client** and **server**, allowing distributed training without sharing raw data.  
However, SL often suffers from **gradient misalignment** and **forgetting** across rounds.

This project introduces **Orthogonal Projection** at the cut-layer level to align gradient updates within a shared subspace.  
By periodically updating the projection basis using **top-r Singular Value Decomposition (SVD)** of stored activations, the method maintains stability and improves convergence in non-IID client data.


## Architecture

- **Model:** ResNet-18 (client–server split)
- **Dataset:** CIFAR-10
- **Learning Setup:** Split Learning (SL)
- **Orthogonal Enhancement:** Projection matrices updated via top-r SVD  
- **Feature Memory:** Stores cut-layer activations per round for subspace update  
- **Implementation:** PyTorch
- **Hardware:** NVIDIA A100 (4 GPUs)


## Key Features

- Split Learning pipeline (client + server)  
- Orthogonal gradient projection module  
- Dynamic subspace update using SVD  
- Feature memory capacity (`FMC`) control  
- Logging for accuracy, loss, and runtime per round  
- Compatible with multi-GPU (DataParallel) setup
 

### Algorithm: FedProject Applied at the Cut Layer in Split Learning

```text
Algorithm 1: FedProject Applied at the Cut Layer in Split Learning

Initialize FEATURE_MEM ← [ ]
Initialize U_old ← None

for each global round do
    for each client do
        fx_c ← client_model(x)                   # Forward pass on client
        fx_s ← server_model(fx_c)                # Forward pass on server
        L ← CrossEntropy(fx_s, y)
        loss.backward()                          # Compute gradients incl. ∇(fx_c)L
        g ← fx_client.grad.detach()

        if U_old ≠ None then
            g ← g - (g * U_old) * U_oldᵀ         # Orthogonal projection step
        end if

        client.backward(g)
        client.optimizer.step()
    end for

    if FEATURE_MEM not empty then
        X ← concatenate(FEATURE_MEM)
        U_old ← top_r_svd(X)                     # Update projection subspace
    end if
end for
```

## Method Summary

1. During each round, activations (`fx_client`) from all clients are stored.  
2. At the end of the round, compute the **top-r singular vectors**:
   \[
   X = \text{concat}(\text{FEATURE\_MEM}), \quad U = \text{top\_r\_svd}(X)
   \]
3. Use `U` to **project gradients** before the backward pass:
   \[
   g' = U(U^T g)
   \]
4. Update `U` every round using the new feature memory.

This approach ensures that learning updates remain within the most informative subspace of past activations — reducing interference between tasks and stabilizing split learning.


## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Thoophoms/split_learning_with_orthogonal.git
cd split_learning_with_orthogonal

2. Set Up Environment

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Run Training

python train.py --epochs 10 --alpha 0.5 --dataset CIFAR10 --model resnet18
```

## Citation

If you use or reference this work, please cite:

> Thoop-hom Supannopaj, *Split Learning with Orthogonal Projection*, GitHub repository, 2025.  
> [https://github.com/Thoophoms/split_learning_with_orthogonal](https://github.com/Thoophoms/split_learning_with_orthogonal)


### Inspired by

> **Yavuz Faruk Bakman, Duygu Nur Yaldiz, Yahya H. Ezzeldin, Salman Avestimehr.**  
> *Federated Orthogonal Training: Mitigating Global Catastrophic Forgetting in Continual Federated Learning.*  
> *International Conference on Learning Representations (ICLR), 2024.*  
> [OpenReview link →](https://openreview.net/forum?id=nAs4LdaP9Y)

> **D. Nuryıldız, H. T. Kahraman, and M. K. Güllü.**  
> *Federated Orthogonal Training*, GitHub repository, 2023.  
> [https://github.com/duygunuryldz/Federated_Orthogonal_Training](https://github.com/duygunuryldz/Federated_Orthogonal_Training)

> **Jingtao Li.**  
> *Awesome Split Learning*, GitHub repository, 2023.  
> [https://github.com/zlijingtao/Awesome-Split-Learning](https://github.com/zlijingtao/Awesome-Split-Learning)


## Contact

Created by Thoop-hom Supannopaj
M.S. Computer Science @ Montclair State University
📧 [LinkedIn](https://www.linkedin.com/in/trisha-supannopaj/)
🌐 [Website](https://www.thoophoms.com)


⭐ If you find this project helpful, please consider giving it a star!
