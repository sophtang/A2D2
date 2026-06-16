# [A2D2: Fine-Tuning Any-Length Discrete Diffusion for Adaptive Decoding](https://arxiv.org/abs/2606.13565) 🃏🔮

[**Sophia Tang**](https://sophtang.github.io/), [**Yuchen Zhu**](https://yuchen-zhu-zyc.github.io/), [**Molei Tao**](https://mtao8.math.gatech.edu/), and [**Pranam Chatterjee**](https://www.chatterjeelab.com/)

<p>
  <a href="https://arxiv.org/abs/2606.13565"><img src="https://img.shields.io/badge/arXiv-6B67EE?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://sophtang.github.io/a2d2/"><img src="https://img.shields.io/badge/Project_Page-6B67EE?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiBmaWxsPSJ3aGl0ZSI+PHBhdGggZD0iTTEwLjUgMS41QzExIDcuMyAxMy4yIDkuNSAxOSAxMEMxMy4yIDEwLjUgMTEgMTIuNyAxMC41IDE4LjVDMTAgMTIuNyA3LjggMTAuNSAyIDEwQzcuOCA5LjUgMTAgNy4zIDEwLjUgMS41WiIvPjxwYXRoIGQ9Ik0xOC41IDEzLjVDMTguNyAxNS44IDE5LjcgMTYuOCAyMiAxN0MxOS43IDE3LjIgMTguNyAxOC4yIDE4LjUgMjAuNUMxOC4zIDE4LjIgMTcuMyAxNy4yIDE1IDE3QzE3LjMgMTYuOCAxOC4zIDE1LjggMTguNSAxMy41WiIvPjxwYXRoIGQ9Ik01IDE1LjVDNS4xMiAxNyA1LjUgMTcuMzggNyAxNy41QzUuNSAxNy42MiA1LjEyIDE4IDUgMTkuNUM0Ljg4IDE4IDQuNSAxNy42MiAzIDE3LjVDNC41IDE3LjM4IDQuODggMTcgNSAxNS41WiIvPjwvc3ZnPg==" alt="Project Page"></a>
</p>

![A2D2](assets/a2d2.gif)

This is the repository for the paper [**A2D2: Fine-Tuning Any-Length Discrete Diffusion for Adaptive Decoding**](https://arxiv.org/abs/2606.13565).

Masked discrete diffusion models (MDMs) offer a simple, stable likelihood-based framework for sequence generation, recently extended to **any-length** settings via token insertion. **A2D2** is a unified framework for reward-guided fine-tuning of any-length MDMs that **jointly optimizes the insertion and unmasking policies together with a quality-based inference schedule**, converging to the intractable reward-tilted distribution without requiring target samples.

🃏 We derive the **Radon–Nikodym derivative** for the joint insertion–unmasking path measures, enabling theoretically guaranteed convergence to the reward-tilted sequence distribution.

🃏 We establish **unmasking and insertion quality** as tractable approaches for minimizing decoding error (compounding parallelization error), and train lightweight quality predictors alongside the policy.

🃏 We introduce the **Adaptive Joint Decoding (AJD)** loss, which provably yields the optimal path measure that generates the reward-tilted distribution while remasking low-quality tokens and dropping low-quality insertions at inference.

🃏 Empirically, A2D2 improves reward optimization while enhancing generation **flexibility** and **accuracy** over prior fixed-length fine-tuning and inference-time guidance methods.

## Drug-Like Small Molecule Design 🧪

We pre-train an any-length MDM on the **SAFE** dataset ([Noutahi et al. 2024](https://arxiv.org/abs/2310.10773), ~950M molecules from ZINC and Unichem in SAFE notation) and fine-tune it with **A2D2** to optimize **QED** (drug-likeness) and **synthetic accessibility (SA)**. A2D2 jointly raises QED and lowers SA over the pre-trained baseline while increasing the fraction of valid, unique, drug-like, and synthesizable molecules. Code and instructions are in [`/a2d2_mol`](a2d2_mol).

## Multi-Objective Therapeutic Peptide Generation 💉

We pre-train an any-length **peptide SMILES** MDM on ~11M peptides (CycPeptMPDB, SmProt, CycloPs) and fine-tune with **A2D2** on five therapeutic properties simultaneously: **target-protein binding affinity, solubility, non-hemolysis, non-fouling, and permeability**. A2D2 outperforms inference-time multi-objective guidance and fixed-length off-policy RL fine-tuning on almost all objectives, while improving the fraction of valid peptides. Code and instructions are in [`/a2d2_pep`](a2d2_pep).

## Language Model Reasoning 🧠

We additionally apply **A2D2** to reward fine-tuning of any-length language MDMs (LLaDA / FlexMDM), optimizing math-reasoning correctness and format rewards (GSM8K / MATH), including infilling variants. Code is in [`/a2d2_language`](a2d2_language).

## Repository Structure

| Directory | Experiment |
|-----------|------------|
| [`a2d2_mol`](a2d2_mol) | Drug-like small molecule design (QED, SA) |
| [`a2d2_pep`](a2d2_pep) | Multi-objective therapeutic peptide generation |
| [`a2d2_language`](a2d2_language) | Language model reasoning reward fine-tuning (code soon) |
| [`lightning_modules`](lightning_modules) | Any-length insertion MDM Lightning modules (policy + quality predictors) |
| [`model`](model) | Shared model architecture |
| [`demo`](demo) | Quality-guided inference demo notebook |

Each experiment directory contains its own `README.md` with environment setup, pretrained weight placement, fine-tuning commands, and evaluation instructions.

## Citation

If you find this repository helpful for your publications, please consider citing our paper:

```python
@article{tang2026a2d2,
  title={A2D2: Fine-Tuning Any-Length Discrete Diffusion for Adaptive Decoding},
  author={Sophia Tang and Yuchen Zhu and Molei Tao and Pranam Chatterjee},
  journal={arXiv preprint arXiv:2606.13565},
  year={2026}
}
```

To use this repository, you agree to abide by the MIT License.