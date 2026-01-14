# Research Attribution & Technical Foundation

## Core Research Papers

### 1. **JEPA (Joint Embedding Predictive Architecture)**
- **Authors**: Yann LeCun et al., Meta AI
- **Paper**: "A Path Towards Autonomous Machine Intelligence" (2022)
- **DOI**: Not formally published, available on OpenReview
- **Key Contribution**: Self-supervised learning through joint embedding spaces without generative modeling
- **How We Use It**: Our dual-encoder architecture (context encoder + target encoder) learns to map code and natural language queries into a shared embedding space

### 2. **VICReg (Variance-Invariance-Covariance Regularization)**
- **Authors**: Adrien Bardes, Jean Ponce, Yann LeCun (Meta AI, 2022)
- **Paper**: "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
- **Published**: ICLR 2022
- **ArXiv**: https://arxiv.org/abs/2105.04906
- **Key Contribution**: Prevents representation collapse in self-supervised learning
- **How We Use It**: Our loss function explicitly enforces:
  - **Variance**: Each code snippet has a unique embedding fingerprint
  - **Invariance**: Semantically similar code maps to nearby embeddings
  - **Covariance**: Prevents dimensional collapse (decorrelates embedding dimensions)

### 3. **CodeBERT**
- **Authors**: Microsoft Research
- **Paper**: "CodeBERT: A Pre-Trained Model for Programming and Natural Languages"
- **Published**: EMNLP 2020
- **ArXiv**: https://arxiv.org/abs/2002.08155
- **How We Use It**: Base encoder (RoBERTa-style) pre-trained on code, providing initial code understanding

### 4. **CodeSearchNet**
- **Authors**: GitHub, Microsoft Research
- **Paper**: "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search"
- **ArXiv**: https://arxiv.org/abs/1909.09436
- **Dataset**: 6M+ code-docstring pairs across 6 languages
- **How We Use It**: Training dataset (Python subset: 411K functions) and benchmark

## Novel Contributions

### 1. **JEPA for Code Understanding**
First application of LeCun's JEPA framework specifically for code-to-query semantic search. Unlike traditional approaches that use contrastive learning (SimCLR, MoCo), we use predictive alignment.

### 2. **VICReg + EMA Target**
Novel combination of VICReg regularization with Exponential Moving Average (EMA) target encoder updates, preventing mode collapse during training.

### 3. **Dual-Encoder with Asymmetric Updates**
- **Context Encoder** (trainable): Encodes code snippets
- **Target Encoder** (EMA-updated): Encodes natural language queries
- Prevents overfitting while maintaining semantic alignment

## Performance Benchmarks

| Metric | Repo-JEPA | Target | Improvement |
|--------|-----------|--------|-------------|
| **MRR** | 0.9052 | 0.60 | +50.8% |
| **Hits@1** | 86.2% | - | - |
| **Hits@5** | 95.9% | - | - |
| **Median Rank** | 1.0 | - | - |

## Hardware & Training

- **GPU**: NVIDIA H100 PCIe (80GB VRAM)
- **Training Time**: ~1.5 hours (10 epochs)
- **Dataset Size**: 411,000 Python functions
- **Framework**: PyTorch 2.6.0, Transformers 4.x
- **Optimization**: AdamW + OneCycleLR scheduler

## Acknowledgments

This work builds on foundational research from:
- **Meta AI** (JEPA, VICReg)
- **Microsoft Research** (CodeBERT, CodeSearchNet)
- **GitHub** (CodeSearchNet dataset)
- **NVIDIA** (H100 architecture enabling efficient training)

## Citation

If you use Repo-JEPA in your research, please cite:

```bibtex
@software{repojepa2026,
  author = {Uddeshya Kumar},
  title = {Repo-JEPA: Semantic Code Search via Joint Embedding Predictive Architecture},
  year = {2026},
  url = {https://github.com/uddeshya-23/Repo-JEPA},
  note = {Model: https://huggingface.co/uddeshya-k/RepoJepa}
}
```

## Future Work

1. **Multi-language Support**: Extend beyond Python to JavaScript, TypeScript, Go
2. **Fine-grained Search**: Function-level → statement-level semantic search
3. **Code Generation**: Reverse the architecture for natural language → code synthesis
4. **Continuous Learning**: Update embeddings as codebase evolves
