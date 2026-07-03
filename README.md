# Positional Convolution Experts (*Work In Progress*)
## Intro & Novelty
The goal of this project is to implement a ResNet-MoE architecture with a new type of router. Positional Convolution Expert (PCE) hase two principal novelty : 
1. My architecture is the first complete ResNet-MoE developed, more paper have developed different MoE-CNN but nobody has developed a complete ResNet-MoE with a large scale training on different dataset. Their study is concentrate on training dynamics analisys. 
2. My router-gate uses a Bi-Linear form to ensure an interaction between positional-features and semantic-features. The Bi-Linear form ensures that position and semantic can interact, and the expert's choice depends on their interaction.
In addition to the architecture novelty i have also introduced a new auxiliary loss that is called *Decorrelation Loss*, it is used to force the decorrelation of experts when the tokens have the same position but different semantics pattern.

## Router-Gate
The Router-Gate has two independent branches : 
- **Semantic Branch :** Semantic branch extracts a semantic representation. Semantic representation helps to decide which expert is more suitable for a specific pattern inside a patch at a specific position. This is important because patches in the same position can contain different semantic patterns.
- **Position Branch :** Positional branch uses Fourier Features to extract a positional representation from the the Fourier codification. 

Each branches returns its own logits, then Router-Gate merges pos-logits and sem-logits with a bi-linear form: why bi-linear form and not sum-form ? Because through bi-linear form the Router-Gate can choose the experts who agree with semantic and positional interaction, making sure that the experts can be specialized on semantic and positional marges pattern, with sum-form don't interact among them, and the gradient does not depend on the other.
After the model computes the int-logits through bi-linear form, the model executes the expert-choice algorithm to route each token to experts and experts process their assigned token.
After each experts have processed its assigned tokens through convolutional layers, we reconstruct the entire feature map using the outputs of the experts. We also add a residual connection, so that dropped tokens can preserve their original identity. 
After the reconstruction we apply a dense branch that mixes the expert outputs in order to reduce discontinuities between patches.

Each experts is composed of :

> Conv → GN → SiLU → Conv → GN + Residual Connection

**More details on the architecture and training :** [Architecture and Training detail latex documentation]{docs/latex_docs/main.pdf} (*Work In Progress*)

## What i m working on :
- **Comparison with baselines** : The baselines are ResNet, ViT and MoE-ViT with the same number of parameters 
- **Developing of loss to ensure a contigous routing** : This loss minimizes the distance between different patches that are assigned to the same expert
