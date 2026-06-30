## Positional Convolution Experts

Goal of this project is to study a dual branch routing with two-stage expert selection process inside a ResNet-Like backbone. Architecture is ispired by pMoE and ViT. We perform  patch-level routing like pMoE and ViT but inside a  ResNet architecture.

Router has two independent branch : 

- **Semantic Branch [N, C, Ph, Pw] :** Semantic branch extracts a semantic representation, which is used by router in the  second stage of selection process. Semantic representation helps to decide which expert is more suitable for a specific pattern inside a patch at a specific position. This is importanto because patches in the same position can contain different semantic patterns.
- **Position Branch [N, FC] :** Position branch use Fourier Features to decide which expert is more suitable for a specific position. Fourier feature help the router to capture positional patterns and represent where each patch is located inside the image.

Each branch returns its own logits. To decide which expert should process each patch, we apply softmax to the positon logits divded by a temperature. Temperature decays during training. In the first stage, experts select the position tokens they are suited for. Howeber, tokens in the same position can still contain different semantic patterns. For this reason in the second stage, we compute the softmax on the semantic logits only for tokens selected by each experts, and we assign higher scores to the experts that are more suitable for those tokens. To ensure that experts get different position tokens use a Centroid Loss.

After each expertr has processed its assigned tokens through convolutional layers, we reconstruct the entire feature map using the outputs of the experts. We also add a residual connection, so that dropped tokens can preserve their original identity. 
After the reconstruction we apply a dense branch that mixes the expert outputs in order to reduce discontinuities between patches.

Each experts is composed of :

> Conv → GN → SiLU → Conv → GN + Residual Connection


## What i m working on :

- **Reduce Latency :** Latency at the moment is 70ms whitout Torch.compile, goal is reduce it with vmap
- **Improve Training** : 200 epochs, eta_min = 0.01, EMA on backbone weights
- **Reduce Dense Branch :** At the moment the network  relies heavily on the dense branch

### Ablation :

- **Static Contigous Router :** Study Static Contigous Router 
- **Static Position Learnable map :** Map position → experts learnable, same for each images, independent of semantic and fourier
- **Static Position Map :** Map position → experts, not learnable (hardcoded)
- **Only Fourier or Semantic Routing :** Use only semantic or fourier feature to route token to expert
- **Dual Branch sum Routing :** Use dual branch routing, softmax is applied on semantic and fourier logits sum

## Tiny ImageNet Tests : 
*Work in progress...*

## Pascal-VOC Test
*Work in progress...*

**To more details of the architecture and training :** [Architecture and Training detail latex documentation]{docs/latex_docs/main.pdf}