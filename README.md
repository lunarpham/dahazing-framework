# MSFA-DeNet: A Multi-Scaled Feature Attention Dehazing Network

**Student:** Pham Duy Truong (21020014)  
**Advisor:** Nguyen Van Tho  

---

## 0. Abstract
Removing haze from outdoor images is a challenging but necessary task for computer vision. While early statistical methods and lightweight deep learning models established a strong foundation for this field, recovering natural colors in complex areas like the sky remains a challenge. This report investigates a lightweight, efficient solution called **MSFA-DeNet (Multi-Scale Feature Attention Dehazing Network)**. By combining a multi-scale architecture with Feature Attention mechanisms, the proposed model aims to effectively remove haze while maintaining low computational costs. Trained on the OTS dataset, the model shows highly accurate results compared to previous lightweight methods.

## 1. Introduction: Challenges in Image Dehazing
Haze is caused by tiny particles in the atmosphere that scatter light, making images look faded and gray. To solve this, researchers first created groundbreaking statistical rules like the Dark Channel Prior (DCP). DCP is very effective for general landscapes; however, because it relies on finding "dark" pixels, it can encounter difficulties when processing naturally bright areas like the sky, sometimes causing color shifts.

To improve upon this, researchers introduced machine learning, creating innovative lightweight Convolutional Neural Networks (CNNs) like DehazeNet and AOD-Net. While these models are very fast, their simpler structures can sometimes struggle with thick haze, occasionally leaving the final images a bit dark or with residual fog. Therefore, I propose MSFA-DeNet, which adopts the advanced techniques of these larger networks but simplifies them to run efficiently on standard hardware.

## 2. Proposed Method: MSFA-DeNet Architecture
To make the model lightweight but capable of handling complex scenes, MSFA-DeNet uses a simplified encoder-decoder structure with two main strategies: Multi-Scale processing and Feature Attention. Below are the key blocks that make this possible.

### 2.1. The Simplified Residual Convolutional Block
To stay lightweight, MSFA-DeNet uses a simple 2-layer residual convolutional block combined with Group Normalization, which is much more stable for small batch sizes. *(Note: MSFA-DeNet v2 removes normalization entirely, relying on careful initialization and residual scaling for even greater stability).*

### 2.2. Feature Attention (FA) Block
To help the model process bright areas like the sky naturally, it uses a Feature Attention block. This block acts as a guide, telling the network to focus mainly on thick haze pixels and apply less processing to areas that are already clear. It contains two consecutive parts:
- **Channel Attention (CA)** learns which color channels are most important.
- **Pixel Attention (PA)** learns which specific spatial areas on the image contain the thickest haze. It then combines with CA to form the final FeatureAttention block.

### Multi-Scale Forward Pass
MSFA-DeNet uses multiple scales: full resolution (scale 0) and half resolution (scale 1). The forward pass implementation demonstrates how the features are extracted, down-sampled, and then fused back together using a skip connection.

*(Figure 1: Model Pipeline of MSFA-DeNet - Showing the multi-scale encoder and decoder, guided by Feature Attention.)*

## 3. Training Process and Analysis
The model was programmed using PyTorch. To test its real-world capability, I trained it using the OTS (Outdoor Training Set) dataset on a personal RTX 3070 GPU. Because the model is optimized, the total training time was very efficient: 10.5 hours for 103 epochs.

*(Figure 2: Training Loss and Learning Rate)*
The training chart shows a highly stable learning process. The training loss rapidly dropped from 0.19 and smoothly converged to around 0.12, indicating that the lightweight blocks steadily learned how to map hazy pixels to clear ones.

*(Figure 3 & 4: Validation using SSIM and PSNR methods during training process)*
During training, the model's quality was tracked using PSNR and SSIM metrics. The curves show continuous improvement.

## 4. Visual Comparison and Results
I compared MSFA-DeNet against the foundational models: DCP, AOD-Net, and DehazeNet.

*(Figure 5: Visual Comparison on Outdoor Images)*
*(Figure 6: Benchmarking metrics over 103 epochs with customized hazy images)*

**Analysis of the Results:**
- **DCP:** Performs well on the buildings but struggles slightly with the bright sky regions, resulting in some color shifts and visible borders around the architecture (the so-called halo artifact).
- **AOD-Net & DehazeNet:** These lightweight models show great improvements in color naturalness. However, because of their simpler structure, they leave some residual haze, making the overall scenes appear slightly washed out.
- **MSFA-DeNet (Ours):** By utilizing the Feature Attention block, the model has the capacity to restore images closely approaching the Ground Truth, preserving the natural color of the sky and the sharp contrast of the buildings.

## 5. Conclusion and Future Work
By carefully adopting Multi-Scale processing and Feature Attention, MSFA-DeNet establishes a strong foundation. It improves upon the limitations of earlier lightweight methods and achieves highly capable image dehazing while remaining accessible for standard computing hardware.

However, during this research, two primary areas for future improvement were identified:
1. **Training on Spatially Varying Haze:** Real-world weather is rarely uniform. A single outdoor image often contains several different thicknesses of haze at once (e.g., dense fog near the ground, but a lighter mist higher up). The current model was trained primarily on homogeneously hazy datasets. Future iterations must be fine-tuned with datasets containing spatially varying haze across different zones of the image to improve generalizability.
2. **Advanced Upsampling Techniques:** Currently, the Decoder uses standard bilinear interpolation to enlarge the feature maps back to full resolution. While computationally cheap, this basic upsampling method can sometimes blur fine, high-frequency details (such as distant tree leaves). Future research should explore learnable upsampling methods, such as PixelShuffle to reconstruct the image while strictly preserving sharp edges without destroying details. *(Note: This has been successfully integrated into MSFA-DeNet v2!)*

---

## Appendix: Project Structure
The repository has been streamlined to exclusively support the **MSFA-DeNet v2** architecture for maximum efficiency and readability.

```
dahazing-framework/
├── options/                          # TOML config files
│   ├── train_msfa_denet_v2.toml
│   └── train_msfa_denet_v2_nhhaze.toml
├── scripts/
│   ├── train.py                      # Training entry point
│   └── infer.py                      # Inference entry point
├── src/
│   ├── datasets/                     # Data loading & pairing
│   │   ├── dataset.py
│   │   ├── pairing.py
│   │   └── sampler.py
│   ├── models/                       # Neural networks
│   │   ├── builder.py
│   │   └── msfa_denet_v2.py          # Core architecture
│   └── utils/                        # Utilities
│       ├── options.py
│       ├── losses.py
│       ├── metrics.py
│       ├── logger.py
│       └── image_io.py
├── experiments/                      # Auto-created experiment outputs
├── README.md
└── requirements.txt
```
