<div align="center">
<h1>🎨 LucidFlux:<br/>Caption-Free Universal Image Restoration with a Large-Scale Diffusion Transformer</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2506.10741-red)](https://github.com/W2GenAI-Lab/LucidFlux)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/W2GenAI-Lab/LucidFlux)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://github.com/W2GenAI-Lab/LucidFlux)
[![Website](https://img.shields.io/badge/🌐-Website-green)](https://github.com/W2GenAI-Lab/LucidFlux)
[![HF Demo](https://img.shields.io/badge/🤗-HF_Demo-orange)](https://github.com/W2GenAI-Lab/LucidFlux)

<img src="images/logo/logo2.png" alt="LucidFlux Logo" width="1000"/>

### [**🌐 Website**](https://github.com/W2GenAI-Lab/LucidFlux) | [**🎯 Demo**](https://github.com/W2GenAI-Lab/LucidFlux) | [**📄 Paper**](https://github.com/W2GenAI-Lab/LucidFlux) | [**🤗 Models**](https://github.com/W2GenAI-Lab/LucidFlux) | [**🤗 HF Demo**](https://github.com/W2GenAI-Lab/LucidFlux)

</div>

---

## News & Updates


<!-- - 🖥️ **[2025.06]** We have pushed our work on [MeiGen-AI](https://github.com/MeiGen-AI), where you can explore not only our project but also the work of other colleagues. Feel free to check it out for more insights and contributions.
- 🧩 **[2025.06]** Community user [@AIFSH](https://github.com/AIFSH) has successfully integrated **LucidFlux into ComfyUI**!  
  You can check out the full workflow here: [LucidFlux-ComfyUI Example](https://www.xiangongyun.com/image/detail/68b711eb-a31e-47db-82eb-47438359f4bf?r=XLVYLW)  
  Big thanks to the contributor — this will be helpful for many users! See [Issue #6](https://github.com/Ephemeral182/LucidFlux/issues/6) for details.
- 📖 **[2025.06]** Our **Chinese article** providing a detailed introduction and technical walkthrough of LucidFlux is now available!  
Read it here: [中文解读｜高质量美学海报生成框架 LucidFlux](https://mp.weixin.qq.com/s/gq6DwohKP0z333OSDRe7Xw)
- 🔥 **[2025.06]** We have deployed a demo on Hugging Face Space, feel free to give it a try!
- 🚀 **[2025.06]** Our gradio demo and inference code are now available!
- 📊 **[2025.06]** We have released partial datasets and model weights on HuggingFace. -->

---

Let me know if this works!

## 👥 Authors

> [**Song Fei**](https://github.com/FeiSong123)<sup>1</sup>\*, [**Tian Ye**](https://owen718.github.io/)<sup>1</sup>\*‡, [**Lei Zhu**](https://sites.google.com/site/indexlzhu/home)<sup>1,2</sup>†
>
> <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)  
> <sup>2</sup>The Hong Kong University of Science and Technology  
>
> \*Equal Contribution, ‡Project Leader, †Corresponding Author

---

## 🌟 What is LucidFlux?

<!-- <div align="center">
<img src="images/demo/demo2.png" alt="What is LucidFlux - Quick Prompt Demo" width="1000"/>
<br>
</div> -->

LucidFlux is a framework designed to perform high-fidelity image restoration across a wide range of degradations without requiring textual captions. By combining a Flux-based DiT backbone with dual ControlNet branches and SigLIP semantic alignment, LucidFlux enables caption-free guidance while preserving structural and semantic consistency, achieving superior restoration quality.

<!-- ## 🚀 Quick Start

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/ephemeral182/LucidFlux.git
cd LucidFlux

# Create conda environment
conda create -n postercraft python=3.11
conda activate postercraft

# Install dependencies
pip install -r requirements.txt

``` -->

<!-- ### 🚀 Quick Generation

Generate high-quality aesthetic posters from your prompt with `BF16` precision:

```bash
python inference.py \
  --prompt "Urban Canvas Street Art Expo poster with bold graffiti-style lettering and dynamic colorful splashes" \
  --enable_recap \
  --num_inference_steps 28 \
  --guidance_scale 3.5 \
  --seed 42 \
  --pipeline_path "black-forest-labs/FLUX.1-dev" \
  --custom_transformer_path "LucidFlux/LucidFlux-v1_RL" \
  --qwen_model_path "Qwen/Qwen3-8B"
```

If you are running on a GPU with limited memory, you can use `inference_offload.py` to offload some components to the CPU:

```bash
python inference_offload.py \
  --prompt "Urban Canvas Street Art Expo poster with bold graffiti-style lettering and dynamic colorful splashes" \
  --enable_recap \
  --num_inference_steps 28 \
  --guidance_scale 3.5 \
  --seed 42 \
  --pipeline_path "black-forest-labs/FLUX.1-dev" \
  --custom_transformer_path "LucidFlux/LucidFlux-v1_RL" \
  --qwen_model_path "Qwen/Qwen3-8B"
``` -->
<!-- 
### 💻 Gradio Web UI

We provide a Gradio web UI for LucidFlux. 

```bash
python demo_gradio.py
``` -->


## 📊 Performance Benchmarks

<div align="center">

### 📈 Quantitative Results

<table>
<thead>
  <tr>
    <th>Benchmark</th>
    <th>Metric</th>
    <th>SeeSR</th>
    <th>DreamClear</th>
    <th>SUPIR</th>
    <th>Ours</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="7" style="text-align:center; vertical-align:middle;">RealLQ250</td>
    <td style="white-space: nowrap;">CLIP-IQA ↑</td>
    <td>0.7063</td>
    <td>0.6950</td>
    <td>0.5767</td>
    <td><b>0.7122</b></td>
  </tr>
  <tr>
    <td style="white-space: nowrap;">NIQE ↓</td>
    <td>4.4383</td>
    <td>3.8700</td>
    <td><b>3.6591</b></td>
    <td>3.6742</td>
  </tr>
  <tr>
    <td style="white-space: nowrap;">MUSIQ ↑</td>
    <td>70.38</td>
    <td>67.08</td>
    <td>65.81</td>
    <td><b>73.01</b></td>
  </tr>
  <tr>
    <td style="white-space: nowrap;">MANIQA ↑</td>
    <td>0.4895</td>
    <td>0.4400</td>
    <td>0.3826</td>
    <td><b>0.5589</b></td>
  </tr>
  <tr>
    <td style="white-space: nowrap;">QAlign ↑</td>
    <td>4.1423</td>
    <td>4.0640</td>
    <td>4.1347</td>
    <td><b>4.3935</b></td>
  </tr>
  <tr>
    <td style="white-space: nowrap;">NIMA ↑</td>
    <td>5.3146</td>
    <td>5.2200</td>
    <td>5.0806</td>
    <td><b>5.4836</b></td>
  </tr>
  <tr>
    <td style="white-space: nowrap;">CLIP-IQA+ ↑</td>
    <td>0.7034</td>
    <td>0.6810</td>
    <td>0.6532</td>
    <td><b>0.7406</b></td>
  </tr>
</tbody>
</table>



<!-- <img src="images/user_study/hpc.png" alt="User Study Results" width="1000"/> -->

</div>

---

## 🎭 Gallery & Examples

<div align="center">

### 🎨 LucidFlux Gallery

<table>
<tr align="center">
    <td width="200"><b>LQ</b></td>
    <td width="200"><b>SeeSR</b></td>
    <td width="200"><b>SUPIR</b></td>
    <td width="200"><b>DreamClear</b></td>
    <td width="200"><b>Ours</b></td>
</tr>
<tr align="center"><td colspan="5"><img src="images/gallery/006.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/013.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/019.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/040.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/041.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/079.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/082.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/111.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/123.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/135.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/137.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/144.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/151.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/166.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/182.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/183.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/202.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/207.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/208.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/214.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/222.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/224.jpg" width="1000"></td></tr>
<tr align="center"><td colspan="5"><img src="images/gallery/231.jpg" width="1000"></td></tr>
</table>

</div>

---

## 🏗️ Model Architecture

<div align="center">
<img src="images/framework/framework.png" alt="LucidFlux Framework Overview" width="1000"/>
<br>
<em><strong>Caption-Free Universal Image Restoration with a Large-Scale Diffusion Transformer</strong></em>
</div>

Our unified framework consists of **four critical components in the training workflow**:

### 🔤 Scaling Up Real-world High-Quality Data for Universal Image Restoration
Existing restoration datasets are too small to support large diffusion models. To address this, we build a large-scale high-quality dataset through a three-stage automatic filtering pipeline (blur, flat-region, and IQA), selecting 342K images from 2.95M candidates. Using Real-ESRGAN, we further synthesize 1.37M paired samples, ensuring diverse and realistic degradations for effective training.

### 🎨 Two Parallel ControlNet Branches for Low-Quality Image Conditioning 
Flux.1 was not built for restoration, and existing ControlNet strategies either leave artifacts or oversmooth details. We follow DreamClear with a dual-branch design: one branch uses the raw LQ image to keep textures, while the other applies lightweight degradation removal. Together, they balance artifact removal and detail preservation for better restoration.

### 🎯 Timestep and Layer-Adaptive Condition Injection
In diffusion models, early timesteps mainly capture global structures, while later ones enrich fine details. A similar hierarchy exists within DiT layers: shallow layers tend to model coarse, low-frequency patterns, whereas deeper layers refine high-frequency textures. Injecting identical conditions across all layers neglects this distinction. To address this, we design a timestep- and layer-adaptive modulation strategy, where ControlNet features are injected in accordance with both the current timestep and the target layer. This dynamic conditioning better aligns with the coarse-to-fine restoration process, leading to more effective detail recovery.

### 🔄 SigLIP-Redux for Caption-Free Semantic Alignment
Text-to-image (T2I) diffusion models are designed to generate images from text, and prior works often use captions during training and inference to guide restoration. However, training captions come from clean images, which are unavailable in real world scenario, and captions generated from degraded inputs may describe the degradation itself, misleading the model. To overcome this, we extract semantic features directly from the low-quality input using SigLIP and align them with the DiT feature space via a learnable connector, providing robust caption-free semantic guidance for high-fidelity restoration.



<!-- ---

## 💾 Model Zoo

We provide the weights for our core models, fine-tuned at different stages of the LucidFlux pipeline.

<div align="center">
<table>
<tr>
<th>Model</th>
<th>Stage</th>
<th>Description</th>
<th>Download</th>
</tr>
<tr>
<td>🎯 <b>LucidFlux-v1_RL</b></td>
<td>Stage 3: Aesthetic-Text RL</td>
<td>Optimized via Aesthetic-Text Preference Optimization for higher-order aesthetic trade-offs.</td>
<td><a href="https://huggingface.co/LucidFlux/LucidFlux-v1_RL">🤗 HF</a></td>
</tr>
<tr>
<td>🔄 <b>LucidFlux-v1_Reflect</b></td>
<td>Stage 4: Vision-Language Feedback</td>
<td>Iteratively refined using vision-language feedback for further harmony and content accuracy.</td>
<td><a href="https://huggingface.co/LucidFlux/LucidFlux-v1_Reflect">🤗 HF</a></td>
</tr>
</table>
</div>

--- -->

<!-- ## 📚 Datasets


We provide **four specialized datasets** for training LucidFlux workflow:

### 🔤 Text-Render-2M
<div align="center">
<img src="images/dataset/dataset1.png" alt="Text-Render-2M Dataset" width="1000"/>
<br>
<em><strong>Text-Render-2M: Multi-instance text rendering with diverse selections</strong></em>
</div>

A comprehensive text rendering dataset containing **2 million high-quality examples**. Features multi-instance text rendering, diverse text selections (varying in size, count, placement, and rotation), and dynamic content generation through both template-based and random string approaches.

### 🎨 HQ-Poster-100K
<div align="center">
<img src="images/dataset/dataset2.png" alt="HQ-Poster-100K Dataset" width="1000"/>
<br>
<em><strong>HQ-Poster-100K: Curated high-quality aesthetic posters</strong></em>
</div>

**100,000** meticulously curated high-quality posters with advanced filtering techniques and multi-modal scoring. Features Gemini-powered mask generation with detailed captions for comprehensive poster understanding.

### 👍 Poster-Preference-100K
<div align="center">
<img src="images/dataset/dataset3.png" alt="Poster-Preference-100K Dataset" width="1000"/>
<br>
<em><strong>Poster-Preference-100K: Preference learning pairs for aesthetic optimization</strong></em>
</div>

This preference dataset is sourced from over **100,000** generated poster images. Through comprehensive evaluation by Gemini and aesthetic evaluators, we construct high-quality preference pairs designed for reinforcement learning to align poster generation with human aesthetic judgments.

### 🔄 Poster-Reflect-120K
<div align="center">
<img src="images/dataset/dataset4.png" alt="Poster-Reflect-120K Dataset" width="1000"/>
<br>
<em><strong>Poster-Reflect-120K: Vision-language feedback pairs for iterative refinement</strong></em>
</div>

This vision-language feedback dataset is sourced from over **120,000** generated poster images. Through comprehensive evaluation by Gemini and aesthetic evaluators, this dataset captures the iterative refinement process and provides detailed feedback for further improvements.

<div align="center">
<table>
<tr>
<th>Dataset</th>
<th>Size</th>
<th>Description</th>
<th>Download</th>
</tr>
<tr>
<td>🔤 <b>Text-Render-2M</b></td>
<td>2M samples</td>
<td>High-quality text rendering examples with multi-instance support</td>
<td><a href="https://huggingface.co/datasets/LucidFlux/Text-Render-2M">🤗 HF</a></td>
</tr>
<tr>
<td>🎨 <b>HQ-Poster-100K</b></td>
<td>100K samples</td>
<td>Curated high-quality posters with aesthetic evaluation</td>
<td><a href="https://huggingface.co/datasets/LucidFlux/HQ-Poster-100K">🤗 HF</a></td>
</tr>
<tr>
<td>👍 <b>Poster-Preference-100K</b></td>
<td>100K images</td>
<td>Preference learning poster pairs for RL training</td>
<td><a href="https://huggingface.co/datasets/LucidFlux/Poster-Preference-100K">🤗 HF</a></td>
</tr>
<tr>
<td>🔄 <b>Poster-Reflect-120K</b></td>
<td>120K images</td>
<td>Vision-language feedback pairs for iterative refinement</td>
<td><a href="https://huggingface.co/datasets/LucidFlux/Poster-Reflect-120K">🤗 HF</a></td>
</tr>
</table>
</div>

--- -->

<!-- ## 📝 Citation

If you find LucidFlux useful for your research, please cite our paper:

```bibtex
@article{chen2025postercraft,
  title={LucidFlux: Rethinking High-Quality Aesthetic Poster Generation in a Unified Framework},
  author={Chen, Sixiang and Lai, Jianyu and Gao, Jialin and Ye, Tian and Chen, Haoyu and Shi, Hengyu and Shao, Shitong and Lin, Yunlong and Fei, Song and Xing, Zhaohu and Jin, Yeying and Luo, Junfeng and Wei, Xiaoming and Zhu, Lei},
  journal={arXiv preprint arXiv:2506.10741},
  year={2025}
}
```

--- -->

## 🙏 Acknowledgments

- 🏛️ Thanks to our affiliated institutions for their support.
- 🤝 Special thanks to the open-source community for inspiration.

---

## 📬 Contact

For any questions or inquiries, please reach out to us:

- **Song Fei**: `sfei285@connect.hkust-gz.edu.cn`
- **Tian Ye**: `tye610@connect.hkust-gz.edu.cn`


</div>
