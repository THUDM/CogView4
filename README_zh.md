# CogView4 & CogView3 & CogView-3Plus

[Read this in English](./README.md)

<div align="center">
<img src=resources/logo.svg width="50%"/>
</div>
<p align="center">
<a href="https://huggingface.co/spaces/THUDM-HF-SPACE/CogView3-Plus-3B-Space" target="_blank"> 🤗 HuggingFace Space</a>  <a href="https://modelscope.cn/studios/ZhipuAI/CogView4" target="_blank">  🤖ModelScope Space</a>  <a href="resources/WECHAT.md" target="_blank"> 👋 微信社群</a>  <a href="https://arxiv.org/abs/2403.05121" target="_blank">📚 CogView3 论文</a> 
</p>


![showcase.png](resources/showcase.png)

## 项目更新

- 🔥🔥 ```2025/03/04```: 我们适配和开源了 [diffusers](https://github.com/huggingface/diffusers) 版本的  **CogView-4**
  模型，该模型具有6B权重，支持原生中文输入，支持中文文字绘画。你可以前往[在线体验](https://huggingface.co/spaces/THUDM-HF-SPACE/CogView4)。
- ```2024/10/13```: 我们适配和开源了 [diffusers](https://github.com/huggingface/diffusers) 版本的  **CogView-3Plus-3B**
  模型。你可以前往[在线体验](https://huggingface.co/spaces/THUDM-HF-SPACE/CogView3-Plus-3B-Space)。
- ```2024/9/29```: 我们已经开源了 **CogView3**  以及 **CogView-3Plus-3B** 。**CogView3** 是一个基于级联扩散的文本生成图像系统，采用了接力扩散框架。
  **CogView-3Plus** 是一系列新开发的基 Diffusion Transformer 的文本生成图像模型。

## 项目计划

- [X] diffusers 工作流适配
- [ ] ComfyUI diffusers节点
- [ ] 微调脚本和生态套件
- [ ] ControlNet模型发布
- [ ] Cog系列微调套件

## 模型介绍

### 模型对比

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">模型名称</th>
    <th style="text-align: center;">CogView4</th>
    <th style="text-align: center;">CogView3-Plus-3B</th>
  </tr>
    <td style="text-align: center;">分辨率</td>
    <td colspan="2" style="text-align: center;">
            512 <= H, W <= 2048 <br>
            H * W <= 2^{21} <br>
            H, W \mod 32 = 0
    </td>
  <tr>
    <td style="text-align: center;">推理精度</td>
    <td colspan="2" style="text-align: center;">仅支持BF16, FP32</td>
  <tr>
  <td style="text-align: center;">编码器</td>
  <td style="text-align: center;"><a href="https://huggingface.co/THUDM/glm-4-9b-hf" target="_blank">GLM-4-9B</a></td>
  <td style="text-align: center;"><a href="https://huggingface.co/google/t5-v1_1-xxl" target="_blank">T5-XXL</a></td>
</tr>
  <tr>
    <td style="text-align: center;">提示词语言</td>
    <td  style="text-align: center;">中文，English</td>
    <td style="text-align: center;">English</td>
  </tr>
  <tr>
    <td style="text-align: center;">提示词长度上限</td>
    <td style="text-align: center;">1024 Tokens</td>
    <td style="text-align: center;">224 Tokens</td>
  </tr>
  <tr>
    <td style="text-align: center;">下载链接 </td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogView4-6B">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogView4-6B">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogView4-6B">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogView3-Plus-3B">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogView3-Plus-3B">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogView3-Plus-3B">🟣 WiseModel</a></td>
  </tr>

</table>

### 显存占用

DIT模型均使用 `BF16` 精度,  `batchsize=4` 进行测试，测试结果如下表所示:

| 分辨率         | enable_model_cpu_offload OFF | enable_model_cpu_offload ON | enable_model_cpu_offload ON </br> Text Encoder 4bit | 
|-------------|------------------------------|-----------------------------|-----------------------------------------------------| 
| 512 * 512   | 33GB                         | 20GB                        | 13G                                                 | 
| 1280 * 720  | 35GB                         | 20GB                        | 13G                                                 | 
| 1024 * 1024 | 35GB                         | 20GB                        | 13G                                                 | 
| 1920 * 1280 | 39GB                         | 20GB                        | 14G                                                 | 
| 2048 * 2048 | 43GB                         | 21GB                        | 14G                                                 | 

此外, 建议您的设备至少拥有`32GB`内存，以防止进程被杀。

### 模型指标

我们在多个榜单上进行了测试, 并得到了如下的成绩:

#### dpg_bench

| Model        | Overall   | Global    | Entity    | Attribute | Relation  | Other     |
|--------------|-----------|-----------|-----------|-----------|-----------|-----------|
| SDXL         | 74.65     | 83.27     | 82.43     | 80.91     | 86.76     | 80.41     |
| PixArt-alpha | 71.11     | 74.97     | 79.32     | 78.60     | 82.57     | 76.96     |
| sd3-medium   | 84.08     | 87.90     | **91.01** | 88.83     | 80.70     | 88.68     |
| dalle-3      | 83.50     | **90.97** | 89.61     | 88.39     | 90.58     | 89.83     |
| Flux.1-dev   | 83.79     | 85.80     | 86.79     | 89.98     | 90.04     | **89.90** |
| Janus-Pro-7B | 84.19     | 86.90     | 88.90     | 89.40     | 89.32     | 89.48     |
| **cogview4** | **85.13** | 83.85     | 90.35     | **91.17** | **91.14** | 87.29     |

#### Geneval

| Model           | Overall  | Single Obj. | Two Obj. | Counting | Colors   | Position | Color attribution |
|-----------------|----------|-------------|----------|----------|----------|----------|-------------------|
| SDXL            | 0.55     | 0.98        | 0.74     | 0.39     | 0.85     | 0.15     | 0.23              |
| PixArt-alpha    | 0.48     | 0.98        | 0.50     | 0.44     | 0.80     | 0.08     | 0.07              |
| SD3-Medium      | **0.74** | **0.99**    | **0.94** | 0.72     | **0.89** | 0.33     | 0.60              |
| DALL-E 3        | 0.67     | 0.96        | 0.87     | 0.47     | 0.83     | 0.43     | 0.45              |
| Flux.1-dev      | 0.66     | 0.98        | 0.79     | **0.73** | 0.77     | 0.22     | 0.45              |
| Janus-Pro-7B    | 0.80     | 0.99        | 0.89     | 0.59     | 0.90     | 0.79     | **0.66**          |
| **CogView4-6B** | 0.73     | **0.99**    | 0.86     | 0.66     | 0.79     | **0.48** | 0.58              |

#### t2i_compbench

| Model           | Color      | Shape      | Texture    | 2D-Spatial | 3D-Spatial | Numeracy   | Non-spatial Clip | Complex 3-in-1 |
|-----------------|------------|------------|------------|------------|------------|------------|------------------|----------------|
| SDXL            | 0.5879     | 0.4687     | 0.5299     | 0.2133     | 0.3566     | 0.4988     | 0.3119           | 0.3237         |
| PixArt-alpha    | 0.6690     | 0.4927     | 0.6477     | 0.2064     | 0.3901     | 0.5058     | **0.3197**       | 0.3433         |
| sd3-medium      | **0.8132** | 0.5885     | **0.7334** | **0.3200** | **0.4084** | 0.6174     | 0.3140           | 0.3771         |
| DALL-E 3        | 0.7785     | **0.6205** | 0.7036     | 0.2865     | 0.3744     | 0.5880     | 0.3003           | 0.3773         |
| Flux.1-dev      | 0.7572     | 0.5066     | 0.6300     | 0.2700     | 0.3992     | 0.6165     | 0.3065           | 0.3628         |
| Janus-Pro-7B    | 0.5145     | 0.3323     | 0.4069     | 0.1566     | 0.2753     | 0.4406     | 0.3137           | 0.3806         |
| **CogView4-6B** | 0.7786     | 0.5880     | 0.6983     | 0.3075     | 0.3708     | **0.6626** | 0.3056           | **0.3869**     |

## 中文文字准确率评测

| model           | Precision  | Recall     | F1 Score   | pick@4     |
|-----------------|------------|------------|------------|------------|
| kolors          | 0.6094     | 0.1886     | 0.2880     | 0.1633     |
| **CogView4-6B** | **0.6969** | **0.5532** | **0.6168** | **0.3265** |

## 推理模型

### 提示词优化

虽然 CogView4 系列模型都是通过长篇合成图像描述进行训练的，但我们强烈建议在文本生成图像之前，基于大语言模型进行提示词的重写操作，这将大大提高生成质量。

我们提供了一个 [示例脚本](inference/prompt_optimize.py)。我们建议您运行这个脚本，以实现对提示词对润色。请注意，`CogView4` 和
`CogView3` 模型的提示词优化使用的few shot不同。需要区分。

```shell
cd inference
python prompt_optimize.py --api_key "智谱AI API Key" --prompt {你的提示词} --base_url "https://open.bigmodel.cn/api/paas/v4" --model "glm-4-plus" --cogview_version "cogview4"
```

### 推理模型

以 `BF16` 的精度运行模型:

```python
from diffusers import CogView4Pipeline
import torch

pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16).to("cuda")

# Open it for reduce GPU memory usage
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."
image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    num_images_per_prompt=1,
    num_inference_steps=50,
    width=1024,
    height=1024,
).images[0]

image.save("cogview4.png")
```

其他更多推理代码，请查看:

1. 使用`int4`加载`ext encoder` 以及推理代码的完整注释，查看 [这里](inference/cli_demo_cogview4.py)。
2. 搭建 `gradio` GUI DEMO，查看 [这里](inference/gradio_web_demo.py)

## 开源协议

本仓库代码和 CogView3 模型均采用 [Apache 2.0](./LICENSE) 开源协议。

我们欢迎和感谢你贡献代码，你可以在 [这里](resources/contribute.md) 查看贡献指南。
