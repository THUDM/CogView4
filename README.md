# CogView4 & CogView3 & CogView-3Plus

[阅读中文版](./README_zh.md)

<div align="center">
<img src=resources/logo.svg width="50%"/>
</div>
<p align="center">
<a href="https://huggingface.co/spaces/THUDM-HF-SPACE/CogView3-Plus-3B-Space" target="_blank"> 🤗 HuggingFace Space</a>  <a href="https://modelscope.cn/studios/ZhipuAI/CogView4" target="_blank">  🤖ModelScope Space</a> <a href="resources/WECHAT.md" target="_blank"> 👋 WeChat Community</a>  <a href="https://arxiv.org/abs/2403.05121" target="_blank">📚 CogView3 Paper</a> 
</p>


![showcase.png](resources/showcase.png)

## Project Updates

- 🔥🔥 ```2025/03/04```: We've adapted and open-sourced the [diffusers](https://github.com/huggingface/diffusers) version
  of **CogView-4** model, which has 6B parameters, supports native Chinese input, and Chinese text-to-image generation.
  You can try it [online](https://huggingface.co/spaces/THUDM-HF-SPACE/CogView4).
- ```2024/10/13```: We've adapted and open-sourced the [diffusers](https://github.com/huggingface/diffusers) version of
  **CogView-3Plus-3B** model. You can try
  it [online](https://huggingface.co/spaces/THUDM-HF-SPACE/CogView3-Plus-3B-Space).
- ```2024/9/29```: We've open-sourced **CogView3** and **CogView-3Plus-3B**. **CogView3** is a text-to-image system
  based on cascading diffusion, using a relay diffusion framework. **CogView-3Plus** is a series of newly developed
  text-to-image models based on Diffusion Transformer.

## Project Plan

- [X] diffusers workflow adaptation
- [ ] ComfyUI diffusers nodes
- [ ] Fine-tuning scripts and ecosystem kits
- [ ] ControlNet model release
- [ ] Cog series fine-tuning kit

## Model Introduction

### Model Comparison

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">Model Name</th>
    <th style="text-align: center;">CogView4</th>
    <th style="text-align: center;">CogView3-Plus-3B</th>
  </tr>
    <td style="text-align: center;">Resolution</td>
    <td colspan="2" style="text-align: center;">
            512 <= H, W <= 2048 <br>
            H * W <= 2^{21} <br>
            H, W \mod 32 = 0
    </td>
  <tr>
    <td style="text-align: center;">Inference Precision</td>
    <td colspan="2" style="text-align: center;">Only supports BF16, FP32</td>
  <tr>
  <td style="text-align: center;">Encoder</td>
  <td style="text-align: center;"><a href="https://huggingface.co/THUDM/glm-4-9b-hf" target="_blank">GLM-4-9B</a></td>
  <td style="text-align: center;"><a href="https://huggingface.co/google/t5-v1_1-xxl" target="_blank">T5-XXL</a></td>
</tr>
  <tr>
    <td style="text-align: center;">Prompt Language</td>
    <td style="text-align: center;">Chinese, English</td>
    <td style="text-align: center;">English</td>
  </tr>
  <tr>
    <td style="text-align: center;">Prompt Length Limit</td>
    <td style="text-align: center;">1024 Tokens</td>
    <td style="text-align: center;">224 Tokens</td>
  </tr>
  <tr>
    <td style="text-align: center;">Download Links</td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogView4-6B">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogView4-6B">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogView4-6B">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogView3-Plus-3B">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogView3-Plus-3B">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogView3-Plus-3B">🟣 WiseModel</a></td>
  </tr>
</table>

### Memory Usage

DIT models are tested with `BF16` precision and `batchsize=4`, with results shown in the table below:

| Resolution  | enable_model_cpu_offload OFF | enable_model_cpu_offload ON | enable_model_cpu_offload ON </br> Text Encoder 4bit |
|-------------|------------------------------|-----------------------------|-----------------------------------------------------|
| 512 * 512   | 33GB                         | 20GB                        | 13G                                                 |
| 1280 * 720  | 35GB                         | 20GB                        | 13G                                                 |
| 1024 * 1024 | 35GB                         | 20GB                        | 13G                                                 |
| 1920 * 1280 | 39GB                         | 20GB                        | 14G                                                 |
| 2048 * 2048 | 43GB                         | 21GB                        | 14G                                                 |

Additionally, we recommend that your device has at least `32GB` of RAM to prevent the process from being killed.

### Model Metrics

We've tested on multiple benchmarks and achieved the following scores:

#### dpg_bench

| Model        | Overall   | Global    | Entity    | Attribute | Relation  | Other     |
|--------------|-----------|-----------|-----------|-----------|-----------|-----------|
| SDXL         | 74.65     | 83.27     | 82.43     | 80.91     | 86.76     | 80.41     |
| PixArt-alpha | 71.11     | 74.97     | 79.32     | 78.60     | 82.57     | 76.96     |
| SD3-Medium   | 84.08     | 87.90     | **91.01** | 88.83     | 80.70     | 88.68     |
| DALL-E 3      | 83.50     | **90.97** | 89.61     | 88.39     | 90.58     | 89.83     |
| Flux.1-dev   | 83.79     | 85.80     | 86.79     | 89.98     | 90.04     | **89.90** |
| Janus-Pro-7B | 84.19     | 86.90     | 88.90     | 89.40     | 89.32     | 89.48     |
| **CogView4-6B** | **85.13** | 83.85     | 90.35     | **91.17** | **91.14** | 87.29     |

#### Geneval

| Model           | Overall  | Single Obj. | Two Obj. | Counting | Colors   | Position | Color attribution |
|-----------------|----------|-------------|----------|----------|----------|----------|-------------------|
| SDXL            | 0.55     | 0.98        | 0.74     | 0.39     | 0.85     | 0.15     | 0.23              |
| PixArt-alpha    | 0.48     | 0.98        | 0.50     | 0.44     | 0.80     | 0.08     | 0.07              |
| SD3-Medium      | 0.74     | **0.99**    | **0.94** | 0.72     | 0.89     | 0.33     | 0.60              |
| DALL-E 3        | 0.67     | 0.96        | 0.87     | 0.47     | 0.83     | 0.43     | 0.45              |
| Flux.1-dev      | 0.66     | 0.98        | 0.79     | **0.73** | 0.77     | 0.22     | 0.45              |
| Janus-Pro-7B    | **0.80** | **0.99**    | 0.89     | 0.59     | **0.90** | **0.79** | **0.66**          |
| **CogView4-6B** | 0.73     | **0.99**    | 0.86     | 0.66     | 0.79     | 0.48     | 0.58              |

#### t2i_compbench

| Model           | Color      | Shape      | Texture    | 2D-Spatial | 3D-Spatial | Numeracy   | Non-spatial Clip | Complex 3-in-1 |
|-----------------|------------|------------|------------|------------|------------|------------|------------------|----------------|
| SDXL            | 0.5879     | 0.4687     | 0.5299     | 0.2133     | 0.3566     | 0.4988     | 0.3119           | 0.3237         |
| PixArt-alpha    | 0.6690     | 0.4927     | 0.6477     | 0.2064     | 0.3901     | 0.5058     | **0.3197**       | 0.3433         |
| SD3-Medium      | **0.8132** | 0.5885     | **0.7334** | **0.3200** | **0.4084** | 0.6174     | 0.3140           | 0.3771         |
| DALL-E 3        | 0.7785     | **0.6205** | 0.7036     | 0.2865     | 0.3744     | 0.5880     | 0.3003           | 0.3773         |
| Flux.1-dev      | 0.7572     | 0.5066     | 0.6300     | 0.2700     | 0.3992     | 0.6165     | 0.3065           | 0.3628         |
| Janus-Pro-7B    | 0.5145     | 0.3323     | 0.4069     | 0.1566     | 0.2753     | 0.4406     | 0.3137           | 0.3806         |
| **CogView4-6B** | 0.7786     | 0.5880     | 0.6983     | 0.3075     | 0.3708     | **0.6626** | 0.3056           | **0.3869**     |

## Chinese Text Accuracy Evaluation

| Model           | Precision  | Recall     | F1 Score   | pick@4     |
|-----------------|------------|------------|------------|------------|
| Kolors          | 0.6094     | 0.1886     | 0.2880     | 0.1633     |
| **CogView4-6B** | **0.6969** | **0.5532** | **0.6168** | **0.3265** |

## Inference Model

### Prompt Optimization

Although CogView4 series models are trained with lengthy synthetic image descriptions, we strongly recommend using a
large language model to rewrite prompts before text-to-image generation, which will greatly improve generation quality.

We provide an [example script](inference/prompt_optimize.py). We recommend running this script to refine your prompts.
Note that `CogView4` and `CogView3` models use different few-shot examples for prompt optimization. They need to be
distinguished.

```shell
cd inference
python prompt_optimize.py --api_key "Zhipu AI API Key" --prompt {your prompt} --base_url "https://open.bigmodel.cn/api/paas/v4" --model "glm-4-plus" --cogview_version "cogview4"
```

### Inference Model

Run the model with `BF16` precision:

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

For more inference code, please check:

1. For using `int4` to load `text encoder` and complete inference code annotations,
   check [here](inference/cli_demo_cogview4.py).
2. For setting up a `gradio` GUI DEMO, check [here](inference/gradio_web_demo.py)

## License

The code in this repository and the CogView3 models are licensed under [Apache 2.0](./LICENSE).

We welcome and appreciate your code contributions. You can view the contribution
guidelines [here](resources/contribute.md).
