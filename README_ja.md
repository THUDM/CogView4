# CogView4 & CogView3 & CogView-3Plus

[Read this in English](./README.md)
[阅读中文版](./README_zh.md)

<div align="center">
<img src=resources/logo.svg width="50%"/>

</div>
<p align="center">
<a href="https://huggingface.co/spaces/THUDM-HF-SPACE/CogView4"  target="_blank"> 🤗 HuggingFace Space</a>  
<a href="https://modelscope.cn/studios/ZhipuAI/CogView4" target="_blank">  🤖ModelScope Space</a> 
<a href="https://zhipuaishengchan.datasink.sensorsdata.cn/t/4z" target="_blank"> 🛠️ZhipuAI MaaS(Faster)</a>
<br>
<a href="resources/WECHAT.md" target="_blank"> 👋 WeChat Community</a>  <a href="https://arxiv.org/abs/2403.05121" target="_blank">📚 CogView3 Paper</a> <a href="https://replicate.com/lucataco/cogview4-6b" target="_blank"> 🤖 Replicate</a>
</p>


![showcase.png](resources/showcase.png)

## プロジェクトの更新

- 🔥🔥 ```2025/03/04```: [diffusers](https://github.com/huggingface/diffusers) バージョンの **CogView-4**
  モデルを適応し、オープンソース化しました。このモデルは6Bのパラメータを持ち、ネイティブの中国語入力と中国語のテキストから画像生成をサポートしています。オンラインで試すことができます [こちら](https://huggingface.co/spaces/THUDM-HF-SPACE/CogView4)。
- ```2024/10/13```: [diffusers](https://github.com/huggingface/diffusers) バージョンの **CogView-3Plus-3B**
  モデルを適応し、オープンソース化しました。オンラインで試すことができます [こちら](https://huggingface.co/spaces/THUDM-HF-SPACE/CogView3-Plus-3B-Space)。
- ```2024/9/29```: **CogView3** と **CogView-3Plus-3B** をオープンソース化しました。**CogView3**
  はカスケード拡散に基づくテキストから画像生成システムで、リレーディフュージョンフレームワークを使用しています。*
  *CogView-3Plus** は新たに開発されたDiffusion Transformerに基づくテキストから画像生成モデルのシリーズです。

## プロジェクト計画

- [X] Diffusers ワークフローの適応
- [ ] Cogシリーズのファインチューニングスイート (近日公開)
- [ ] ControlNetモデルとトレーニングコード

## コミュニティの取り組み

本リポジトリに関連するいくつかのコミュニティプロジェクトをここにまとめました。これらのプロジェクトはコミュニティメンバーによって維持されており、彼らの貢献に感謝します。

+ [ComfyUI_CogView4_Wrapper](https://github.com/chflame163/ComfyUI_CogView4_Wrapper) - ComfyUI における CogView4 プロジェクトの実装。

## モデル紹介

### モデル比較

<table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th style="text-align: center;">モデル名</th>
    <th style="text-align: center;">CogView4</th>
    <th style="text-align: center;">CogView3-Plus-3B</th>
  </tr>
    <td style="text-align: center;">解像度</td>
    <td colspan="2" style="text-align: center;">
            512 <= H, W <= 2048 <br>
            H * W <= 2^{21} <br>
            H, W \mod 32 = 0
    </td>
  <tr>
    <td style="text-align: center;">推論精度</td>
    <td colspan="2" style="text-align: center;">BF16, FP32 のみサポート</td>
  <tr>
  <td style="text-align: center;">エンコーダ</td>
  <td style="text-align: center;"><a href="https://huggingface.co/THUDM/glm-4-9b-hf" target="_blank">GLM-4-9B</a></td>
  <td style="text-align: center;"><a href="https://huggingface.co/google/t5-v1_1-xxl" target="_blank">T5-XXL</a></td>
</tr>
  <tr>
    <td style="text-align: center;">プロンプト言語</td>
    <td style="text-align: center;">中国語、英語</td>
    <td style="text-align: center;">英語</td>
  </tr>
  <tr>
    <td style="text-align: center;">プロンプト長さの制限</td>
    <td style="text-align: center;">1024 トークン</td>
    <td style="text-align: center;">224 トークン</td>
  </tr>
  <tr>
    <td style="text-align: center;">ダウンロードリンク</td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogView4-6B">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogView4-6B">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogView4-6B">🟣 WiseModel</a></td>
    <td style="text-align: center;"><a href="https://huggingface.co/THUDM/CogView3-Plus-3B">🤗 HuggingFace</a><br><a href="https://modelscope.cn/models/ZhipuAI/CogView3-Plus-3B">🤖 ModelScope</a><br><a href="https://wisemodel.cn/models/ZhipuAI/CogView3-Plus-3B">🟣 WiseModel</a></td>
  </tr>
</table>

### メモリ使用量

DITモデルは `BF16` 精度と `batchsize=4` でテストされ、結果は以下の表に示されています：

| 解像度         | enable_model_cpu_offload OFF | enable_model_cpu_offload ON | enable_model_cpu_offload ON </br> Text Encoder 4bit | 
|-------------|------------------------------|-----------------------------|-----------------------------------------------------| 
| 512 * 512   | 33GB                         | 20GB                        | 13G                                                 | 
| 1280 * 720  | 35GB                         | 20GB                        | 13G                                                 | 
| 1024 * 1024 | 35GB                         | 20GB                        | 13G                                                 | 
| 1920 * 1280 | 39GB                         | 20GB                        | 14G                                                 | 

さらに、プロセスが強制終了されないようにするために、少なくとも`32GB`のRAMを持つデバイスを推奨します。

### モデル指標

複数のベンチマークでテストを行い、以下のスコアを達成しました：

#### DPG-Bench

| モデル             | 全体        | グローバル     | エンティティ    | 属性        | 関係        | その他       |
|-----------------|-----------|-----------|-----------|-----------|-----------|-----------|
| SDXL            | 74.65     | 83.27     | 82.43     | 80.91     | 86.76     | 80.41     |
| PixArt-alpha    | 71.11     | 74.97     | 79.32     | 78.60     | 82.57     | 76.96     |
| SD3-Medium      | 84.08     | 87.90     | **91.01** | 88.83     | 80.70     | 88.68     |
| DALL-E 3        | 83.50     | **90.97** | 89.61     | 88.39     | 90.58     | 89.83     |
| Flux.1-dev      | 83.79     | 85.80     | 86.79     | 89.98     | 90.04     | **89.90** |
| Janus-Pro-7B    | 84.19     | 86.90     | 88.90     | 89.40     | 89.32     | 89.48     |
| **CogView4-6B** | **85.13** | 83.85     | 90.35     | **91.17** | **91.14** | 87.29     |

#### GenEval

| モデル             | 全体       | 単一オブジェクト | 二つのオブジェクト | カウント     | 色        | 位置       | 色の属性     |
|-----------------|----------|----------|-----------|----------|----------|----------|----------|
| SDXL            | 0.55     | 0.98     | 0.74      | 0.39     | 0.85     | 0.15     | 0.23     |
| PixArt-alpha    | 0.48     | 0.98     | 0.50      | 0.44     | 0.80     | 0.08     | 0.07     |
| SD3-Medium      | 0.74     | **0.99** | **0.94**  | 0.72     | 0.89     | 0.33     | 0.60     |
| DALL-E 3        | 0.67     | 0.96     | 0.87      | 0.47     | 0.83     | 0.43     | 0.45     |
| Flux.1-dev      | 0.66     | 0.98     | 0.79      | **0.73** | 0.77     | 0.22     | 0.45     |
| Janus-Pro-7B    | **0.80** | **0.99** | 0.89      | 0.59     | **0.90** | **0.79** | **0.66** |
| **CogView4-6B** | 0.73     | **0.99** | 0.86      | 0.66     | 0.79     | 0.48     | 0.58     |

#### T2I-CompBench

| モデル             | 色          | 形          | テクスチャ      | 2D-空間      | 3D-空間      | 数量         | 非空間 Clip   | 複雑な3-in-1  |
|-----------------|------------|------------|------------|------------|------------|------------|------------|------------|
| SDXL            | 0.5879     | 0.4687     | 0.5299     | 0.2133     | 0.3566     | 0.4988     | 0.3119     | 0.3237     |
| PixArt-alpha    | 0.6690     | 0.4927     | 0.6477     | 0.2064     | 0.3901     | 0.5058     | **0.3197** | 0.3433     |
| SD3-Medium      | **0.8132** | 0.5885     | **0.7334** | **0.3200** | **0.4084** | 0.6174     | 0.3140     | 0.3771     |
| DALL-E 3        | 0.7785     | **0.6205** | 0.7036     | 0.2865     | 0.3744     | 0.5880     | 0.3003     | 0.3773     |
| Flux.1-dev      | 0.7572     | 0.5066     | 0.6300     | 0.2700     | 0.3992     | 0.6165     | 0.3065     | 0.3628     |
| Janus-Pro-7B    | 0.5145     | 0.3323     | 0.4069     | 0.1566     | 0.2753     | 0.4406     | 0.3137     | 0.3806     |
| **CogView4-6B** | 0.7786     | 0.5880     | 0.6983     | 0.3075     | 0.3708     | **0.6626** | 0.3056     | **0.3869** |

## 中国語テキストの正確性評価

| モデル             | 精度         | リコール       | F1スコア      | Pick@4     |
|-----------------|------------|------------|------------|------------|
| Kolors          | 0.6094     | 0.1886     | 0.2880     | 0.1633     |
| **CogView4-6B** | **0.6969** | **0.5532** | **0.6168** | **0.3265** |

## 推論モデル

### プロンプトの最適化

CogView4シリーズのモデルは長文の合成画像説明でトレーニングされていますが、テキストから画像生成を行う前に大規模言語モデルを使用してプロンプトをリライトすることを強くお勧めします。これにより生成品質が大幅に向上します。

[例のスクリプト](inference/prompt_optimize.py)を提供しています。このスクリプトを実行してプロンプトをリファインすることをお勧めします。
`CogView4` と `CogView3` モデルのプロンプト最適化には異なるfew-shotが使用されていることに注意してください。区別が必要です。

```shell
cd inference
python prompt_optimize.py --api_key "Zhipu AI API Key" --prompt {your prompt} --base_url "https://open.bigmodel.cn/api/paas/v4" --model "glm-4-plus" --cogview_version "cogview4"
```

### 推論モデル

`BF16` 精度でモデルを実行します：

```python
from diffusers import CogView4Pipeline
import torch

pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16).to("cuda")

# GPUメモリ使用量を減らすために開く
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

1. For using `BNB int4` to load `text encoder` and complete inference code annotations,
   check [here](inference/cli_demo_cogview4.py).
2. For using `TorchAO int8 or int4` to load `text encoder & transformer` and complete inference code annotations,
   check [here](inference/cli_demo_cogview4_int8.py).
3. For setting up a `gradio` GUI DEMO, check [here](inference/gradio_web_demo.py).
## Installation
```
git clone https://github.com/THUDM/CogView4
cd CogView4
git clone https://huggingface.co/THUDM/CogView4-6B
pip install -r inference/requirements.txt
```
## Quickstart
12G VRAM
```
MODE=1 python inference/gradio_web_demo.py
```
24G VRAM 32G RAM
```
MODE=2 python inference/gradio_web_demo.py
```
24G VRAM 64G RAM
```
MODE=3 python inference/gradio_web_demo.py
```
48G VRAM 64G RAM
```
MODE=4 python inference/gradio_web_demo.py
```

## ライセンス

このリポジトリのコードとCogView3モデルは [Apache 2.0](./LICENSE) のライセンスの下で提供されています。

コードの貢献を歓迎し、感謝します。貢献ガイドラインは [こちら](resources/contribute.md) で確認できます。
