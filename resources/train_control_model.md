## Fine-Tuning Control Models (ControlNet)

### Principles

We increase the number of channels in the `latent` from `16` to `32`. Of these, an additional 16 channels are used for
the control model. The values of these channels will be used to control the output of the generator. We treat these
channel values as part of the `latent` and concatenate them with the `latent`. In this way, we can control the output of
the generator by adjusting the values of these channels.

### Server Requirements

- At least one `A100 GPU` is required. Using `zero2` for training, each card can handle a batch of 8.
- If you want to fine-tune the model in full, we recommend using `batchsize=128`.
- Linux operating system is required for installing `deepspeed`.

### Preparing the Dataset

In this example, we use [open_pose_controlnet](https://huggingface.co/datasets/raulc0399/open_pose_controlnet) for
fine-tuning.
You can also use your own dataset, but you will need to follow the `open_pose_controlnet` dataset format or modify the
dataloader accordingly.

**Note**

+ All images will be resized to a fixed size. Dynamic resolutions are not supported.
+ Dynamic-length tokens are not supported. Tokens in each batch will be padded to the maximum length.

### Start Training

1. Clone the source code and install [diffusers](https://github.com/huggingface/diffusers), then navigate to the
   fine-tuning directory:

```shell
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
cd examples/cogview4-control
```

2. Set up deepspeed and accelerate environments

Here is an example accelerate configuration file using zero2:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  # deepspeed_hostfile: hostfile # If using multi-machine multi-card training, prepare the hostfile configuration
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero_stage: 2
num_machines: 1
num_processes: 8 # 8 processes in total, write 16 if using two machines
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
rdzv_backend: static
same_network: true
tpu_env: [ ]
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Save this configuration as `accelerate_ds.yaml`.

3. Run the following command to start training:

```shell
accelerate launch --config_file=accelerate_ds.yaml train_control_cogview4.py \
  --pretrained_model_name_or_path="THUDM/CogView4-6B" \
  --dataset_name="raulc0399/open_pose_controlnet" \
  --output_dir="pose-control" \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --proportion_empty_prompts=0 \
  --learning_rate=5e-5 \
  --adam_weight_decay=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=1000 \
  --checkpointing_steps=100 \
  --max_train_steps=50000 \
  --validation_steps=100 \
  --validation_image "pose.jpg" \
  --validation_prompt "two friends sitting by each other enjoying a day at the park, full hd, cinematic" \
  --offload \
  --seed="0"
```

**Note**

- Training must be done using bf16 mixed precision or fp32. fp16 and fp8 are not supported.
- idation_image and validation_prompt need to be prepared and placed in the same directory. In this example, the first
  entry from the open_pose_controlnet dataset is used.

## Using the Fine-Tuned Weights

Assuming you used the results from `10000` steps, and your model resolution is `1024`:

## SFT

```python
from diffusers import CogView4ControlPipeline, CogView4Transformer2DModel
from controlnet_aux import CannyDetector
from diffusers.utils import load_image
import torch

transformer = CogView4Transformer2DModel.from_pretrained("pose-control/checkpoint-10000/transformer",
                                                         torch_dtype=torch.bfloat16).to("cuda:0")
pipe = CogView4ControlPipeline.from_pretrained("THUDM/CogView4-6B", transformer=transformer,
                                               torch_dtype=torch.bfloat16).to("cuda:0")

prompt = "two friends sitting by each other enjoying a day at the park, full hd, cinematic"
control_image = load_image("pose.jpg")
processor = CannyDetector()
control_image = processor(
    control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024
)
image = pipe(
    prompt=prompt,
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=3.5,
).images[0]

image.save("cogview4.png")
```
