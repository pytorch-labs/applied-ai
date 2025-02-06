
## Integration of Liger Kernel with torchtune

## Introduction to torchtune

torchtune is a PyTorch library for easily authoring, finetuning and experimenting with LLMs.

torchtune provides:

- PyTorch implementations of popular LLMs from Llama, Gemma, Mistral, Phi, and Qwen model families
- Hackable training recipes for full finetuning, LoRA, QLoRA, DPO, PPO, QAT, knowledge distillation, and more
- Out-of-the-box memory efficiency, performance improvements, and scaling with the latest PyTorch APIs
- YAML configs for easily configuring training, evaluation, quantization or inference recipes
- Built-in support for many popular dataset formats and prompt templates

You can read more about torchtune in the [GitHub repo](https://github.com/pytorch/torchtune)

## Installation

torchtune is tested with the latest stable PyTorch release as well as the preview nightly version. torchtune leverages
torchvision for finetuning multimodal LLMs and torchao for the latest in quantization techniques; you should install these as well.

&nbsp;

### Install stable release

```bash
# Install stable PyTorch, torchvision, torchao stable releases
pip install torch torchvision torchao
pip install torchtune
```

&nbsp;

### Install nightly release

```bash
# Install PyTorch, torchvision, torchao nightlies
pip install --pre --upgrade torch torchvision torchao --index-url https://download.pytorch.org/whl/nightly/cu121 # full options are cpu/cu118/cu121/cu124
pip install --pre --upgrade torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

You can also check out torchtune's [install documentation](https://pytorch.org/torchtune/main/install.html) for more information, including installing torchtune from source.


## Introduction to Liger Kernel

Liger Kernel is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU training throughput by 20% and reduces memory usage by 60%. We have implemented Hugging Face Compatible RMSNorm, RoPE, SwiGLU, CrossEntropy, FusedLinearCrossEntropy, and more to come.

## Installlation

### Dependencies

#### CUDA

- `torch >= 2.1.2`
- `triton >= 2.3.0`

#### ROCm

- `torch >= 2.5.0` Install according to the instruction in Pytorch official webpage.
- `triton >= 3.0.0` Install from pypi. (e.g. `pip install triton==3.0.0`)

### Optional Dependencies

- `transformers >= 4.x`: Required if you plan to use the transformers models patching APIs. The specific model you are working will dictate the minimum version of transformers.

> **Note:**
> Our kernels inherit the full spectrum of hardware compatibility offered by [Triton](https://github.com/triton-lang/triton).

For integrating changes specific to torchtune, you need to use the fork in this repo and build from source


```bash
cd Liger-Kernel

# Install Default Dependencies
# Setup.py will detect whether you are using AMD or NVIDIA
pip install -e .

```

## Integrating LCE Kernel with torchtune full finetuning recipe

We demonstrate integration of Liger Kernel with torchtune by running a full finetuning recipe with `meta-llama/Llama-3.2-1B`.  To make this integration happen, we have defined a custom full finetuning recipe, the details of the changes are mentioned below.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 tune run --nproc_per_node 4 recipes/full_finetune_distributed.py --config llama3_2/1B_full optimizer=torch.optim.AdamW optimizer.fused=True optimizer_in_bwd=False gradient_accumulation_steps=1  dataset.packed=True compile=True enable_activation_checkpointing=True tokenizer.max_seq_len=512  batch_size=128
```

One of the inputs to the LCE Kernel is the forward projection weights. torchtune is designed as a modular library with composable blocks. There is a `TransformerDecoder` [block](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/transformer.py#L322) where at the end of the block, we pass the final hidden state through a linear layer to get the final output. Since the linear layer is combined with the CE loss in LCE Kernel, we write a custom `forward` function for `TransformerDecoder` where we skip the computation through the linear layer.

In the full finetuning recipe, we override the model's forward method with this custom method

```python
import types
from liger_kernel.torchtune.modules.transformers import decoder_forward
self._model.forward = types.MethodType(decoder_forward, self._model)
```

We then pass the model's forward projection weights to calculate the loss with LCE Kernel

```python
from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)

self._loss_fn = LigerFusedLinearCrossEntropyLoss()

 current_loss = (
     self._loss_fn(
         self._model.output.tied_module.weight,
         logits,
         labels,
     )
     * current_num_tokens
 )
```
