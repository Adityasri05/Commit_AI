# LoRA and QLoRA for Parameter-Efficient Fine-Tuning

Full fine-tuning updates every model parameter. For a large language model, that can require substantial accelerator memory for weights, gradients, and optimizer states. Parameter-efficient fine-tuning (PEFT) reduces the number of trainable parameters.

## Low-Rank Adaptation (LoRA)

LoRA freezes a pretrained weight matrix W and learns a low-rank update:

    W' = W + (alpha / r) * B A

If W maps from d_in to d_out, A and B are small matrices whose inner rank r is much smaller than the original dimensions. The base matrix remains frozen; training updates only the adapter matrices.

Important settings include:

- **Rank (r):** adapter capacity. Larger ranks add trainable parameters and memory.
- **Alpha:** scales the adapter contribution.
- **Target modules:** specify which projections or layers receive adapters.
- **Dropout:** regularizes adapter updates.
- **Bias and module-saving options:** determine whether extra parameters are trained or retained.

LoRA is not guaranteed to match full fine-tuning. Choose target modules and rank using a validation set that reflects the intended task.

## QLoRA

QLoRA combines LoRA adapters with a quantized frozen base model. The original method uses 4-bit NormalFloat (NF4) quantization, double quantization of quantization constants, and paged optimizers to reduce memory pressure.

The base weights are stored in low precision, while adapter computation and optimizer behavior use configured compute dtypes. Exact support depends on hardware, library versions, and model architecture.

QLoRA reduces memory requirements; it does not make training free or remove the need to evaluate quality, latency, and compatibility.

## Hugging Face PEFT sketch

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
```

The target module names are architecture-specific. Inspect the model before selecting them. For QLoRA, load the base model using a supported 4-bit quantization configuration before attaching adapters, following the current Transformers and PEFT documentation.

## A careful fine-tuning workflow

1. Define the task and a held-out evaluation set.
2. Check data rights, privacy, duplicates, and train/evaluation contamination.
3. Establish a prompt-only or existing-model baseline.
4. Choose LoRA targets and rank; use QLoRA only when its quantization path is supported.
5. Track trainable parameter count, GPU memory, steps, and training loss.
6. Evaluate task quality, regressions, refusal behavior, and safety cases.
7. Save adapter configuration and base-model revision together.
8. Test adapter loading and deployment with the exact runtime.

## Common mistakes

- Copying target module names from a different model architecture.
- Evaluating on examples that appeared in fine-tuning data.
- Treating lower training loss as proof of better generalization.
- Forgetting tokenizer and base-model revisions.
- Merging adapters without testing numerical and runtime compatibility.
- Assuming a 4-bit model has identical output behavior to its full-precision counterpart.

## Practice

Fine-tune a small instruction model with LoRA on a narrow classification or formatting task. Compare prompt-only, LoRA, and—if hardware permits—QLoRA. Record quality, trainable parameters, peak memory, and inference latency.

## References

- [Hugging Face PEFT: LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora)
- [Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314)

**Key takeaway:** LoRA learns small low-rank updates while keeping the base model frozen. QLoRA adds a quantized base to reduce memory; validate the full training and deployment setup.