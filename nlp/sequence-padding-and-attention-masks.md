# Sequence Padding and Attention Masks

Text examples have different lengths, but neural networks often process them in batches of equal-shaped tensors. **Padding** makes sequences in a batch the same length by adding a special padding token. An **attention mask** tells the model which positions are real input and which positions are padding.

## Padding a batch

Suppose a tokenizer produces:

- Example A: [101, 25, 88, 102]
- Example B: [101, 42, 102]

If the batch is padded to length four, the second example may become:

`[101, 42, 102, PAD]`

A common attention-mask convention is:

- 1 for a real token.
- 0 for a padding position.

So the second example's mask is `[1, 1, 1, 0]`.

Always follow the model and framework convention; not every custom implementation uses the same representation.

## Hugging Face example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased"
)

batch = tokenizer(
    ["short sentence", "a somewhat longer sentence for batching"],
    padding=True,
    truncation=True,
    return_tensors="pt",
)

print(batch["input_ids"].shape)
print(batch["attention_mask"])
```

Dynamic padding to the longest sequence in each batch avoids wasting compute on a fixed maximum length for every example. For stable accelerator shapes, some workloads pad to a multiple of a hardware-friendly size.

## Do not confuse two different masks

- **Padding mask:** prevents attention to padded positions.
- **Causal mask:** prevents a token from attending to future positions in autoregressive generation.

A decoder-only language model commonly needs causal attention and padding handling. The model implementation may combine or construct these masks internally, so avoid adding a second mask without checking the API.

## Padding labels during training

For token-level tasks, padded label positions should not contribute to the loss. In many Hugging Face token-classification and seq2seq workflows, the ignored label value is `-100`; verify the loss function and data collator being used.

For sequence classification, padding positions are usually handled by the attention mask, while the target is one label per sequence.

## Left vs right padding

Encoder models commonly use right padding. Some decoder-only generation setups use left padding so the final non-padding token aligns across a batch. The correct choice depends on the model and generation implementation.

## Common mistakes

- Forgetting to pass the attention mask when the model expects it.
- Treating padding token IDs as meaningful words.
- Masking future tokens with the padding mask instead of a causal mask.
- Including padded positions in token-level loss.
- Truncating long examples without checking whether the answer-bearing content was removed.

## Practice

Build a batch with three different sequence lengths. Print input IDs and masks, then verify that changing only the padded token IDs does not change the intended model attention behavior. Inspect how truncation affects a long example.

## References

- [Hugging Face: Padding and truncation](https://huggingface.co/docs/transformers/pad_truncation)
- [Hugging Face: Data collators](https://huggingface.co/docs/transformers/main_classes/data_collator)

**Key takeaway:** Padding creates equal-shaped batches; attention masks preserve the distinction between real tokens and padding. Causal masking is a separate concern.