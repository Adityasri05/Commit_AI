# Sequence-to-Sequence Models and Teacher Forcing

Sequence-to-sequence (seq2seq) models map an input sequence to an output sequence. Examples include translation, summarization, speech recognition, and structured text generation.

## Encoder-decoder structure

A classic encoder-decoder model has two components:

1. **Encoder:** reads the source tokens and produces hidden representations.
2. **Decoder:** generates target tokens one step at a time, conditioned on the encoded source and previously generated target tokens.

An attention mechanism lets the decoder refer to different encoder positions instead of relying on one fixed-size summary vector. Transformer encoder-decoder models use attention throughout; recurrent seq2seq models often use recurrent layers.

## Teacher forcing

During training, the decoder needs a previous token as input. With teacher forcing, the next decoder step receives the **ground-truth** previous target token rather than the model's previous prediction.

For a target sequence:

    <BOS> I like machine learning <EOS>

the decoder is trained with shifted inputs and targets:

| Decoder input | Expected next token |
|---|---|
| <BOS> | I |
| I | like |
| like | machine |
| machine | learning |
| learning | <EOS> |

The loss is computed between each step's logits and the corresponding target token. Padding positions should be excluded from the loss, commonly with an ignore index.

## Why teacher forcing helps—and what it changes

Teacher forcing gives the decoder a reliable previous token during training, making optimization easier. At inference time, the true target is unavailable: the model must consume its own generated tokens.

This mismatch is called **exposure bias**. A mistaken early token can place the decoder in a state it rarely saw during training, and errors may accumulate.

Scheduled sampling gradually replaces some ground-truth decoder inputs with model-generated tokens during training. It is one proposed mitigation, not a universal fix; it can introduce its own optimization and statistical issues. Compare alternatives empirically.

## Training and inference checklist

- Add explicit beginning-of-sequence and end-of-sequence tokens where the architecture expects them.
- Shift decoder inputs and labels by one position; do not train the model to copy the current token.
- Apply causal masking in autoregressive decoders so future target tokens cannot leak into predictions.
- Mask padding in attention and loss calculations.
- At inference, stop on EOS or a maximum length.
- Compare greedy decoding and beam search on task-appropriate metrics.
- Evaluate complete sequences, not only token-level training loss.

## Pseudocode

```python
# target includes BOS and EOS.
decoder_input = target[:, :-1]
labels = target[:, 1:]

logits = model(
    source_ids=source_ids,
    decoder_input_ids=decoder_input,
    source_padding_mask=source_padding_mask,
)

loss = cross_entropy(
    logits.reshape(-1, vocab_size),
    labels.reshape(-1),
    ignore_index=pad_id,
)
```

This sketch assumes the model implements the appropriate decoder attention masks and returns logits aligned with decoder_input positions.

## Practice

Train a small character-level seq2seq model. First use full teacher forcing. Then compare a scheduled-sampling experiment. Track validation exact-match or BLEU/ROUGE as appropriate, generation length, and common early-token errors.

## References

- [PyTorch: NLP From Scratch — Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**Key takeaway:** Teacher forcing stabilizes decoder training by supplying true previous tokens, but inference must operate on the model's own outputs. Align shifted targets and masks carefully.