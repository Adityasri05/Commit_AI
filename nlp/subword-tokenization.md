# Subword Tokenization for Language Models

Language models do not usually process raw words directly. A tokenizer converts text into token IDs from a finite vocabulary. **Subword tokenization** breaks text into reusable pieces so a model can represent common words efficiently while still handling unfamiliar words.

For example, a rare word may be represented as several frequent pieces instead of becoming one unknown token. The exact split depends on the tokenizer vocabulary and algorithm.

## Common algorithms

- **Byte Pair Encoding (BPE):** begins with small units and repeatedly merges frequent adjacent pairs. It is used by many language-model tokenizers.
- **WordPiece:** chooses subword units using a learned vocabulary objective; often associated with BERT-family tokenizers.
- **Unigram language model:** starts with a larger candidate vocabulary and removes pieces to optimize a probabilistic tokenization model; used in SentencePiece-based systems.

These methods are related but not interchangeable. A model is trained with a particular tokenizer and vocabulary.

## Hugging Face example

```python
from transformers import AutoTokenizer

model_id = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "Machine learning systems need reliable evaluation."
encoded = tokenizer(
    text,
    add_special_tokens=True,
    truncation=True,
    max_length=128,
)

print(encoded["input_ids"])
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
print(tokenizer.decode(encoded["input_ids"]))
```

Use the tokenizer intended for the model checkpoint. Swapping in a different vocabulary can make token IDs refer to the wrong learned embeddings.

## Important concepts

- **Vocabulary:** the mapping between token strings and integer IDs.
- **Special tokens:** markers such as beginning/end-of-sequence, classification, separator, or padding tokens.
- **Unknown handling:** some tokenizers use an unknown token; byte-level or character-aware approaches can represent a wider range of input.
- **Offsets:** tokenizers may return character offsets that help map predictions back to the original text.
- **Truncation:** inputs longer than the model limit may be cut; this can remove important context.
- **Token count:** token count is not the same as word count. It affects context-window usage, latency, and API cost.

## Practical cautions

1. Inspect tokenization for names, URLs, code, mixed scripts, and domain-specific vocabulary.
2. Do not assume a token corresponds to one word or one character.
3. Keep tokenizer and model revisions aligned.
4. Record truncation and padding settings in experiments.
5. For multilingual tasks, test each relevant language rather than judging quality from English examples only.

## Practice

Compare the tokenization of a short paragraph, a long technical identifier, an emoji, and a sentence in two languages. Record token counts and inspect where each tokenizer splits unfamiliar terms.

## References

- [Hugging Face: Tokenizers overview](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Hugging Face: Pretrained tokenizers](https://huggingface.co/docs/transformers/main_classes/tokenizer)

**Key takeaway:** Tokenization is part of the model interface. Use the matching tokenizer and measure token counts rather than estimating from words.