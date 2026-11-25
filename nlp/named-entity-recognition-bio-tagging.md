# Named Entity Recognition and BIO Tagging

Named Entity Recognition (NER) identifies spans of text that refer to categories such as people, organizations, locations, dates, or product names. It is a sequence-labeling task: the model assigns a tag to each token, then adjacent tags are converted into entity spans.

## BIO tagging

A common scheme uses:

- B-TYPE: beginning of an entity of TYPE;
- I-TYPE: a continuation token inside that entity;
- O: token outside any labeled entity.

Example:

    Ada       B-PER
    Lovelace  I-PER
    joined    O
    Analytical B-ORG
    Engines   I-ORG

The tags preserve boundaries that a plain entity label would lose. Other schemes include BIOES/BILOU, which mark entity endings and single-token entities explicitly.

## Tokenization alignment

Modern models often tokenize words into subwords. If a word is split into multiple pieces, the training pipeline must decide how to assign labels to those pieces. A common convention labels the first subword and masks continuation subwords from the loss with a special ignore index such as -100. Another convention propagates I-TYPE to continuation pieces. Use the same convention in training and decoding.

Special tokens such as CLS and SEP should not be scored as entities. Keep an offset mapping if the final output must refer to character spans in the original text.

## Hugging Face token-classification sketch

~~~python
from transformers import AutoTokenizer, AutoModelForTokenClassification

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint, num_labels=len(labels), id2label=dict(enumerate(labels)),
    label2id={name: i for i, name in enumerate(labels)}
)

batch = tokenizer(
    ["Ada Lovelace joined Analytical Engines."],
    truncation=True, padding=True, return_tensors="pt"
)
logits = model(**batch).logits
predicted_ids = logits.argmax(dim=-1)
~~~

The sketch shows inference only. Fine-tuning requires token-level labels aligned to tokenizer output, an attention mask, and ignored positions for special or intentionally unscored subword tokens.

## Evaluation

Token accuracy can be misleading because O tokens are often the majority. Prefer entity-level precision, recall, and F1 using exact span and type matches. Also inspect boundary errors, type confusions, and performance by entity category.

A valid decoder should handle malformed sequences consistently—for example, an I-ORG without a preceding B-ORG may be treated as a new ORG span or flagged according to the chosen scheme.

## Common pitfalls

- Splitting text into train and test after creating overlapping chunks from the same source.
- Misaligning word labels with subword tokens.
- Counting special tokens in evaluation.
- Ignoring nested or overlapping entities, which plain BIO tagging cannot represent directly.
- Assuming annotation guidelines are consistent across annotators and domains.

## Practice

Annotate ten sentences with people, organizations, and locations. Tokenize them with a pretrained tokenizer, align labels, and calculate entity-level F1. Review every boundary error manually.

## References

- [Hugging Face: Token classification](https://huggingface.co/docs/transformers/tasks/token_classification)
- [spaCy: Named Entity Recognition](https://spacy.io/usage/linguistic-features#named-entities)

**Key takeaway:** NER quality depends as much on span boundaries, annotation rules, and token-label alignment as on the underlying model.