# Text Preprocessing for NLP

Text preprocessing converts raw language into a representation suitable for analysis or modeling. The right steps depend on the task and model.

## Common operations
- Normalization: consistent casing or Unicode handling.
- Tokenization: split text into words, subwords, or other tokens.
- Stop-word filtering: remove selected frequent terms when appropriate.
- Stemming: reduce words using heuristic rules.
- Lemmatization: map words to dictionary forms.
- N-grams: represent sequences of adjacent tokens.

## Classical representations
- Bag of Words counts token occurrences.
- TF-IDF weights terms by their frequency in a document relative to the corpus.
- Word embeddings map tokens to dense vectors.

## Modern transformer models
Subword tokenizers often handle rare words and morphology without requiring aggressive manual cleaning. Preserve punctuation and casing when the pretrained model expects them.

## Avoid leakage
Fit vocabulary and IDF statistics on training data only. Do not use test documents to build a training vocabulary.

## Exercise
Compare a TF-IDF + logistic regression baseline with a pretrained transformer classifier on a sentiment dataset. Track preprocessing, validation setup, and inference cost.