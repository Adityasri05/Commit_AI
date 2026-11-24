# Connectionist Temporal Classification (CTC) for Sequence Recognition

Connectionist Temporal Classification (CTC) is a loss function for sequence tasks where the input and target are not aligned frame by frame. A speech recognizer, for example, receives acoustic frames but usually has only a transcript, not the exact time position of each character.

CTC lets a model learn an alignment by summing over many possible frame-level paths.

## Why alignment is difficult

An audio clip may contain hundreds of frames but only a few transcript characters. The model must learn which frames correspond to each character and which frames represent no emitted character.

CTC adds a special **blank** symbol. A path is converted to an output sequence by:
1. Merging consecutive repeated symbols.
2. Removing blank symbols.

For example, the path A, A, blank, B, B collapses to AB. The path A, blank, A collapses to AA because the blank separates the repeated labels.

## CTC objective

Let x be the input sequence and y the target label sequence. Many frame-level paths π can collapse to the same target y. CTC sums their probabilities:

\[
P(y \mid x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} P(\pi \mid x)
\]

The loss is the negative log likelihood:

\[
L_{\mathrm{CTC}} = -\log P(y \mid x)
\]

The summation is computed efficiently using dynamic programming rather than enumerating every path.

## Model architecture

A common system contains:
- An acoustic encoder that converts audio frames into hidden states.
- A linear projection that produces a probability distribution over vocabulary symbols plus blank at every frame.
- CTC loss during training.
- Greedy or beam-search decoding during inference.

The encoder may be a CNN, RNN, Transformer, or Conformer-style network.

## PyTorch loss example

PyTorch CTCLoss expects log probabilities with shape [time, batch, classes]. Target sequences are concatenated or padded according to the chosen representation.

~~~python
import torch
import torch.nn as nn

ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

# logits: [time, batch, classes], including blank at class 0
log_probs = logits.log_softmax(dim=-1)

# targets: concatenated target labels, with no blank symbols
# input_lengths: valid encoder time steps for each item
# target_lengths: number of labels in each target
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
~~~

Check that input_lengths describe the encoder output length, not necessarily the raw audio frame count. If the encoder downsamples time, the lengths must reflect that reduction.

## Decoding

- **Greedy decoding:** Select the most likely symbol at each frame, then collapse repeats and blanks. It is fast but may not find the most probable output sequence.
- **Beam search:** Maintains multiple candidate prefixes and can combine acoustic scores with a language model. It is more expensive but often improves recognition.

A language model can help resolve ambiguous sequences, but overly strong language-model scores may cause plausible words to override the audio evidence.

## Common pitfalls

- Forgetting the blank class or using the wrong blank index.
- Passing probabilities instead of log probabilities to PyTorch CTCLoss.
- Incorrect input lengths after temporal downsampling.
- Targets that contain blank symbols.
- Assuming greedy decoding is globally optimal.
- Ignoring repeated-label alignment requirements.
- Comparing word error rate (WER) without normalizing text consistently.

## Evaluation

For speech recognition, Word Error Rate is:

\[
WER = \frac{S + D + I}{N}
\]

where S is substitutions, D deletions, I insertions, and N is the number of words in the reference transcript. Character Error Rate is useful for languages or tasks where character-level errors are more meaningful.

Always document normalization rules for punctuation, casing, numbers, and language-specific tokenization.

## Mini-project

Build a small CTC recognizer on a public speech dataset. Plot frame-level predictions for a few examples, compare greedy and beam decoding, and report WER with a documented text-normalization procedure.

## References

- Graves et al., *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks* (2006): https://www.cs.toronto.edu/~graves/icml_2006.pdf
- PyTorch CTCLoss documentation: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
