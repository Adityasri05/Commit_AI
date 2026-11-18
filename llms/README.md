# Transformers and Large Language Models

A Large Language Model (LLM) is a neural network trained on large text datasets to model language. Many modern LLMs use the Transformer architecture.

## Tokens and embeddings
Text is split into tokens. A tokenizer maps tokens to IDs, and an embedding layer maps those IDs to vectors that a model can process.

## Self-attention
Self-attention lets each token build a representation using information from other tokens in the sequence. Attention uses queries (Q), keys (K), and values (V):

Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V

This helps the model represent context and relationships between tokens.

## Transformer components
- Multi-head attention learns different attention patterns.
- Feed-forward layers transform token representations.
- Residual connections and normalization support stable training.
- Positional information helps represent token order.

## How text generation works
1. The prompt is tokenized.
2. The model processes the token sequence.
3. It estimates a probability distribution for the next token.
4. A decoding strategy selects a token.
5. Generation repeats until a stopping condition is met.

## Important limitations
LLMs can produce plausible but incorrect statements, reflect biases in training data, and struggle with information outside their context or knowledge. Verify important claims and provide reliable source material when factual grounding matters.