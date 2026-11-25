# Multimodal Large Language Models

A multimodal large language model (MLLM) processes or generates more than one modality, such as text, images, audio, or video. A text-only model receives token IDs from a text tokenizer; a multimodal system must also turn non-text signals into representations the language model can use.

## Common architecture patterns

- **Separate encoders:** An image, audio, or video encoder converts raw input into feature vectors.
- **Projection or adapter layers:** Map modality-specific features into a space compatible with the language model.
- **Cross-attention:** Text tokens can attend to modality features through dedicated attention layers.
- **Unified token streams:** Some systems represent text and non-text content as sequences of modality-specific tokens.

Architectures differ. A model that accepts image embeddings is not necessarily trained to generate images, and audio input/output may use different encoders, decoders, and objectives.

## Image inputs

Images may be resized, cropped, divided into patches, or encoded at multiple resolutions. High-resolution inputs can preserve small text and details but increase compute and context usage. A user-facing application should document supported formats, resolution limits, and how multiple images are handled.

## Audio and video

Audio pipelines may use spectrograms, learned audio tokens, speech encoders, or text representations from an ASR system. Some models support speech perception, speech generation, or both. Video commonly requires sampling frames and representing timestamps; sampling too sparsely can miss short events, while dense sampling increases cost.

## Fusion choices

- **Early fusion:** Combine modality representations before most of the reasoning layers.
- **Late fusion:** Process modalities separately and combine higher-level outputs.
- **Cross-modal fusion:** Let one modality attend to another during intermediate layers.

The choice affects alignment, latency, data requirements, and the ability to handle missing modalities.

## Practical evaluation

Build a test set that covers each modality and combinations of modalities. Include:
- visual counting, spatial relations, OCR, and charts for images;
- accents, noise, speaker changes, and timestamps for audio;
- temporal ordering and brief events for video;
- questions that require combining text with visual or audio evidence.

Score factual accuracy, grounding to the input, instruction following, latency, and robustness. Keep examples with uncertain or unanswerable inputs so the system can be evaluated for appropriate abstention.

## Safety and security

Treat text found inside an image, audio transcript, or video frame as untrusted content. It may contain prompt-injection attempts. Do not let an embedded instruction override system policy or tool authorization. Avoid sending sensitive media to services without appropriate consent and retention controls. Provide human review for high-impact interpretations.

## Common pitfalls

- Assuming every modality is represented as ordinary text tokens.
- Ignoring resolution, frame rate, or audio sampling choices.
- Evaluating only text quality while overlooking perception errors.
- Treating a fluent description as proof that the model correctly perceived the input.
- Failing to track source timestamps or regions when evidence must be auditable.

## Practice

Create a small evaluation set with 20 images: charts, receipts, diagrams, and ordinary photographs. Ask the same questions of a text-only baseline with OCR and a multimodal model. Compare exact facts, unsupported claims, and latency.

## References

- [Hugging Face: Multimodal models](https://huggingface.co/docs/transformers/tasks)
- [LLaVA: Large Language and Vision Assistant](https://arxiv.org/abs/2304.08485)

**Key takeaway:** Multimodal systems must align and evaluate each input modality, not simply generate convincing text about it.