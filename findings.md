# Findings

## Repository Findings

- `minicpm_demo` implements full-duplex audio with private audio units and a
  learned listen/speak first token.
- `StreamDecoder.feed()` accepts arbitrary token embeddings and updates the
  KV cache.
- `duplex_prepare()` already feeds text embeddings into the same decoder, so a
  knowledge-context prefill hook is technically feasible.
- Current duplex generation has `force_listen_override` but no force-speak
  path.
- Backend unit processing is serialized with an operation lock and requires
  `finalize_unit()` before the next audio prefill.
- GPU 0 was idle at implementation start.
- Local environment has PyTorch, Transformers, scikit-learn, ONNX Runtime,
  librosa, FastAPI, and websockets.
- `faster-whisper`, FAISS, sentence-transformers, and Qdrant are not installed.

## Design Decisions

- Use character n-gram TF-IDF as the POC retrieval baseline. It works offline
  for Chinese and keeps retrieval testable without another model download.
- Keep ASR pluggable. Default to Transformers Whisper when enabled, with text
  input and deterministic mock backends for UI/protocol tests.
- Demonstrate the real duplex model without waiting for a learned knowledge
  token. Add an experimental injected knowledge unit and force the first
  generated token to `speak`; label it as experimental in the UI.
- Preserve the original code tree; only copied code inside
  `minicpm_rag_test/MiniCPMO45` and `minicpm_rag_test/core` may be adapted.

## Open Risks

- Mid-session text prefill is outside MiniCPM-o's original duplex training
  distribution and may produce unstable answers.
- The hard gate creates intentional silence while ASR/retrieval run.
- Whisper model weights may require a first-run download.
- The real 9B model startup can take tens of seconds and substantial VRAM.
