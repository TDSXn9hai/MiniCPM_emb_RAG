# MiniCPM RAG A2A Test Plan

## Goal

Build an isolated audio-to-audio (A2A) experiment under
`minicpm_rag_test/` that demonstrates:

1. VAD-delimited user speech.
2. Local speech-to-text.
3. Knowledge retrieval and evidence-card construction.
4. Knowledge context injection into the MiniCPM-o duplex session.
5. A hard answer gate: the model does not answer until retrieval/injection is
   ready.
6. A simple browser UI for microphone input, live state, transcript, evidence,
   latency, and streamed answer audio.

`minicpm_demo/` is source-only input and must not be modified.

## Constraints

- No MuseTalk or digital-human video driver.
- Work only inside `minicpm_rag_test/`.
- Model weights are referenced from `../minicpm_demo/model`; they are not
  copied.
- Keep the protocol testable without loading the 9B model by providing a mock
  runtime.
- The real model path must remain available for a final integration check.

## Phases

### Phase 1: Scaffold and baselines

- [x] Copy only required model adapters, VAD, and assets.
- [x] Add planning/status files and project documentation.
- [x] Define A2A WebSocket protocol and state model.

### Phase 2: Retrieval and gating (TDD)

- [x] Write failing tests for character n-gram retrieval.
- [x] Implement retrieval and evidence-card formatting.
- [x] Write failing tests for the hard knowledge gate and interruption reset.
- [x] Implement the gate state machine.

### Phase 3: ASR and duplex runtime (TDD)

- [x] Write failing tests for ASR provider selection and transcript cleanup.
- [x] Implement local Transformers Whisper ASR plus manual/mock fallback.
- [x] Run local Whisper on a real Chinese speech sample.
- [x] Write failing tests for knowledge context injection protocol.
- [x] Implement a mock duplex runtime.
- [x] Add experimental MiniCPM-o duplex runtime with `force_speak`.

### Phase 4: Server and frontend

- [x] Write failing WebSocket protocol tests.
- [x] Implement FastAPI server and static frontend.
- [x] Verify VAD-only, no-model browser flow.

### Phase 5: Verification

- [x] Run unit tests with pristine output.
- [x] Run browser tests with Playwright.
- [x] Run a real MiniCPM-o smoke test on GPU if available.
- [x] Run VAD + local Whisper + retrieval integration.
- [x] Run real microphone-to-answer browser integration.
- [x] Document how to launch and what remains experimental.

## Acceptance Criteria

- Original tree remains unchanged apart from already-existing untracked files.
- `pytest` passes for retrieval, gate, ASR fallback, protocol, and mock runtime.
- Browser UI shows `listening -> retrieving -> injecting -> answering`.
- A fake speech segment produces a retrieved evidence card and an answer.
- Interrupting during retrieval invalidates the pending answer.
- The real runtime can be selected without code edits.
- No digital-human endpoints or MuseTalk dependencies are required.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| None yet | - | - |
