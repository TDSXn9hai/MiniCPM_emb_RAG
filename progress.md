# Progress

## Session Log

### 2026-11-13

- Inspected original duplex, backend, decoder, VAD, and test code.
- Confirmed GPU 0 is free.
- Created `minicpm_rag_test/`.
- Copied `MiniCPMO45/`, `core/`, `vad/`, VAD asset, prompt assets, and
  reference audio.
- Removed only copied `__pycache__` and `.pyc` artifacts from the new tree.

## Verification Status

- Original tree: `git diff -- minicpm_demo` is empty.
- Unit/protocol tests: 22 passed.
- Mock text-query browser flow: passed.
- Chromium fake-microphone VAD + mock ASR flow: passed.
- Desktop and 390px responsive screenshots: passed without horizontal overflow.
- Real MiniCPM-o knowledge injection + force-speak + 24kHz audio: passed.
- Real browser text-query flow against MiniCPM-o: passed.
- Local Whisper download: 966,995,080-byte safetensors assembled and validated;
  SHA256 matches `1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b`.
- VAD + local Whisper: detected a 3.39s speech segment and transcribed
  `当出现植物大战僵尸的时候提醒我`.
- Full real A2A browser test: fake microphone -> VAD -> Whisper -> retrieval ->
  MiniCPM-o knowledge injection -> 24kHz audio answer, passed.
- Temporary Whisper range-download files were removed after SHA validation.
- Final regression: 22 pytest tests, mock text browser, mock fake-mic browser,
  real text browser, and real fake-mic browser all passed.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| `rg` not installed | 1 | Used `find` and `grep`. |
| Playwright Chromium missing `libatk` | 1 | Installed Chromium system dependencies through the configured proxy. |
| Chromium install-deps initially had no proxy | 1 | Re-ran apt with explicit `Acquire::http::Proxy`. |
| Real model ignored force-speak on first turn | 1 | Set `force_listen_count=0` for the knowledge runtime and kept an explicit force-speak override. |
| Single knowledge unit ended early | 1 | Added continuation units with no-repeat prompts; final safety cap is 8 units. |
| Continuation repeated the user question | 1 | Added an explicit “do not repeat the question” instruction. |
| VAD test had no trailing silence | 1 | Appended silence to the test utterance. |
| Mock browser timing stayed at 0 ms on the audio path | 1 | Record query start time on VAD speaking-false transition. |
| Real-browser test matched the assistant placeholder | 1 | Wait for the `answer.done` event and reject placeholder text. |
| Local Whisper emitted decoder/attention warnings | 1 | Passed an attention mask and explicit decoder prompt IDs. |
