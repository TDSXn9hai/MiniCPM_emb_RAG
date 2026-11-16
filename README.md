# MiniCPM-o Duplex RAG Test

这是一个独立于数字人视频驱动的 A2A（audio-to-audio）RAG 实验站。
它复用 `minicpm_demo` 的 MiniCPM-o 模型代码和权重，但不会修改原目录。

## What It Tests

```text
浏览器麦克风
  -> 16 kHz float32 PCM
  -> Silero VAD
  -> 本地 Whisper ASR
  -> 字符 n-gram TF-IDF 检索
  -> 证据卡
  -> MiniCPM-o duplex knowledge unit
  -> force-speak
  -> 24 kHz float32 PCM
  -> 浏览器播放
```

主要验证点：

- 检索期间模型保持 listen。
- 证据注入完成前不放行回答。
- 注入完成后使用第一个 `speak` token 触发回答。
- 单个双工 unit 提前结束时，最多进行 5 个知识续答 unit。
- 用户打断会使尚未完成的检索结果失效。
- 文本问题可绕过 ASR，用于稳定复现和浏览器自动化测试。

## Important Limitation

MiniCPM-o 原始训练没有“会话中途注入知识文本 unit”这种格式。
本目录的 `duplex_inject_knowledge()` 是实验性适配：

1. 在 unit 边界把知识 prompt 作为文本 embedding 写入 KV Cache。
2. 用一个 synthetic knowledge unit 生成回答。
3. 对未完成回答做有限次 continuation unit。

它适合验证链路与交互，不应直接视为生产级事实性保证。

当前版本是“硬门禁”的 A2A POC：检索期间模型保持 listen，回答生成结束后
再恢复监听。浏览器可以发送打断并停止播放，但还没有实现同一时刻持续
处理新麦克风 unit 与生成回答并行的完整双工调度；那是下一阶段工作。

## Quick Start: Mock

Mock 模式不需要 GPU 或 9B 模型，用于前端和协议调试。

```bash
cd minicpm_rag_test
RAG_RUNTIME=mock RAG_ASR=manual \
  ../.venv/bin/python -m uvicorn server:app \
  --host 0.0.0.0 --port 8010
```

打开 `http://127.0.0.1:8010`。

也可以直接运行：

```bash
./start_mock.sh
```

## Quick Start: Real MiniCPM-o

模型权重默认从以下位置读取：

```text
../minicpm_demo/model
```

```bash
cd minicpm_rag_test
RAG_RUNTIME=minicpm \
RAG_ASR=whisper \
CUDA_VISIBLE_DEVICES=0 \
  ./start_real.sh
```

首次使用 Whisper 需要下载 `openai/whisper-small`。
ASR 模型只在第一次收到语音段时懒加载。
如果 `models/whisper-small/model.safetensors` 已存在，`start_real.sh`
会自动使用本地目录。

模型权重不包含在 Git 仓库中。运行前需要准备：

```text
../minicpm_demo/model/
models/whisper-small/
assets/vad_model/silero_vad.onnx
```

Silero VAD 权重可以从原项目 `minicpm_demo/assets/vad_model/` 复制，
或从 Silero VAD 官方发布包获取同名文件。

如果不希望下载 ASR，可以用文本问题测试真实模型回答：

```bash
RAG_RUNTIME=minicpm RAG_ASR=manual CUDA_VISIBLE_DEVICES=0 ./start_real.sh
```

## Environment Variables

| Variable | Default | Meaning |
|---|---|---|
| `RAG_HOST` | `0.0.0.0` | HTTP/WebSocket 监听地址 |
| `RAG_PORT` | `8010` | 服务端口 |
| `RAG_RUNTIME` | `mock` | `mock` or `minicpm` |
| `RAG_ASR` | `whisper` | `whisper`, `manual`, `none`, `mock` |
| `RAG_ASR_MODEL` | `openai/whisper-small` | Transformers Whisper 模型 |
| `RAG_ASR_LANGUAGE` | `zh` | Whisper 语言 |
| `RAG_MODEL_PATH` | `../minicpm_demo/model` | MiniCPM-o 权重 |
| `RAG_REF_AUDIO` | `assets/ref_audio/ref_minicpm_signature.wav` | TTS 参考音色 |
| `RAG_KNOWLEDGE_PATH` | `knowledge/sample.json` | 知识库 JSON |
| `RAG_VAD_SILENCE_MS` | `700` | VAD 静音结束阈值 |
| `RAG_VAD_THRESHOLD` | `0.75` | VAD 语音阈值 |

## Knowledge File

当前 POC 使用 JSON 文档列表：

```json
{
  "documents": [
    {
      "id": "travel-policy",
      "title": "差旅报销制度",
      "source": "travel-policy.pdf#page=8",
      "content": "上海属于一线城市，住宿标准为每晚六百元。"
    }
  ]
}
```

检索基线使用字符 2-4 gram TF-IDF，适合快速测试中文短文档。
后续可以替换为 hybrid dense/BM25 retrieval 和 reranker，接口保持不变。

当前 ASR 是 **VAD 控制的分段转写**：VAD 检测说完整后，把语音段交给
Whisper。它不是逐 token 的流式增量解码，但边界由流式 VAD 控制，
ASR 服务可以后续替换为 FunASR/WhisperLive 等真正的 streaming provider。

## WebSocket Protocol

Endpoint:

```text
ws://127.0.0.1:8010/ws/a2a
```

Client -> Server:

```json
{"type":"session.start"}
```

```json
{
  "type":"audio.append",
  "sequence":1,
  "audio":"<base64 float32 little-endian, 16 kHz mono>"
}
```

```json
{"type":"text.query","text":"上海出差住宿能报多少"}
```

```json
{"type":"interrupt"}
```

Server -> Client:

- `server.ready`
- `session.ready`
- `audio.ack`
- `vad.state`
- `state` with `listening`, `retrieving`, `answer_ready`
- `asr.final`
- `retrieval.result`
- `knowledge.injected`
- `answer.started`
- `answer.text`
- `answer.audio`
- `answer.done`
- `response.interrupted`
- `error`

## Tests

Unit and protocol tests:

```bash
../.venv/bin/python -m pytest tests -q
```

Mock browser test:

```bash
python3 /root/.comate/.baidu-cx/skills/webapp-testing/scripts/with_server.py \
  --server "RAG_RUNTIME=mock RAG_ASR=manual ../.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8010" \
  --port 8010 \
  -- ../.venv/bin/python tests/browser/test_a2a_ui.py
```

Real-model direct smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  ../.venv/bin/python tests/integration_real_model.py
```

Fake-microphone browser test:

```bash
python3 /root/.comate/.baidu-cx/skills/webapp-testing/scripts/with_server.py \
  --server "RAG_RUNTIME=mock RAG_ASR=mock RAG_MOCK_TRANSCRIPT='上海住宿能报多少' ../.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8012" \
  --port 8012 \
  -- ../.venv/bin/python tests/browser/test_mic_ui.py
```

Current verified result:

```text
上海出差住宿可以报销每晚六百元。
audio: 2.96s @ 24 kHz
knowledge_units: 3
```

已验证：

- 22 个单元/协议测试通过。
- mock 文本问题浏览器流程通过。
- Chromium fake microphone 音频输入、VAD、mock ASR、检索、回答通过。
- 真实 MiniCPM-o 文本问题、knowledge prefill、force-speak、24kHz 音频通过。
- 真实完整 A2A 通过：Chromium fake microphone → Silero VAD → 本地
  Whisper → 检索 → MiniCPM-o knowledge prefill → 24kHz 语音。

## Directory Layout

```text
minicpm_rag_test/
├── MiniCPMO45/             # copied model adapter, modified only in this copy
├── core/                   # copied processor/views
├── vad/                    # copied Silero VAD wrapper
├── assets/                 # VAD model and reference audio
├── knowledge/sample.json   # sample knowledge base
├── static/                 # browser experiment UI
├── rag_core.py             # retrieval and evidence card
├── gate.py                 # hard knowledge gate
├── knowledge_injection.py  # knowledge-unit prefill
├── engine.py               # ASR -> retrieval -> injection pipeline
├── runtime.py              # mock and real duplex runtime
├── server.py               # FastAPI A2A service
└── tests/                  # unit, protocol, browser, real-model tests
```
