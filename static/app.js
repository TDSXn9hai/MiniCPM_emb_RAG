const elements = {
  connectButton: document.querySelector("#connectButton"),
  connectionState: document.querySelector("#connectionState"),
  runtimeLabel: document.querySelector("#runtimeLabel"),
  stateLabel: document.querySelector("#stateLabel"),
  vadLabel: document.querySelector("#vadLabel"),
  audioState: document.querySelector("#audioState"),
  micButton: document.querySelector("#micButton"),
  micButtonLabel: document.querySelector("#micButtonLabel"),
  interruptButton: document.querySelector("#interruptButton"),
  queryForm: document.querySelector("#queryForm"),
  textQuery: document.querySelector("#textQuery"),
  sendQuery: document.querySelector("#sendQuery"),
  userStream: document.querySelector("#userStream p"),
  assistantStream: document.querySelector("#assistantStream p"),
  evidenceList: document.querySelector("#evidenceList"),
  retrievalMs: document.querySelector("#retrievalMs"),
  firstTextMs: document.querySelector("#firstTextMs"),
  audioSamples: document.querySelector("#audioSamples"),
  eventLog: document.querySelector("#eventLog"),
  waveform: document.querySelector("#waveform"),
  toast: document.querySelector("#toast"),
  steps: [...document.querySelectorAll(".pipeline span")],
};

const stateLabels = {
  offline: "离线",
  listening: "聆听",
  retrieving: "检索中",
  injecting: "注入中",
  answer_ready: "准备回答",
  speaking: "回答中",
  answering: "回答中",
};

let socket = null;
let systemReady = false;
let mic = null;
let playbackContext = null;
let nextPlaybackTime = 0;
let audioSources = [];
let queryStartedAt = 0;
let firstTextAt = 0;
let receivedAudioSamples = 0;
let currentAssistantText = "";
let toastTimer = null;

function setConnection(text, connected = false) {
  elements.connectionState.textContent = text;
  elements.connectionState.classList.toggle("is-connected", connected);
}

function setState(state) {
  elements.stateLabel.textContent = stateLabels[state] || state;
  const activeMap = {
    listening: "listening",
    retrieving: "retrieving",
    injecting: "injecting",
    answer_ready: "answering",
    speaking: "answering",
    answering: "answering",
    offline: "",
  };
  const active = activeMap[state] || "";
  for (const step of elements.steps) {
    step.classList.toggle("is-active", step.dataset.step === active);
  }
}

function toast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(
    () => elements.toast.classList.remove("is-visible"),
    4200,
  );
}

function appendEvent(event) {
  elements.eventLog.textContent =
    `${new Date().toLocaleTimeString()} ${JSON.stringify(event)}\n` +
    elements.eventLog.textContent;
  elements.eventLog.textContent = elements.eventLog.textContent.slice(0, 16000);
}

function connect() {
  if (socket) {
    socket.close();
  }
  systemReady = false;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws/a2a`);
  setConnection("连接中");
  elements.connectButton.disabled = true;

  socket.addEventListener("open", () => {
    setConnection("等待模型");
  });
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    appendEvent(message);
    handleEvent(message);
  });
  socket.addEventListener("close", () => {
    systemReady = false;
    setConnection("已断开");
    setState("offline");
    elements.connectButton.disabled = false;
    elements.micButton.disabled = true;
    elements.sendQuery.disabled = true;
    elements.interruptButton.disabled = true;
    stopMic();
  });
  socket.addEventListener("error", () => {
    setConnection("连接失败");
    toast("无法连接实验服务");
  });
}

function handleEvent(message) {
  switch (message.type) {
    case "server.ready":
      elements.runtimeLabel.textContent = `${message.runtime} / ${message.asr}`;
      socket.send(
        JSON.stringify({
          type: "session.start",
        }),
      );
      break;
    case "session.ready":
      systemReady = true;
      setConnection("已连接", true);
      setState("listening");
      elements.connectButton.disabled = false;
      elements.micButton.disabled = false;
      elements.sendQuery.disabled = false;
      elements.interruptButton.disabled = false;
      toast("实验台已就绪");
      break;
    case "state":
      setState(message.state);
      break;
    case "vad.state":
      elements.vadLabel.textContent = message.speaking ? "VAD / 说话中" : "VAD / 静默";
      if (message.speaking) {
        queryStartedAt = 0;
      } else if (!queryStartedAt) {
        queryStartedAt = performance.now();
        elements.retrievalMs.textContent = "—";
        elements.firstTextMs.textContent = "—";
      }
      break;
    case "asr.final":
      elements.userStream.textContent = message.text || "（空转写）";
      break;
    case "retrieval.result":
      renderEvidence(message.citations || []);
      if (queryStartedAt) {
        elements.retrievalMs.textContent = `${Math.round(
          performance.now() - queryStartedAt,
        )} ms`;
      }
      break;
    case "knowledge.injected":
      setState("answer_ready");
      toast(`已注入 ${message.evidence_chars} 字证据`);
      break;
    case "answer.started":
      currentAssistantText = "";
      firstTextAt = 0;
      elements.assistantStream.textContent = "";
      document.querySelector("#assistantStream").classList.add("is-live");
      if (!queryStartedAt) queryStartedAt = performance.now();
      break;
    case "answer.text":
      if (!firstTextAt) {
        firstTextAt = performance.now();
        elements.firstTextMs.textContent = `${Math.round(
          firstTextAt - queryStartedAt,
        )} ms`;
      }
      currentAssistantText += message.text || "";
      elements.assistantStream.textContent = currentAssistantText;
      setState("speaking");
      break;
    case "answer.audio":
      playAudio(message.audio, message.sample_rate || 24000);
      break;
    case "answer.done":
      document.querySelector("#assistantStream").classList.remove("is-live");
      setState("listening");
      queryStartedAt = 0;
      break;
    case "response.interrupted":
      document.querySelector("#assistantStream").classList.remove("is-live");
      setState("listening");
      stopPlayback();
      toast("已打断当前回答");
      break;
    case "audio.ack":
      elements.audioState.textContent = `seq ${message.sequence} · ${message.samples} 采样`;
      break;
    case "error":
      toast(message.message || message.code || "请求失败");
      break;
  }
}

function renderEvidence(citations) {
  elements.evidenceList.replaceChildren();
  if (!citations.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "本轮没有超过阈值的知识证据。";
    elements.evidenceList.append(empty);
    return;
  }
  for (const citation of citations) {
    const item = document.createElement("article");
    item.className = "evidence-item";
    const header = document.createElement("header");
    const title = document.createElement("h3");
    title.textContent = `${citation.id} · ${citation.title}`;
    const score = document.createElement("code");
    score.textContent = Number(citation.score).toFixed(3);
    header.append(title, score);
    const source = document.createElement("p");
    source.textContent = citation.source || "未提供来源";
    item.append(header, source);
    elements.evidenceList.append(item);
  }
}

async function toggleMic() {
  if (mic) {
    stopMic();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    const context = new AudioContext({ sampleRate: 16000 });
    await context.resume();
    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    const processor = context.createScriptProcessor(2048, 1, 1);
    const mute = context.createGain();
    mute.gain.value = 0;
    let pending = new Float32Array(0);

    processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const resampled = resampleLinear(
        input,
        context.sampleRate,
        16000,
      );
      const merged = new Float32Array(pending.length + resampled.length);
      merged.set(pending);
      merged.set(resampled, pending.length);
      pending = merged;
      while (pending.length >= 16000) {
        const chunk = pending.slice(0, 16000);
        pending = pending.slice(16000);
        sendAudioChunk(chunk);
      }
    };

    source.connect(analyser);
    source.connect(processor);
    processor.connect(mute);
    mute.connect(context.destination);
    mic = { stream, context, source, analyser, processor, mute, raf: 0 };
    elements.micButton.classList.add("is-active");
    elements.micButtonLabel.textContent = "关闭麦克风";
    elements.audioState.textContent = `采集 ${context.sampleRate} Hz`;
    drawWaveform();
    setState("listening");
  } catch (error) {
    toast(`无法打开麦克风：${error.message}`);
  }
}

function stopMic() {
  if (!mic) return;
  cancelAnimationFrame(mic.raf);
  mic.stream.getTracks().forEach((track) => track.stop());
  mic.processor.disconnect();
  mic.source.disconnect();
  mic.mute.disconnect();
  mic.context.close();
  mic = null;
  elements.micButton.classList.remove("is-active");
  elements.micButtonLabel.textContent = "打开麦克风";
  elements.audioState.textContent = "16 kHz PCM";
  drawIdleWaveform();
}

function resampleLinear(input, sourceRate, targetRate) {
  if (sourceRate === targetRate) return input;
  const ratio = sourceRate / targetRate;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i += 1) {
    const position = i * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const mix = position - left;
    output[i] = input[left] * (1 - mix) + input[right] * mix;
  }
  return output;
}

function sendAudioChunk(samples) {
  if (!systemReady || socket?.readyState !== WebSocket.OPEN) return;
  const bytes = new Uint8Array(
    samples.buffer,
    samples.byteOffset,
    samples.byteLength,
  );
  socket.send(
    JSON.stringify({
      type: "audio.append",
      sequence: Date.now(),
      audio: bytesToBase64(bytes),
    }),
  );
  elements.audioSamples.textContent = String(
    Number(elements.audioSamples.textContent || "0") + samples.length,
  );
}

function bytesToBase64(bytes) {
  let binary = "";
  const block = 0x8000;
  for (let i = 0; i < bytes.length; i += block) {
    binary += String.fromCharCode(...bytes.subarray(i, i + block));
  }
  return btoa(binary);
}

function base64ToFloat32(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength / 4,
  );
}

async function playAudio(base64, sampleRate) {
  if (!base64) return;
  if (!playbackContext) {
    playbackContext = new AudioContext();
  }
  await playbackContext.resume();
  const samples = base64ToFloat32(base64);
  const buffer = playbackContext.createBuffer(1, samples.length, sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = playbackContext.createBufferSource();
  source.buffer = buffer;
  source.connect(playbackContext.destination);
  const startAt = Math.max(
    playbackContext.currentTime + 0.03,
    nextPlaybackTime,
  );
  source.start(startAt);
  nextPlaybackTime = startAt + buffer.duration;
  audioSources.push(source);
  source.onended = () => {
    audioSources = audioSources.filter((item) => item !== source);
  };
}

function stopPlayback() {
  for (const source of audioSources) {
    try {
      source.stop();
    } catch {
      // Source may already have ended.
    }
  }
  audioSources = [];
  nextPlaybackTime = 0;
}

function drawIdleWaveform() {
  const canvas = elements.waveform;
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#45636a";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
}

function drawWaveform() {
  if (!mic) return;
  const canvas = elements.waveform;
  const ctx = canvas.getContext("2d");
  const data = new Uint8Array(mic.analyser.fftSize);
  mic.analyser.getByteTimeDomainData(data);
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#ce552e";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < data.length; i += 1) {
    const x = (i / (data.length - 1)) * width;
    const y = (data[i] / 255) * height;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  mic.raf = requestAnimationFrame(drawWaveform);
}

elements.connectButton.addEventListener("click", connect);
elements.micButton.addEventListener("click", toggleMic);
elements.interruptButton.addEventListener("click", () => {
  stopPlayback();
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "interrupt" }));
  }
});
elements.queryForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = elements.textQuery.value.trim();
  if (!text || !systemReady) return;
  queryStartedAt = performance.now();
  elements.retrievalMs.textContent = "—";
  elements.firstTextMs.textContent = "—";
  socket.send(JSON.stringify({ type: "text.query", text }));
});

setState("offline");
drawIdleWaveform();
