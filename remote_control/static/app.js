"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const connectionBadge = $("#connectionBadge");
const connectionText = $("#connectionText");
const currentCommand = $("#currentCommand");
const joystick = $("#joystick");
const joystickKnob = $("#joystickKnob");
const linearValue = $("#linearValue");
const angularValue = $("#angularValue");
const stopButton = $("#stopButton");
const estopButton = $("#estopButton");
const resetEstopButton = $("#resetEstopButton");
const voiceButton = $("#voiceButton");
const cameraCanvas = $("#cameraCanvas");
const cameraContext = cameraCanvas.getContext("2d", { alpha: false, desynchronized: true });
const cameraNative = $("#cameraNative");
const cameraPlaceholder = $("#cameraPlaceholder");
const cameraPlaceholderText = $("#cameraPlaceholderText");
const cameraStatus = $("#cameraStatus");
const depthToggle = $("#depthToggle");
const armControls = $("#armControls");
const taskForm = $("#taskForm");
const taskInput = $("#taskInput");
const taskFeedback = $("#taskFeedback");
const heardText = $("#heardText");
const taskReply = $("#taskReply");
const taskMessage = $("#taskMessage");
const taskState = $("#taskState");
const cancelTaskButton = $("#cancelTaskButton");
const displayFps = $("#displayFps");
const renderFps = $("#renderFps");
const frameAge = $("#frameAge");
const activeCamera = $("#activeCamera");
const droppedFrames = $("#droppedFrames");

const DRIVE_PERIOD_MS = 90;
const DEAD_ZONE = 0.12;
let connected = false;
let emergencyStopped = false;
let joystickActive = false;
let activePointer = null;
let driveLinear = 0;
let driveAngular = 0;
let driveRequestActive = false;
let armSide = "left";
let capabilitiesLoaded = false;
let cameras = new Map();
let selectedCamera = "head-left";
let cameraMode = "rgb";
let cameraLoopGeneration = 0;
let cameraAbortController = null;
let displayedFrameTimes = [];
let pendingCameraFrame = null;
let cameraDecodeActive = false;
let clientDroppedFrames = 0;
let cameraFallbackTimer = null;
let nativeCameraActive = false;
let lastStatusAt = 0;
let recognition = null;
let speechActive = false;
let speechStarting = false;
let speechPointer = null;
let stopSpeechWhenStarted = false;
let pendingTranscript = "";
let speechFailed = false;

cameraContext.imageSmoothingEnabled = true;
cameraContext.imageSmoothingQuality = "low";
const isIosWebKit = /iP(?:hone|ad|od)/.test(navigator.userAgent || "")
  && /WebKit/.test(navigator.userAgent || "");

async function api(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    ...options,
    headers: options.body
      ? { "Content-Type": "application/json", ...(options.headers || {}) }
      : options.headers,
  });
  let result = {};
  try { result = await response.json(); } catch (_) { /* non-JSON failure */ }
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}

function post(path, payload = {}) {
  return api(path, { method: "POST", body: JSON.stringify(payload) });
}

function setConnection(isConnected) {
  const changed = connected !== isConnected;
  connected = isConnected;
  connectionBadge.className = `connection ${isConnected ? "connection--online" : "connection--offline"}`;
  connectionText.textContent = isConnected ? "CONNECTED" : "DISCONNECTED";
  if (!isConnected) {
    cameraPlaceholderText.textContent = "Simulation unavailable";
    cameraStatus.textContent = "The phone cannot reach the Butler[bot] simulation.";
    cameraStatus.classList.add("is-error");
  }
  if (changed) {
    if (isConnected) startCameraLoop();
    else stopCameraLoop();
  }
}

function setFeedback(message, isError = false) {
  taskFeedback.textContent = message || "";
  taskFeedback.classList.toggle("is-error", isError);
}

function updateSafety(status) {
  emergencyStopped = Boolean(status.emergency_stop);
  document.body.classList.toggle("is-estopped", emergencyStopped);
  estopButton.classList.toggle("hidden", emergencyStopped);
  resetEstopButton.classList.toggle("hidden", !emergencyStopped);
  currentCommand.textContent = status.fault || status.reason || "Ready";
  $("#backendLabel").textContent = `Robot backend: ${status.capabilities?.backend || "unknown"}`;
  $("#watchdogLabel").textContent = `Safety watchdog: ${status.watchdog_ms || "?"} ms`;
}

function updateTask(task = {}) {
  const state = task.state || "idle";
  taskState.textContent = state.toUpperCase();
  taskState.className = `task-state task-state--${state}`;
  heardText.textContent = task.command || "No command yet";
  taskReply.textContent = task.reply || (state === "idle" ? "Waiting for a command..." : "Working...");
  taskMessage.textContent = task.message || "Idle";
  cancelTaskButton.disabled = !task.active;
  const llmState = task.source === "model"
    ? "Local LLM: connected"
    : task.llm_error
      ? `Local LLM: unavailable (${task.llm_error})`
      : "Local LLM: waiting";
  $("#llmLabel").textContent = llmState;
  if (task.llm_error && task.active) {
    setFeedback("The local LLM was unavailable; the existing command fallback is being used.", true);
  }
}

function loadCapabilities(status) {
  if (capabilitiesLoaded || !status.capabilities) return;
  const listed = status.capabilities.cameras || [];
  cameras = new Map(listed.map((camera) => [camera.id, camera]));
  $$("[data-camera-id]").forEach((button) => {
    const available = cameras.has(button.dataset.cameraId);
    button.disabled = !available;
    if (available) button.title = cameras.get(button.dataset.cameraId).label;
  });
  buildArmControls(status.capabilities.arm_joints || []);
  capabilitiesLoaded = true;
  selectCamera(selectedCamera);
}

async function pollStatus() {
  try {
    const status = await api("/api/status");
    lastStatusAt = Date.now();
    setConnection(Boolean(status.connected));
    updateSafety(status);
    updateTask(status.task);
    loadCapabilities(status);
    updateArmPositions(status.telemetry?.arm_positions || {});
    updateServerCameraDiagnostics(status.camera);
  } catch (_) {
    if (Date.now() - lastStatusAt > 1400) {
      setConnection(false);
      centerJoystick();
      cameraPlaceholder.classList.remove("hidden");
      cameraPlaceholderText.textContent = "Simulation unavailable";
      cameraStatus.textContent = "Connection to the simulation was lost.";
      cameraStatus.classList.add("is-error");
      currentCommand.textContent = "Connection lost";
    }
  }
}

function updateServerCameraDiagnostics(camera = {}) {
  if (!nativeCameraActive) return;
  renderFps.textContent = Number(camera.render_fps || 0).toFixed(1);
  frameAge.textContent = camera.frame_age_ms == null
    ? "-- ms"
    : `${Math.round(camera.frame_age_ms)} ms`;
  droppedFrames.textContent = String(camera.dropped_frames || 0);
}

function buildArmControls(joints) {
  armControls.replaceChildren();
  for (const joint of joints) {
    const row = document.createElement("div");
    row.className = "arm-control";
    row.dataset.side = joint.side;
    const label = document.createElement("label");
    label.htmlFor = `joint-${joint.name}`;
    label.textContent = joint.label;
    const output = document.createElement("output");
    output.id = `output-${joint.name}`;
    output.textContent = `0.00 ${joint.unit}`;
    const slider = document.createElement("input");
    slider.type = "range";
    slider.id = `joint-${joint.name}`;
    slider.min = joint.minimum;
    slider.max = joint.maximum;
    slider.step = joint.step;
    slider.value = Math.max(joint.minimum, Math.min(joint.maximum, 0));
    slider.addEventListener("input", () => {
      output.textContent = `${Number(slider.value).toFixed(2)} ${joint.unit}`;
    });
    slider.addEventListener("change", () => sendArmTarget(joint.name, slider.value, joint.unit));
    row.append(label, output, slider);
    armControls.append(row);
  }
  if (!joints.length) armControls.innerHTML = '<p class="empty-state">Arm control is unavailable.</p>';
  showArmSide();
}

function showArmSide() {
  $$(".arm-control").forEach((row) => row.classList.toggle("hidden", row.dataset.side !== armSide));
}

function updateArmPositions(positions) {
  for (const [joint, value] of Object.entries(positions)) {
    const slider = $(`#joint-${joint}`);
    if (slider && document.activeElement !== slider) {
      slider.value = value;
      const output = $(`#output-${joint}`);
      const unit = joint.endsWith("0") ? "m" : "rad";
      output.textContent = `${Number(value).toFixed(2)} ${unit}`;
    }
  }
}

async function sendArmTarget(joint, value, unit) {
  try {
    const result = await post("/api/arm", { joint, value: Number(value) });
    $(`#output-${joint}`).textContent = `${Number(result.target).toFixed(2)} ${unit}`;
    setFeedback("Manual arm control active; autonomous task cancelled.");
  } catch (error) { setFeedback(error.message, true); }
}

function updateJoystick(event) {
  const rect = joystick.getBoundingClientRect();
  const radius = rect.width / 2;
  let x = (event.clientX - (rect.left + radius)) / radius;
  let y = (event.clientY - (rect.top + radius)) / radius;
  const magnitude = Math.hypot(x, y);
  if (magnitude > 1) {
    x /= magnitude;
    y /= magnitude;
  }
  const activeMagnitude = Math.hypot(x, y);
  if (activeMagnitude < DEAD_ZONE) {
    driveLinear = 0;
    driveAngular = 0;
    x = 0;
    y = 0;
  } else {
    const scaled = (activeMagnitude - DEAD_ZONE) / (1 - DEAD_ZONE);
    driveLinear = (-y / activeMagnitude) * scaled;
    driveAngular = (-x / activeMagnitude) * scaled;
  }
  joystickKnob.style.transform = `translate(calc(-50% + ${x * radius * 0.68}px), calc(-50% + ${y * radius * 0.68}px))`;
  linearValue.textContent = driveLinear.toFixed(2);
  angularValue.textContent = driveAngular.toFixed(2);
}

function centerJoystick() {
  joystickActive = false;
  activePointer = null;
  driveLinear = 0;
  driveAngular = 0;
  joystick.classList.remove("is-active");
  joystickKnob.style.transform = "translate(-50%, -50%)";
  linearValue.textContent = "0.00";
  angularValue.textContent = "0.00";
}

async function sendDrive() {
  if (!joystickActive || driveRequestActive || emergencyStopped || !connected) return;
  driveRequestActive = true;
  try {
    await post("/api/drive", { linear: driveLinear, angular: driveAngular });
  } catch (_) {
    setConnection(false);
    centerJoystick();
  } finally {
    driveRequestActive = false;
  }
}

async function stopNow(reason = "Stopped") {
  centerJoystick();
  currentCommand.textContent = reason;
  try {
    await post("/api/stop", {});
    await pollStatus();
  } catch (_) { setConnection(false); }
}

function beaconStop() {
  centerJoystick();
  navigator.sendBeacon("/api/stop", new Blob(["{}"], { type: "application/json" }));
}

joystick.addEventListener("pointerdown", (event) => {
  if (!connected || emergencyStopped) return;
  event.preventDefault();
  activePointer = event.pointerId;
  joystick.setPointerCapture(event.pointerId);
  joystickActive = true;
  joystick.classList.add("is-active");
  updateJoystick(event);
  sendDrive();
});
joystick.addEventListener("pointermove", (event) => {
  if (joystickActive && event.pointerId === activePointer) updateJoystick(event);
});
["pointerup", "pointercancel", "lostpointercapture"].forEach((name) => {
  joystick.addEventListener(name, (event) => {
    if (activePointer === null || event.pointerId === activePointer) stopNow("Joystick released");
  });
});
setInterval(sendDrive, DRIVE_PERIOD_MS);

stopButton.addEventListener("click", () => stopNow("Stopped by operator"));
estopButton.addEventListener("click", async () => {
  centerJoystick();
  try {
    await post("/api/emergency-stop", {});
    await pollStatus();
  } catch (error) { setFeedback(error.message, true); }
});
resetEstopButton.addEventListener("click", async () => {
  try {
    await post("/api/emergency-stop/reset", {});
    await pollStatus();
  } catch (error) { setFeedback(error.message, true); }
});

$$("[data-arm-side]").forEach((button) => button.addEventListener("click", () => {
  armSide = button.dataset.armSide;
  $$("[data-arm-side]").forEach((item) => item.classList.toggle("is-active", item === button));
  showArmSide();
}));

$$("[data-gripper]").forEach((button) => button.addEventListener("click", async () => {
  try {
    const result = await post("/api/gripper", {
      side: $("#gripperSide").value,
      action: button.dataset.gripper,
    });
    setFeedback(result.message);
  } catch (error) { setFeedback(error.message, true); }
}));

function selectCamera(cameraId) {
  if (!cameras.has(cameraId)) return;
  selectedCamera = cameraId;
  $$("[data-camera-id]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.cameraId === selectedCamera);
  });
  const isRgbd = Boolean(cameras.get(selectedCamera)?.rgbd);
  depthToggle.classList.toggle("hidden", !isRgbd);
  if (!isRgbd) setCameraMode("rgb");
  else startCameraLoop();
}

function setCameraMode(mode) {
  cameraMode = mode;
  $$("[data-camera-mode]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.cameraMode === mode);
  });
  startCameraLoop();
}

$$("[data-camera-id]").forEach((button) => {
  button.addEventListener("click", () => selectCamera(button.dataset.cameraId));
});
$$("[data-camera-mode]").forEach((button) => {
  button.addEventListener("click", () => setCameraMode(button.dataset.cameraMode));
});

function stopCameraLoop() {
  cameraLoopGeneration += 1;
  cameraAbortController?.abort();
  cameraAbortController = null;
  pendingCameraFrame = null;
  clearTimeout(cameraFallbackTimer);
  cameraFallbackTimer = null;
  nativeCameraActive = false;
  cameraNative.removeAttribute("src");
  cameraNative.classList.add("hidden");
  cameraCanvas.classList.remove("hidden");
}

function startCameraLoop() {
  stopCameraLoop();
  displayedFrameTimes = [];
  const camera = cameras.get(selectedCamera);
  activeCamera.textContent = `${camera?.label || selectedCamera} / ${cameraMode.toUpperCase()}`;
  if (!connected || document.hidden || !camera) return;
  cameraPlaceholder.classList.remove("hidden");
  cameraPlaceholderText.textContent = "Opening camera stream...";
  cameraStatus.textContent = "Connecting to the live MuJoCo camera...";
  cameraStatus.classList.remove("is-error");
  const generation = cameraLoopGeneration;
  if (isIosWebKit) {
    startNativeCameraFallback(generation, "Using Safari-compatible live video.");
    return;
  }
  cameraFallbackTimer = setTimeout(() => {
    if (generation === cameraLoopGeneration && !displayedFrameTimes.length) {
      startNativeCameraFallback(
        generation, "The browser stream parser produced no frames; using compatible live video.");
    }
  }, 1400);
  void runCameraLoop(generation);
}

function startNativeCameraFallback(generation, reason) {
  if (generation !== cameraLoopGeneration || !connected || document.hidden) return;
  cameraAbortController?.abort();
  clearTimeout(cameraFallbackTimer);
  cameraFallbackTimer = null;
  pendingCameraFrame = null;
  nativeCameraActive = true;
  cameraCanvas.classList.add("hidden");
  cameraNative.classList.remove("hidden");
  cameraPlaceholder.classList.add("hidden");
  displayFps.textContent = "native";
  cameraStatus.textContent = reason;
  cameraStatus.classList.remove("is-error");
  cameraNative.src = `/api/camera/stream?name=${encodeURIComponent(selectedCamera)}`
    + `&mode=${cameraMode}&native=${Date.now()}`;
}

function decodeCameraFrame(blob) {
  if (window.createImageBitmap) return window.createImageBitmap(blob);
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(blob);
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Camera frame could not be decoded."));
    };
    image.src = url;
  });
}

async function runCameraLoop(generation) {
  let streamEnded = false;
  while (generation === cameraLoopGeneration && connected && !document.hidden) {
    const controller = new AbortController();
    cameraAbortController = controller;
    try {
      const response = await fetch(
        `/api/camera/stream?name=${encodeURIComponent(selectedCamera)}&mode=${cameraMode}`,
        { cache: "no-store", signal: controller.signal },
      );
      if (!response.ok || !response.body) {
        throw new Error(`Camera stream unavailable (${response.status})`);
      }
      const reader = response.body.getReader();
      let buffer = new Uint8Array(0);
      while (generation === cameraLoopGeneration) {
        const { done, value } = await reader.read();
        if (done) {
          streamEnded = true;
          break;
        }
        buffer = appendBytes(buffer, value);
        buffer = consumeCameraParts(buffer, generation);
      }
    } catch (error) {
      if (error.name === "AbortError" || generation !== cameraLoopGeneration) break;
      startNativeCameraFallback(
        generation, `Camera transport error: ${error.message}. Trying compatible live video.`);
      break;
    } finally {
      if (cameraAbortController === controller) cameraAbortController = null;
    }
    if (generation === cameraLoopGeneration && streamEnded) {
      cameraPlaceholder.classList.remove("hidden");
      cameraPlaceholderText.textContent = "Camera stream changed";
      cameraStatus.textContent =
        "Another viewer selected a different camera. Select this camera again to take control of the live view.";
      cameraStatus.classList.add("is-error");
      break;
    }
  }
}

const multipartBreak = new Uint8Array([13, 10, 13, 10]);
const textDecoder = new TextDecoder("ascii");

function appendBytes(first, second) {
  const combined = new Uint8Array(first.length + second.length);
  combined.set(first);
  combined.set(second, first.length);
  return combined;
}

function findBytes(buffer, pattern) {
  outer: for (let index = 0; index <= buffer.length - pattern.length; index += 1) {
    for (let offset = 0; offset < pattern.length; offset += 1) {
      if (buffer[index + offset] !== pattern[offset]) continue outer;
    }
    return index;
  }
  return -1;
}

function consumeCameraParts(buffer, generation) {
  while (buffer.length) {
    const headerEnd = findBytes(buffer, multipartBreak);
    if (headerEnd < 0) return buffer;
    const headerText = textDecoder.decode(buffer.slice(0, headerEnd));
    const headers = new Map();
    for (const line of headerText.split("\r\n").slice(1)) {
      const separator = line.indexOf(":");
      if (separator > 0) {
        headers.set(line.slice(0, separator).toLowerCase(), line.slice(separator + 1).trim());
      }
    }
    const length = Number(headers.get("content-length"));
    if (!Number.isFinite(length) || length < 1) throw new Error("Invalid camera stream frame.");
    const imageStart = headerEnd + multipartBreak.length;
    const partEnd = imageStart + length + 2;
    if (buffer.length < partEnd) return buffer;
    offerCameraFrame({
      generation,
      data: buffer.slice(imageStart, imageStart + length),
      headers,
      receivedAt: performance.now(),
    });
    buffer = buffer.slice(partEnd);
  }
  return buffer;
}

function offerCameraFrame(frame) {
  if (pendingCameraFrame) clientDroppedFrames += 1;
  pendingCameraFrame = frame;
  if (!cameraDecodeActive) void displayNewestCameraFrame();
}

async function displayNewestCameraFrame() {
  cameraDecodeActive = true;
  let decodingGeneration = cameraLoopGeneration;
  try {
    while (pendingCameraFrame) {
      const frame = pendingCameraFrame;
      pendingCameraFrame = null;
      decodingGeneration = frame.generation;
      const decoded = await decodeCameraFrame(new Blob(
        [frame.data], { type: frame.headers.get("content-type") || "image/jpeg" }));
      if (frame.generation !== cameraLoopGeneration || pendingCameraFrame) {
        if (pendingCameraFrame) clientDroppedFrames += 1;
        decoded.close?.();
        continue;
      }

      cameraContext.drawImage(decoded, 0, 0, cameraCanvas.width, cameraCanvas.height);
      decoded.close?.();
      const displayedAt = performance.now();
      clearTimeout(cameraFallbackTimer);
      cameraFallbackTimer = null;
      cameraPlaceholder.classList.add("hidden");
      cameraStatus.textContent = "Live low-latency MuJoCo stream.";
      cameraStatus.classList.remove("is-error");
      displayedFrameTimes.push(displayedAt);
      displayedFrameTimes = displayedFrameTimes.filter((stamp) => displayedAt - stamp <= 1000);
      const displayedRate = displayedFrameTimes.length > 1
        ? (displayedFrameTimes.length - 1) * 1000
          / (displayedFrameTimes[displayedFrameTimes.length - 1] - displayedFrameTimes[0])
        : 0;
      const approximateAge = (Number(frame.headers.get("x-frame-age-ms")) || 0)
        + (displayedAt - frame.receivedAt);
      const serverDropped = Number(frame.headers.get("x-dropped-frames")) || 0;
      displayFps.textContent = displayedRate.toFixed(1);
      renderFps.textContent = (Number(frame.headers.get("x-render-fps")) || 0).toFixed(1);
      frameAge.textContent = `${Math.round(approximateAge)} ms`;
      droppedFrames.textContent = String(serverDropped + clientDroppedFrames);
      activeCamera.textContent = `${cameras.get(selectedCamera)?.label || selectedCamera} / ${cameraMode.toUpperCase()}`;
    }
  } catch (error) {
    startNativeCameraFallback(
      decodingGeneration, `Camera decode error: ${error.message}. Trying compatible live video.`);
  } finally {
    cameraDecodeActive = false;
    if (pendingCameraFrame) void displayNewestCameraFrame();
  }
}

cameraNative.addEventListener("load", () => {
  if (!nativeCameraActive) return;
  cameraPlaceholder.classList.add("hidden");
  cameraStatus.textContent = "Live Safari-compatible MuJoCo stream.";
  cameraStatus.classList.remove("is-error");
});
cameraNative.addEventListener("error", () => {
  if (!nativeCameraActive) return;
  cameraPlaceholder.classList.remove("hidden");
  cameraPlaceholderText.textContent = "Camera stream unavailable";
  cameraStatus.textContent = connected
    ? "The simulation is connected, but this browser could not display its camera stream."
    : "The Butler[bot] simulation is unavailable.";
  cameraStatus.classList.add("is-error");
});

async function submitTask(text) {
  const command = String(text || "").trim();
  if (!command) return;
  heardText.textContent = command;
  taskReply.textContent = "Understanding...";
  taskMessage.textContent = "Sending to the local reasoning pipeline...";
  setFeedback("");
  try {
    const result = await post("/api/task", { text: command });
    taskInput.value = "";
    updateTask(result.task);
    await pollStatus();
  } catch (error) {
    setFeedback(error.message, true);
    taskMessage.textContent = error.message;
  }
}

taskForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitTask(taskInput.value);
});
cancelTaskButton.addEventListener("click", async () => {
  try {
    const result = await post("/api/task/cancel", {});
    setFeedback(result.message);
    await pollStatus();
  } catch (error) { setFeedback(error.message, true); }
});

const voiceLabel = voiceButton.querySelector(".talk-button__label");
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

function setVoiceUi(label, active = false) {
  voiceButton.classList.toggle("is-listening", active);
  voiceButton.setAttribute("aria-pressed", String(active));
  voiceLabel.textContent = label;
}

function showVoiceError(message) {
  setVoiceUi("HOLD TO TALK");
  setFeedback(message, true);
  taskMessage.textContent = message;
}

function captureVoicePointer(pointerId) {
  if (pointerId == null) return;
  try { voiceButton.setPointerCapture(pointerId); } catch (_) { /* optional on mobile */ }
}

function releaseVoicePointer(pointerId) {
  if (pointerId == null) return;
  try {
    if (voiceButton.hasPointerCapture(pointerId)) voiceButton.releasePointerCapture(pointerId);
  } catch (_) { /* optional on mobile */ }
}

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  const speechErrorMessage = (code) => {
    const messages = {
      "not-allowed": "Microphone permission was denied. Allow microphone access for this site and try again.",
      "service-not-allowed": "Speech recognition is blocked by this browser. Check site permissions or use the typed command box.",
      "audio-capture": "No working microphone was found by the browser.",
      "no-speech": "No speech was detected. Hold TALK while speaking, then release.",
      "network": `The browser speech service could not be reached.${window.isSecureContext ? "" : " This phone may require HTTPS for microphone features."}`,
      "aborted": "Voice recognition was cancelled before speech was captured.",
    };
    return messages[code] || `Voice recognition failed (${code}). Use the typed command box if it continues.`;
  };

  const beginListening = (event) => {
    if (emergencyStopped || speechActive || speechStarting) return;
    event?.preventDefault();
    speechPointer = event?.pointerId ?? null;
    speechStarting = true;
    stopSpeechWhenStarted = false;
    pendingTranscript = "";
    speechFailed = false;
    setFeedback("");
    setVoiceUi("LISTENING...", true);
    taskMessage.textContent = "Listening...";
    captureVoicePointer(speechPointer);
    try {
      recognition.start();
    } catch (error) {
      speechStarting = false;
      speechPointer = null;
      speechFailed = true;
      showVoiceError(error.name === "InvalidStateError"
        ? "Voice recognition is already busy. Release TALK and try again."
        : `Voice recognition could not start: ${error.message || error.name}`);
    }
  };
  const endListening = (event) => {
    if (event?.pointerId !== undefined && speechPointer !== null
        && event.pointerId !== speechPointer) return;
    event?.preventDefault();
    if (speechStarting && !speechActive) stopSpeechWhenStarted = true;
    else if (speechActive) {
      try { recognition.stop(); } catch (_) { /* browser is already stopping */ }
    }
    releaseVoicePointer(speechPointer);
  };
  voiceButton.addEventListener("pointerdown", beginListening);
  voiceButton.addEventListener("pointerup", endListening);
  voiceButton.addEventListener("pointercancel", endListening);
  voiceButton.addEventListener("lostpointercapture", endListening);
  voiceButton.addEventListener("keydown", (event) => {
    if ((event.key === " " || event.key === "Enter") && !event.repeat) beginListening(event);
  });
  voiceButton.addEventListener("keyup", (event) => {
    if (event.key === " " || event.key === "Enter") endListening(event);
  });
  recognition.addEventListener("start", () => {
    speechStarting = false;
    speechActive = true;
    setVoiceUi("LISTENING...", true);
    if (stopSpeechWhenStarted) {
      try { recognition.stop(); } catch (_) { /* browser is already stopping */ }
    }
  });
  recognition.addEventListener("end", () => {
    const transcript = pendingTranscript.trim();
    speechActive = false;
    speechStarting = false;
    speechPointer = null;
    stopSpeechWhenStarted = false;
    pendingTranscript = "";
    setVoiceUi("HOLD TO TALK");
    if (transcript && !speechFailed) submitTask(transcript);
  });
  recognition.addEventListener("result", (event) => {
    let spoken = "";
    let finalText = "";
    for (let index = 0; index < event.results.length; index += 1) {
      const text = event.results[index][0].transcript;
      spoken += `${text} `;
      if (event.results[index].isFinal) finalText += `${text} `;
    }
    spoken = spoken.trim();
    heardText.textContent = spoken;
    taskInput.value = spoken;
    taskMessage.textContent = "Speech captured. Release TALK to send.";
    if (finalText.trim()) pendingTranscript = finalText.trim();
  });
  recognition.addEventListener("error", (event) => {
    speechFailed = true;
    showVoiceError(speechErrorMessage(event.error));
  });
} else if (window.MediaRecorder && navigator.mediaDevices?.getUserMedia) {
  let recorder = null;
  let recordingStream = null;
  let recordingChunks = [];
  let recordingRequested = false;
  let recordingPointer = null;
  let recordingTimer = null;

  const recordingMimeType = () => [
    "audio/mp4", "audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus",
  ].find((type) => window.MediaRecorder.isTypeSupported?.(type)) || "";

  const finishLocalRecording = async () => {
    clearTimeout(recordingTimer);
    recordingTimer = null;
    recordingStream?.getTracks().forEach((track) => track.stop());
    recordingStream = null;
    speechActive = false;
    speechStarting = true;
    setVoiceUi("TRANSCRIBING...", true);
    taskMessage.textContent = "Transcribing locally on the Butler[bot] computer...";
    const mimeType = recorder?.mimeType || recordingMimeType() || "application/octet-stream";
    const audio = new Blob(recordingChunks, { type: mimeType });
    recorder = null;
    recordingChunks = [];
    try {
      if (audio.size < 128) throw new Error("The recording was empty or too short.");
      const response = await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": mimeType },
        body: audio,
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `Transcription failed (${response.status})`);
      const transcript = String(result.transcript || "").trim();
      if (!transcript) throw new Error("Local speech recognition returned no text.");
      heardText.textContent = transcript;
      taskInput.value = transcript;
      speechStarting = false;
      setVoiceUi("HOLD TO TALK");
      await submitTask(transcript);
    } catch (error) {
      speechStarting = false;
      showVoiceError(error.message);
    }
  };

  const beginRecording = async (event) => {
    if (emergencyStopped || speechActive || speechStarting) return;
    event?.preventDefault();
    if (!window.isSecureContext) {
      showVoiceError(
        "iPhone microphone capture requires HTTPS. Reopen this remote through a trusted HTTPS address.");
      return;
    }
    recordingPointer = event?.pointerId ?? null;
    recordingRequested = true;
    speechStarting = true;
    setFeedback("");
    setVoiceUi("ALLOW MICROPHONE", true);
    taskMessage.textContent = "Requesting microphone access...";
    captureVoicePointer(recordingPointer);
    try {
      recordingStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
      if (!recordingRequested) {
        recordingStream.getTracks().forEach((track) => track.stop());
        recordingStream = null;
        speechStarting = false;
        setVoiceUi("HOLD TO TALK");
        taskMessage.textContent = "Microphone ready. Hold TALK again while speaking.";
        return;
      }
      const mimeType = recordingMimeType();
      recorder = new window.MediaRecorder(
        recordingStream, mimeType ? { mimeType } : undefined);
      recordingChunks = [];
      recorder.addEventListener("dataavailable", (chunk) => {
        if (chunk.data?.size) recordingChunks.push(chunk.data);
      });
      recorder.addEventListener("stop", finishLocalRecording, { once: true });
      recorder.start();
      speechStarting = false;
      speechActive = true;
      setVoiceUi("RECORDING...", true);
      taskMessage.textContent = "Recording locally. Release TALK to transcribe.";
      recordingTimer = setTimeout(() => {
        if (recorder?.state === "recording") recorder.stop();
      }, 15000);
    } catch (error) {
      recordingRequested = false;
      speechStarting = false;
      speechActive = false;
      recordingStream?.getTracks().forEach((track) => track.stop());
      recordingStream = null;
      const messages = {
        NotAllowedError: "Microphone permission was denied. Allow microphone access for this HTTPS site.",
        NotFoundError: "No microphone was found on this device.",
        NotReadableError: "The microphone is busy or unavailable.",
        SecurityError: "The browser blocked microphone access. Use a trusted HTTPS address.",
      };
      showVoiceError(messages[error.name]
        || `Microphone recording failed: ${error.message || error.name}`);
    }
  };

  const endRecording = (event) => {
    if (event?.pointerId !== undefined && recordingPointer !== null
        && event.pointerId !== recordingPointer) return;
    event?.preventDefault();
    recordingRequested = false;
    releaseVoicePointer(recordingPointer);
    recordingPointer = null;
    if (recorder?.state === "recording") recorder.stop();
  };
  voiceButton.addEventListener("pointerdown", beginRecording);
  voiceButton.addEventListener("pointerup", endRecording);
  voiceButton.addEventListener("pointercancel", endRecording);
  voiceButton.addEventListener("lostpointercapture", endRecording);
  voiceButton.addEventListener("keydown", (event) => {
    if ((event.key === " " || event.key === "Enter") && !event.repeat) beginRecording(event);
  });
  voiceButton.addEventListener("keyup", (event) => {
    if (event.key === " " || event.key === "Enter") endRecording(event);
  });
} else {
  voiceButton.disabled = true;
  const unsupported = "This browser cannot record audio. Use Safari 14.1+, Chrome, Edge, or the typed command box.";
  voiceButton.title = unsupported;
  setVoiceUi("VOICE UNAVAILABLE");
  setFeedback(unsupported, true);
}
voiceButton.addEventListener("contextmenu", (event) => event.preventDefault());

window.addEventListener("pagehide", () => {
  stopCameraLoop();
  beaconStop();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    stopCameraLoop();
    beaconStop();
  } else {
    startCameraLoop();
  }
});
setInterval(pollStatus, 700);
pollStatus();
