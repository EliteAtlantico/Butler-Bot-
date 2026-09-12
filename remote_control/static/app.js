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
const cameraSelect = $("#cameraSelect");
const cameraImage = $("#cameraImage");
const cameraPlaceholder = $("#cameraPlaceholder");
const armControls = $("#armControls");
const commandForm = $("#commandForm");
const commandInput = $("#commandInput");
const commandFeedback = $("#commandFeedback");
const voiceButton = $("#voiceButton");

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
let cameraRequestActive = false;
let cameraObjectUrl = null;
let lastStatusAt = 0;

async function api(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    ...options,
    headers: options.body ? { "Content-Type": "application/json", ...(options.headers || {}) } : options.headers,
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
  connected = isConnected;
  connectionBadge.className = `connection ${isConnected ? "connection--online" : "connection--offline"}`;
  connectionText.textContent = isConnected ? "CONNECTED" : "DISCONNECTED";
}

function updateSafety(status) {
  emergencyStopped = Boolean(status.emergency_stop);
  document.body.classList.toggle("is-estopped", emergencyStopped);
  estopButton.classList.toggle("hidden", emergencyStopped);
  resetEstopButton.classList.toggle("hidden", !emergencyStopped);
  currentCommand.textContent = status.fault || status.reason || "Ready";
  $("#backendLabel").textContent = `Robot backend: ${status.capabilities?.backend || "unknown"}`;
  $("#watchdogLabel").textContent = `Safety watchdog: ${status.watchdog_ms || "—"} ms`;
}

async function pollStatus() {
  try {
    const status = await api("/api/status");
    lastStatusAt = Date.now();
    setConnection(Boolean(status.connected));
    updateSafety(status);
    if (!capabilitiesLoaded) configureCapabilities(status.capabilities || {});
    updateArmReadouts(status.telemetry?.arm_positions || {});
  } catch (error) {
    if (Date.now() - lastStatusAt > 1800) {
      setConnection(false);
      currentCommand.textContent = "Connection lost — stopped";
      centerJoystick();
    }
  }
}

function configureCapabilities(capabilities) {
  capabilitiesLoaded = true;
  cameraSelect.replaceChildren();
  (capabilities.cameras || []).forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name.replaceAll("_", " ");
    cameraSelect.append(option);
  });
  if ([...cameraSelect.options].some((option) => option.value === "head_rgb")) {
    cameraSelect.value = "head_rgb";
  }
  buildArmControls(capabilities.arm_joints || []);
}

function buildArmControls(joints) {
  armControls.replaceChildren();
  joints.forEach((joint) => {
    const row = document.createElement("div");
    row.className = "arm-control";
    row.dataset.side = joint.side;
    row.dataset.joint = joint.name;

    const id = `joint-${joint.name}`;
    const label = document.createElement("label");
    label.htmlFor = id;
    label.textContent = `${joint.label} (${joint.name})`;
    const output = document.createElement("output");
    output.htmlFor = id;
    output.textContent = `0.00 ${joint.unit}`;
    const slider = document.createElement("input");
    slider.id = id;
    slider.type = "range";
    slider.min = joint.minimum;
    slider.max = joint.maximum;
    slider.step = joint.step;
    slider.value = 0;
    slider.dataset.unit = joint.unit;
    slider.addEventListener("change", () => sendArmTarget(
      joint.name, Number(slider.value), output, joint.unit,
    ));
    slider.addEventListener("input", () => { output.textContent = `${Number(slider.value).toFixed(2)} ${joint.unit}`; });
    row.append(label, output, slider);
    armControls.append(row);
  });
  showArmSide();
}

function updateArmReadouts(positions) {
  $$(".arm-control").forEach((row) => {
    const slider = row.querySelector("input");
    if (document.activeElement !== slider && positions[row.dataset.joint] !== undefined) {
      slider.value = positions[row.dataset.joint];
      row.querySelector("output").textContent = `${Number(slider.value).toFixed(2)} ${slider.dataset.unit}`;
    }
  });
}

function showArmSide() {
  $$(".arm-control").forEach((row) => row.classList.toggle("hidden", row.dataset.side !== armSide));
}

async function sendArmTarget(joint, value, output, unit) {
  try {
    const result = await post("/api/arm", { joint, value });
    output.textContent = `${Number(result.target).toFixed(2)} ${unit}`;
    setFeedback(`${joint} target updated.`);
  } catch (error) { setFeedback(error.message, true); }
}

function joystickPosition(event) {
  const rect = joystick.getBoundingClientRect();
  const radius = rect.width / 2;
  let x = (event.clientX - (rect.left + radius)) / radius;
  let y = (event.clientY - (rect.top + radius)) / radius;
  const magnitude = Math.hypot(x, y);
  if (magnitude > 1) { x /= magnitude; y /= magnitude; }
  if (magnitude <= DEAD_ZONE) return { x: 0, y: 0 };
  const scaled = (Math.min(magnitude, 1) - DEAD_ZONE) / (1 - DEAD_ZONE);
  const direction = magnitude || 1;
  return { x: (x / direction) * scaled, y: (y / direction) * scaled };
}

function updateJoystick(event) {
  const { x, y } = joystickPosition(event);
  driveLinear = -y;
  driveAngular = -x;
  joystickKnob.style.transform = `translate(calc(-50% + ${x * 82}px), calc(-50% + ${y * 82}px))`;
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
  } catch (error) {
    setConnection(false);
    centerJoystick();
  } finally {
    driveRequestActive = false;
  }
}

async function stopNow(reason = "Stopped") {
  centerJoystick();
  currentCommand.textContent = reason;
  try { await post("/api/stop", {}); } catch (_) { setConnection(false); }
}

function beaconStop() {
  centerJoystick();
  const body = new Blob(["{}"], { type: "application/json" });
  navigator.sendBeacon("/api/stop", body);
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
    emergencyStopped = true;
    document.body.classList.add("is-estopped");
    currentCommand.textContent = "EMERGENCY STOP";
    estopButton.classList.add("hidden");
    resetEstopButton.classList.remove("hidden");
  } catch (error) { setFeedback(error.message, true); }
});
resetEstopButton.addEventListener("click", async () => {
  try {
    await post("/api/emergency-stop/reset", {});
    await pollStatus();
  } catch (error) { setFeedback(error.message, true); }
});

$$('[data-arm-side]').forEach((button) => button.addEventListener("click", () => {
  armSide = button.dataset.armSide;
  $$('[data-arm-side]').forEach((item) => item.classList.toggle("is-active", item === button));
  showArmSide();
}));

$$('[data-gripper]').forEach((button) => button.addEventListener("click", async () => {
  const side = $("#gripperSide").value;
  const action = button.dataset.gripper;
  try {
    const result = await post("/api/gripper", { side, action });
    setFeedback(result.message);
  } catch (error) { setFeedback(error.message, true); }
}));

function setFeedback(message, isError = false) {
  commandFeedback.textContent = message;
  commandFeedback.classList.toggle("is-error", isError);
}

async function submitTextCommand(text) {
  const command = text.trim();
  if (!command) return;
  try {
    const result = await post("/api/command", { text: command });
    setFeedback(result.message);
    currentCommand.textContent = result.message;
    commandInput.value = "";
    await pollStatus();
  } catch (error) { setFeedback(error.message, true); }
}

commandForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitTextCommand(commandInput.value);
});

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  voiceButton.addEventListener("click", () => recognition.start());
  recognition.addEventListener("start", () => {
    voiceButton.classList.add("is-listening");
    voiceButton.textContent = "LIVE";
    setFeedback("Listening…");
  });
  recognition.addEventListener("end", () => {
    voiceButton.classList.remove("is-listening");
    voiceButton.textContent = "MIC";
  });
  recognition.addEventListener("result", (event) => {
    const spoken = event.results[0][0].transcript;
    commandInput.value = spoken;
    submitTextCommand(spoken);
  });
  recognition.addEventListener("error", (event) => setFeedback(`Voice input unavailable: ${event.error}`, true));
} else {
  voiceButton.disabled = true;
  voiceButton.title = "Voice commands are not supported by this browser.";
}

async function refreshCamera() {
  if (!connected || cameraRequestActive || document.hidden || !cameraSelect.value) return;
  cameraRequestActive = true;
  try {
    const response = await fetch(`/api/camera?name=${encodeURIComponent(cameraSelect.value)}&t=${Date.now()}`,
                                 { cache: "no-store" });
    if (!response.ok) throw new Error("Camera unavailable");
    const blob = await response.blob();
    const nextUrl = URL.createObjectURL(blob);
    cameraImage.src = nextUrl;
    cameraPlaceholder.classList.add("hidden");
    if (cameraObjectUrl) URL.revokeObjectURL(cameraObjectUrl);
    cameraObjectUrl = nextUrl;
  } catch (_) {
    cameraPlaceholder.classList.remove("hidden");
  } finally {
    cameraRequestActive = false;
  }
}

window.addEventListener("pagehide", beaconStop);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) beaconStop();
});

setInterval(pollStatus, 900);
setInterval(refreshCamera, 500);
pollStatus();

