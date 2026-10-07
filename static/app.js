// Gemma-Lens frontend. Vanilla JS, no build step.

const $ = (sel) => document.querySelector(sel);
const queryInput = $("#query-input");
const micBtn = $("#mic-btn");
const uploadBtn = $("#upload-btn");
const fileInput = $("#file-input");
const examplesEl = $("#examples");
const statusEl = $("#query-status");
const resultsSection = $("#results-section");
const resultsGrid = $("#results-grid");
const timingEl = $("#timing");
const modal = $("#modal");
const modalTitle = $("#modal-title");
const modalBody = $("#modal-body");
const modalClose = $("#modal-close");

const EMOJI = {
  "red cars": "🚗",
  "sunset over water": "🌅",
  "snowy mountains": "🏔️",
  "kitchen with food": "🍳",
  "white objects": "⚪",
  "dogs playing": "🐕",
  "cow sound": "🐄",
  "clapping": "👏",
  "cat meow": "😺",
  "rain": "🌧️",
};

// ---------- init ----------

document.addEventListener("DOMContentLoaded", async () => {
  await loadExamples();
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && queryInput.value.trim()) {
      submitText(queryInput.value.trim());
    }
  });
  uploadBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", onFileSelected);
  micBtn.addEventListener("click", onMicClick);
  modalClose.addEventListener("click", closeModal);
  modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
});

async function loadExamples() {
  try {
    const r = await fetch("/examples");
    const examples = await r.json();
    examplesEl.innerHTML = `<span class="text-sm text-neutral-500 mr-2 self-center">Try:</span>`;
    for (const ex of examples) {
      const btn = document.createElement("button");
      const emoji = EMOJI[ex.label] || "";
      btn.className = "px-3 py-1.5 rounded-full border border-neutral-300 bg-white " +
                      "hover:bg-blue-50 hover:border-blue-400 text-sm transition-colors";
      btn.textContent = `${emoji} ${ex.label}`.trim();
      btn.addEventListener("click", () => {
        if (ex.modality === "text") {
          queryInput.value = ex.text;
          submitText(ex.text);
        } else if (ex.modality === "audio" && ex.preset) {
          submitAudioPreset(ex.preset);
        }
      });
      examplesEl.appendChild(btn);
    }
  } catch (e) {
    console.error("Failed to load examples:", e);
  }
}

// ---------- query submission ----------

async function submitText(text) {
  setStatus(`Searching: ${text}`);
  try {
    const data = await postSearch({ modality: "text", text });
    renderResults(data, `text · "${text}"`);
  } catch (e) {
    setStatus(`Error: ${e.message}`, true);
  }
}

async function submitImageFile(file) {
  queryInput.value = "";
  setStatus(`Searching with image: ${file.name}`);
  try {
    const b64 = await fileToBase64(file);
    const data = await postSearch({ modality: "image", image_b64: b64 });
    renderResults(data, `image · ${file.name}`);
  } catch (e) {
    setStatus(`Error: ${e.message}`, true);
  }
}

async function submitAudio(blob, label) {
  queryInput.value = "";
  setStatus(`Searching with audio: ${label}`);
  try {
    const b64 = await blobToBase64(blob);
    const data = await postSearch({ modality: "audio", audio_b64: b64 });
    renderResults(data, `audio · ${label}`);
  } catch (e) {
    setStatus(`Error: ${e.message}`, true);
  }
}

async function submitAudioPreset(filename) {
  setStatus(`Loading preset: ${filename}`);
  try {
    const r = await fetch(`/static/presets/${filename}`);
    if (!r.ok) throw new Error(`preset not found: ${filename}`);
    const blob = await r.blob();
    await submitAudio(blob, filename);
  } catch (e) {
    setStatus(`Error: ${e.message}`, true);
  }
}

async function postSearch(body) {
  const r = await fetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`HTTP ${r.status}: ${text}`);
  }
  return r.json();
}

// ---------- rendering ----------

function renderResults(data, queryLabel) {
  resultsSection.classList.remove("hidden");
  timingEl.textContent = `${queryLabel} · ${data.query_ms} ms total (${data.embed_ms} ms embed)`;
  statusEl.textContent = "";
  resultsGrid.innerHTML = "";
  if (!data.results.length) {
    resultsGrid.innerHTML = `<div class="col-span-full text-neutral-500">No matches.</div>`;
    return;
  }
  for (const hit of data.results) {
    resultsGrid.appendChild(renderTile(hit));
  }
}

function renderTile(hit) {
  const card = document.createElement("div");
  card.className = "relative rounded-xl overflow-hidden border border-neutral-200 bg-white " +
                   "hover:border-blue-400 hover:shadow-md transition cursor-pointer group";
  if (hit.modality === "image") {
    const img = document.createElement("img");
    img.src = hit.url;
    img.loading = "lazy";
    img.className = "w-full aspect-square object-cover bg-neutral-100";
    img.alt = hit.filename;
    card.appendChild(img);
  } else {
    const box = document.createElement("div");
    box.className = "w-full aspect-square bg-neutral-100 flex items-center justify-center";
    box.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48"
      viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"
      class="text-neutral-400"><path d="M11 5 6 9H2v6h4l5 4V5z"/>
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>`;
    card.appendChild(box);
    const audio = document.createElement("audio");
    audio.src = hit.url;
    audio.controls = true;
    audio.preload = "none";
    audio.className = "w-full";
    audio.addEventListener("click", (e) => e.stopPropagation());
    card.appendChild(audio);
  }

  const footer = document.createElement("div");
  footer.className = "p-2 text-xs flex items-center justify-between";
  const label = document.createElement("span");
  label.className = "truncate text-neutral-600";
  label.textContent = tileLabel(hit);
  const score = document.createElement("span");
  score.className = "ml-2 font-mono tabular-nums text-blue-600 shrink-0";
  score.textContent = hit.score.toFixed(2);
  footer.appendChild(label);
  footer.appendChild(score);
  card.appendChild(footer);

  card.addEventListener("click", () => openModal(hit));
  return card;
}

function tileLabel(hit) {
  if (hit.modality === "image") return hit.metadata.caption || hit.filename;
  return hit.metadata.category || hit.filename;
}

function openModal(hit) {
  modalTitle.textContent = tileLabel(hit);
  modalBody.innerHTML = "";
  if (hit.modality === "image") {
    const img = document.createElement("img");
    img.src = hit.url;
    img.className = "w-full rounded-lg";
    modalBody.appendChild(img);
  } else {
    const audio = document.createElement("audio");
    audio.src = hit.url;
    audio.controls = true;
    audio.autoplay = true;
    audio.className = "w-full";
    modalBody.appendChild(audio);
  }
  const meta = document.createElement("div");
  meta.className = "mt-4 text-sm text-neutral-600 space-y-1";
  meta.innerHTML = `
    <div><span class="text-neutral-400">score:</span> <span class="font-mono">${hit.score.toFixed(4)}</span></div>
    <div><span class="text-neutral-400">modality:</span> ${hit.modality}</div>
    <div><span class="text-neutral-400">file:</span> ${hit.filename}</div>
    ${Object.entries(hit.metadata).map(([k, v]) =>
       `<div><span class="text-neutral-400">${k}:</span> ${v ?? ""}</div>`
     ).join("")}
  `;
  modalBody.appendChild(meta);
  modal.classList.remove("hidden");
}

function closeModal() { modal.classList.add("hidden"); }

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.className = `mt-3 text-sm min-h-[1.25rem] ${isError ? "text-red-600" : "text-neutral-500"}`;
}

// ---------- audio recording ----------

let mediaRecorder = null;
let audioChunks = [];
let recording = false;

async function onMicClick() {
  if (recording) {
    mediaRecorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.ondataavailable = (e) => { if (e.data.size) audioChunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      recording = false;
      micBtn.classList.remove("bg-red-100", "border-red-500", "text-red-600");
      const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      await submitAudio(blob, "recorded");
    };
    mediaRecorder.start();
    recording = true;
    micBtn.classList.add("bg-red-100", "border-red-500", "text-red-600");
    setStatus("Recording — click mic again to stop");
  } catch (e) {
    setStatus(`Mic error: ${e.message}`, true);
  }
}

// ---------- file + base64 helpers ----------

function onFileSelected(e) {
  const file = e.target.files[0];
  if (file) submitImageFile(file);
  fileInput.value = "";
}

function fileToBase64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}

function blobToBase64(blob) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(blob);
  });
}
