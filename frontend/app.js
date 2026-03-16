const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const bubbleModeSelect = document.getElementById("bubble-mode");
const submitButton = document.getElementById("submit-button");
const statusText = document.getElementById("status-text");
const sourcePreview = document.getElementById("source-preview");
const previewPlaceholder = document.getElementById("preview-placeholder");
const panelsMeta = document.getElementById("panels-meta");
const panelsGrid = document.getElementById("panels-grid");
const captionsList = document.getElementById("captions-list");
const openOverridePageButton = document.getElementById("open-override-page-button");

let latestPanelResult = null;
let latestDialogue = [];
let latestCaptions = [];
let latestBubbleMode = "heuristic";

function setStatus(message) {
  statusText.textContent = message;
}

function resetResults() {
  latestPanelResult = null;
  latestDialogue = [];
  latestCaptions = [];
  latestBubbleMode = bubbleModeSelect?.value || "heuristic";
  panelsMeta.hidden = true;
  panelsMeta.innerHTML = "";
  panelsGrid.innerHTML = '<p class="empty-state">Detected panel crops will appear here.</p>';
  captionsList.innerHTML = '<p class="empty-state">Generated panel captions will appear here.</p>';
  if (openOverridePageButton) {
    openOverridePageButton.disabled = true;
  }
  sessionStorage.removeItem("phase1_override_payload");
}

function renderSourcePreview(file) {
  const objectUrl = URL.createObjectURL(file);
  sourcePreview.src = objectUrl;
  sourcePreview.hidden = false;
  previewPlaceholder.hidden = true;
}

function renderPanels(panelResult, dialogue = [], captions = []) {
  latestPanelResult = panelResult;
  latestDialogue = dialogue;
  latestCaptions = captions;
  panelsMeta.hidden = false;
  panelsMeta.innerHTML = `
    <span class="meta-chip">${panelResult.panels} panels detected</span>
    <span class="meta-chip">${panelResult.filename}</span>
  `;

  panelsGrid.innerHTML = panelResult.panel_boxes
    .map((panel, index) => {
      const panelText = dialogue[index]?.text || "[no dialogue detected]";
      const bubbleCandidates = captions[index]?.bubble_candidates || "0";
      const heuristicBubbleCount = captions[index]?.bubble_count || "0";
      const bubbleSequence = captions[index]?.bubble_sequence || "none";
      const speechCount = captions[index]?.speech_count || "0";
      const narrationCount = captions[index]?.narration_count || "0";
      const sfxCount = captions[index]?.sfx_count || "0";
      const textRegionCount = dialogue[index]?.text_regions || "0";
      const textRole = dialogue[index]?.text_role || "ambient";
      return `
        <article class="panel-card">
          <div class="panel-image-frame">
            <img src="${panel.image_url}" alt="Panel ${panel.index}" />
          </div>
          <div class="panel-content">
            <div class="panel-head">
              <p class="panel-title">Panel ${panel.index}</p>
              <span class="panel-bbox">[${panel.bbox.join(", ")}]</span>
            </div>
            <div class="panel-ocr">
              <span class="panel-ocr-label">OCR Text</span>
              <p>${escapeHtml(panelText)}</p>
            </div>
            <div class="panel-ocr">
              <span class="panel-ocr-label">Bubble Detection</span>
              <p>Bubble candidates: ${escapeHtml(bubbleCandidates)}</p>
              <p>Speech bubbles: ${escapeHtml(heuristicBubbleCount)}</p>
              <p>Bubble sequence: ${escapeHtml(bubbleSequence)}</p>
              <p>Speech regions: ${escapeHtml(speechCount)}</p>
              <p>Narration regions: ${escapeHtml(narrationCount)}</p>
              <p>SFX regions: ${escapeHtml(sfxCount)}</p>
              <p>OCR text regions: ${escapeHtml(textRegionCount)}</p>
              <p>Text role: ${escapeHtml(textRole)}</p>
            </div>
          </div>
        </article>
      `;
    })
    .join("");

  persistOverridePayload();
}

function renderCaptions(captions) {
  captionsList.innerHTML = captions.length
    ? captions
        .map(
          (item, index) => `
            <article class="stack-item">
              <strong>Panel ${index + 1}</strong>
              <p>${escapeHtml(item.caption || "[no caption generated]")}</p>
            </article>
          `,
        )
        .join("")
    : '<p class="empty-state">No panel captions were returned.</p>';
}

function persistOverridePayload() {
  if (!latestPanelResult) {
    return;
  }
  const payload = {
    panelResult: latestPanelResult,
    dialogue: latestDialogue,
    captions: latestCaptions,
    bubbleMode: latestBubbleMode,
    savedAt: new Date().toISOString(),
  };
  sessionStorage.setItem("phase1_override_payload", JSON.stringify(payload));
  if (openOverridePageButton) {
    openOverridePageButton.disabled = false;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function uploadTo(endpoint, file, extraFields = {}) {
  const formData = new FormData();
  formData.append("file", file, file.name);
  for (const [key, value] of Object.entries(extraFields)) {
    formData.append(key, value);
  }

  const response = await fetch(endpoint, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = `Request failed for ${endpoint}`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (error) {
      message = `${message} (${response.status})`;
    }
    throw new Error(message);
  }

  return response.json();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const [file] = fileInput.files;
  if (!file) {
    setStatus("Choose an image before running the pipeline.");
    return;
  }

  submitButton.disabled = true;
  resetResults();
  renderSourcePreview(file);

  try {
    latestBubbleMode = bubbleModeSelect?.value || "heuristic";
    setStatus("Detecting panels...");
    const panelResult = await uploadTo("/detect-panels", file);
    renderPanels(panelResult);

    setStatus("Running panel analysis...");
    const analysisResult = await uploadTo("/analyze-panels", file, { bubble_mode: latestBubbleMode });
    latestPanelResult = { ...panelResult, request_id: analysisResult.request_id };
    renderPanels(latestPanelResult, analysisResult.dialogue || [], analysisResult.captions || []);
    renderCaptions(analysisResult.captions || []);

    setStatus(`Phase 1 completed using ${latestBubbleMode}. Open the override workspace to correct the default model output.`);
  } catch (error) {
    if (latestPanelResult) {
      renderPanels(latestPanelResult);
    }
    setStatus(error.message || "Something went wrong while running the pipeline.");
  } finally {
    submitButton.disabled = false;
  }
});

openOverridePageButton?.addEventListener("click", () => {
  if (!latestPanelResult) {
    setStatus("Run Phase 1 first, then open the override workspace.");
    return;
  }
  window.location.href = "/override";
});
