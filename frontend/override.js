const overrideMeta = document.getElementById("override-meta");
const overridePanelsGrid = document.getElementById("override-panels-grid");
const applyOverridesButton = document.getElementById("apply-overrides-button");
const saveOverridesButton = document.getElementById("save-overrides-button");
const exportOverridesButton = document.getElementById("export-overrides-button");
const overrideJson = document.getElementById("override-json");
const overrideStatusText = document.getElementById("override-status-text");

let latestPanelResult = null;
let latestDialogue = [];
let latestCaptions = [];
let panelOverrides = {};
let latestBubbleMode = "heuristic";

function setStatus(message) {
  overrideStatusText.textContent = message;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll('"', "&quot;");
}

function readSessionPayload() {
  const raw = sessionStorage.getItem("phase1_override_payload");
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw);
  } catch (error) {
    return null;
  }
}

function attachOverrideListeners() {
  const inputs = overridePanelsGrid.querySelectorAll("[data-panel-index][data-override-key]");
  for (const input of inputs) {
    input.addEventListener("input", handleOverrideInput);
  }
}

function handleOverrideInput(event) {
  const target = event.target;
  const panelIndex = target.dataset.panelIndex;
  const overrideKey = target.dataset.overrideKey;
  if (!panelIndex || !overrideKey) {
    return;
  }
  const current = panelOverrides[panelIndex] || {};
  current[overrideKey] = target.value;
  panelOverrides[panelIndex] = current;
}

function renderPanels() {
  if (!latestPanelResult) {
    overrideMeta.hidden = true;
    overridePanelsGrid.innerHTML = '<p class="empty-state">Run Phase 1 on the review page, then open this workspace to correct the results.</p>';
    return;
  }

  overrideMeta.hidden = false;
  overrideMeta.innerHTML = `
    <span class="meta-chip">${latestPanelResult.panels} panels detected</span>
    <span class="meta-chip">${latestPanelResult.filename}</span>
    <span class="meta-chip">Request ${latestPanelResult.request_id}</span>
    <span class="meta-chip">Bubble mode ${latestBubbleMode}</span>
  `;

  overridePanelsGrid.innerHTML = latestPanelResult.panel_boxes
    .map((panel, index) => {
      const override = panelOverrides[panel.index] || {};
      const panelText = latestDialogue[index]?.text || "[no dialogue detected]";
      const bubbleCandidates = latestCaptions[index]?.bubble_candidates || "0";
      const modelBubbleCount = latestCaptions[index]?.bubble_count || "0";
      const bubbleSequence = latestCaptions[index]?.bubble_sequence || "none";
      const speechCount = override.speech_count ?? latestCaptions[index]?.speech_count ?? "0";
      const narrationCount = override.narration_count ?? latestCaptions[index]?.narration_count ?? "0";
      const sfxCount = override.sfx_count ?? latestCaptions[index]?.sfx_count ?? "0";
      const correctedBubbleCount = override.bubble_count ?? modelBubbleCount;
      const correctedBubbleSequence = override.bubble_sequence ?? bubbleSequence;
      const textRegionCount = latestDialogue[index]?.text_regions || "0";
      const textRole = latestDialogue[index]?.text_role || "ambient";
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
              <span class="panel-ocr-label">Default Model Output</span>
              <p>Bubble candidates: ${escapeHtml(bubbleCandidates)}</p>
              <p>Speech bubbles: ${escapeHtml(modelBubbleCount)}</p>
              <p>Bubble sequence: ${escapeHtml(bubbleSequence)}</p>
              <p>Speech regions: ${escapeHtml(latestCaptions[index]?.speech_count || "0")}</p>
              <p>Narration regions: ${escapeHtml(latestCaptions[index]?.narration_count || "0")}</p>
              <p>SFX regions: ${escapeHtml(latestCaptions[index]?.sfx_count || "0")}</p>
              <p>OCR text regions: ${escapeHtml(textRegionCount)}</p>
              <p>Text role: ${escapeHtml(textRole)}</p>
            </div>
            <div class="panel-ocr">
              <span class="panel-ocr-label">Manual Override</span>
              <div class="override-grid">
                <label class="override-field">
                  <span>Speech</span>
                  <small>Character dialogue regions inside speech bubbles.</small>
                  <input type="number" min="0" value="${escapeAttribute(String(speechCount))}" data-panel-index="${panel.index}" data-override-key="speech_count" />
                </label>
                <label class="override-field">
                  <span>Narration</span>
                  <small>Caption or story text boxes, not spoken dialogue.</small>
                  <input type="number" min="0" value="${escapeAttribute(String(narrationCount))}" data-panel-index="${panel.index}" data-override-key="narration_count" />
                </label>
                <label class="override-field">
                  <span>SFX</span>
                  <small>Sound-effect text drawn into the artwork.</small>
                  <input type="number" min="0" value="${escapeAttribute(String(sfxCount))}" data-panel-index="${panel.index}" data-override-key="sfx_count" />
                </label>
                <label class="override-field">
                  <span>Bubbles</span>
                  <small>Total speech bubbles you want counted for this panel.</small>
                  <input type="number" min="0" value="${escapeAttribute(String(correctedBubbleCount))}" data-panel-index="${panel.index}" data-override-key="bubble_count" />
                </label>
              </div>
              <label class="override-field override-field-wide">
                <span>Bubble Sequence</span>
                <small>Bubble positions in reading order, for example: top-right | middle-left.</small>
                <input type="text" value="${escapeAttribute(String(correctedBubbleSequence))}" data-panel-index="${panel.index}" data-override-key="bubble_sequence" />
              </label>
            </div>
          </div>
        </article>
      `;
    })
    .join("");

  attachOverrideListeners();
}

function applyOverrides() {
  if (!latestPanelResult) {
    setStatus("Run Phase 1 on the review page first.");
    return;
  }
  renderPanels();
  overrideJson.hidden = false;
  overrideJson.textContent = JSON.stringify(panelOverrides, null, 2);
  setStatus("Manual overrides applied in the workspace.");
}

function exportOverrides() {
  if (!latestPanelResult) {
    setStatus("Run Phase 1 on the review page first.");
    return;
  }
  overrideJson.hidden = false;
  overrideJson.textContent = JSON.stringify(
    {
      request_id: latestPanelResult.request_id,
      overrides: panelOverrides,
    },
    null,
    2,
  );
  setStatus("Override JSON ready below the panel list.");
}

async function saveOverrides() {
  if (!latestPanelResult?.request_id) {
    setStatus("Run Phase 1 on the review page first.");
    return;
  }
  try {
    const response = await fetch("/panel-overrides", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        request_id: latestPanelResult.request_id,
        overrides: panelOverrides,
      }),
    });
    if (!response.ok) {
      let message = "Failed to save overrides.";
      try {
        const payload = await response.json();
        message = payload.detail || message;
      } catch (error) {
        message = `${message} (${response.status})`;
      }
      throw new Error(message);
    }
    const payload = await response.json();
    overrideJson.hidden = false;
    overrideJson.textContent = JSON.stringify(panelOverrides, null, 2);
    setStatus(`Overrides saved to ${payload.overrides_path}`);
  } catch (error) {
    setStatus(error.message || "Failed to save overrides.");
  }
}

function bootstrap() {
  const payload = readSessionPayload();
  if (!payload?.panelResult) {
    renderPanels();
    return;
  }
  latestPanelResult = payload.panelResult;
  latestDialogue = payload.dialogue || [];
  latestCaptions = payload.captions || [];
  latestBubbleMode = payload.bubbleMode || "heuristic";
  renderPanels();
  setStatus(`Loaded the latest Phase 1 run in ${latestBubbleMode} mode. You can now correct the default model output.`);
}

applyOverridesButton?.addEventListener("click", applyOverrides);
saveOverridesButton?.addEventListener("click", saveOverrides);
exportOverridesButton?.addEventListener("click", exportOverrides);

bootstrap();
