const overrideMeta = document.getElementById("override-meta");
const overridePanelsGrid = document.getElementById("override-panels-grid");
const applyOverridesButton = document.getElementById("apply-overrides-button");
const saveOverridesButton = document.getElementById("save-overrides-button");
const exportOverridesButton = document.getElementById("export-overrides-button");
const overrideJson = document.getElementById("override-json");
const overrideStatusText = document.getElementById("override-status-text");
const panelAnnotationSection = document.getElementById("panel-annotation-section");
const panelAnnotationCanvas = document.getElementById("panel-annotation-canvas");
const panelBoxList = document.getElementById("panel-box-list");
const clearPanelBoxesButton = document.getElementById("clear-panel-boxes-button");
const panelShapeModeSelect = document.getElementById("panel-shape-mode");
const undoPanelPointButton = document.getElementById("undo-panel-point-button");
const finishPanelShapeButton = document.getElementById("finish-panel-shape-button");

let latestPanelResult = null;
let latestDialogue = [];
let latestCaptions = [];
let panelOverrides = {};
let panelRegionOverrides = {};
let latestBubbleMode = "heuristic";
let latestSourceImageUrl = "";
let panelBoxOverrides = [];
let currentPanelPolygonPoints = [];
let currentPanelAnnotationScale = 1;
let activeAnnotationClassByPanel = {};
let displayPanelResult = null;
let loadedOverridePath = "";

function setStatus(message) {
  overrideStatusText.textContent = message;
}

async function fetchSavedOverrides(requestId) {
  const response = await fetch(`/panel-overrides/${encodeURIComponent(requestId)}`);
  if (!response.ok) {
    throw new Error("Failed to load saved overrides.");
  }
  return response.json();
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

function getDefaultPanelBoxes() {
  if (!latestPanelResult?.panel_boxes) {
    return [];
  }
  return latestPanelResult.panel_boxes.map((item, index) => ({
    index: item.index ?? index + 1,
    bbox: item.bbox.slice(),
    points: null,
    role: "panel",
  }));
}

function getEffectivePanelBoxes() {
  return panelBoxOverrides.length ? panelBoxOverrides : getDefaultPanelBoxes();
}

function getDisplayPanelResult() {
  return displayPanelResult || latestPanelResult;
}

function getVisiblePanelByIndex(panelIndex) {
  const visible = getDisplayPanelResult();
  return visible?.panel_boxes?.find((item) => String(item.index) === String(panelIndex)) || null;
}

function getSerializablePanelBoxes() {
  return getEffectivePanelBoxes().map((item) => ({
    index: item.index,
    bbox: item.bbox,
    points: item.points ?? null,
    role: item.role || "panel",
  }));
}

function getNextPanelBoxIndex() {
  const boxes = getEffectivePanelBoxes();
  const maxIndex = boxes.reduce((maxValue, item, index) => {
    const candidate = Number(item.index ?? index + 1);
    return Number.isFinite(candidate) ? Math.max(maxValue, candidate) : maxValue;
  }, 0);
  return maxIndex + 1;
}

function getSerializablePanelRegions() {
  const payload = {};
  for (const [panelIndex, regions] of Object.entries(panelRegionOverrides)) {
    payload[panelIndex] = regions.map((region) => ({
      class_name: region.class_name,
      bbox: region.bbox,
    }));
  }
  return payload;
}

function getPanelRegions(panelIndex) {
  return panelRegionOverrides[String(panelIndex)] || [];
}

function setPanelRegions(panelIndex, regions) {
  panelRegionOverrides[String(panelIndex)] = regions;
}

function getActiveAnnotationClass(panelIndex) {
  return activeAnnotationClassByPanel[String(panelIndex)] || "speech";
}

function setActiveAnnotationClass(panelIndex, className) {
  activeAnnotationClassByPanel[String(panelIndex)] = className;
}

function getCaptionForPanel(panelIndex, fallbackIndex = 0) {
  const target = String(panelIndex);
  return (
    latestCaptions.find((item) => String(item.panel) === target) ||
    latestCaptions[fallbackIndex] ||
    {}
  );
}

function getDialogueForPanel(panelIndex, fallbackIndex = 0) {
  const target = String(panelIndex);
  return (
    latestDialogue.find((item) => String(item.panel) === target) ||
    latestDialogue[fallbackIndex] ||
    {}
  );
}

function getModelRegions(panelIndex, fallbackIndex = 0) {
  const caption = getCaptionForPanel(panelIndex, fallbackIndex);
  return [
    ...(caption.speech_boxes || []).map((bbox) => ({ class_name: "speech", bbox })),
    ...(caption.narration_boxes || []).map((bbox) => ({ class_name: "narration", bbox })),
    ...(caption.sfx_boxes || []).map((bbox) => ({ class_name: "sfx", bbox })),
  ];
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Failed to load image: ${url}`));
    image.src = url;
  });
}

function cropImageToDataUrl(image, bbox) {
  const [x1, y1, x2, y2] = bbox.map((value) => Math.round(value));
  const width = Math.max(1, x2 - x1);
  const height = Math.max(1, y2 - y1);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, x1, y1, width, height, 0, 0, width, height);
  return canvas.toDataURL("image/png");
}

function describeBubbleSequence(boxes, panelWidth, panelHeight) {
  return boxes
    .slice()
    .sort((a, b) => {
      const rowA = (a.bbox[1] + a.bbox[3] / 2) / Math.max(panelHeight / 6, 1);
      const rowB = (b.bbox[1] + b.bbox[3] / 2) / Math.max(panelHeight / 6, 1);
      if (Math.floor(rowA) !== Math.floor(rowB)) {
        return rowA - rowB;
      }
      return (b.bbox[0] + b.bbox[2]) - (a.bbox[0] + a.bbox[2]);
    })
    .map((item) => {
      const [x, y, w, h] = item.bbox;
      const xCenter = x + w / 2;
      const yCenter = y + h / 2;
      const horizontal = xCenter >= panelWidth * 0.6 ? "right" : xCenter <= panelWidth * 0.4 ? "left" : "center";
      const vertical = yCenter <= panelHeight * 0.35 ? "top" : yCenter >= panelHeight * 0.68 ? "bottom" : "middle";
      return `${vertical}-${horizontal}`;
    });
}

function syncOverridesFromRegions(panelIndex, panelWidth, panelHeight) {
  const regions = getPanelRegions(panelIndex);
  const speech = regions.filter((item) => item.class_name === "speech").length;
  const narration = regions.filter((item) => item.class_name === "narration").length;
  const sfx = regions.filter((item) => item.class_name === "sfx").length;
  const speechBoxes = regions.filter((item) => item.class_name === "speech");
  const bubbleSequence = describeBubbleSequence(speechBoxes, panelWidth, panelHeight).join(" | ");
  panelOverrides[String(panelIndex)] = {
    ...(panelOverrides[String(panelIndex)] || {}),
    speech_count: String(speech),
    narration_count: String(narration),
    sfx_count: String(sfx),
    bubble_count: String(speech),
    bubble_sequence: bubbleSequence || "none",
  };
}

function attachOverrideListeners() {
  for (const button of overridePanelsGrid.querySelectorAll("[data-annotation-class-button]")) {
    button.addEventListener("click", (event) => {
      const panelIndex = event.currentTarget.dataset.panelIndex;
      const className = event.currentTarget.dataset.annotationClassButton;
      if (!panelIndex || !className) {
        return;
      }
      setActiveAnnotationClass(panelIndex, className);
      renderPanels();
      setStatus(`Panel ${panelIndex} annotation class set to ${className}. Draw on the image to add that region.`);
    });
  }

  const clearButtons = overridePanelsGrid.querySelectorAll("[data-clear-annotations]");
  for (const button of clearButtons) {
    button.addEventListener("click", (event) => {
      const panelIndex = event.currentTarget.dataset.clearAnnotations;
      if (!panelIndex) {
        return;
      }
      setPanelRegions(panelIndex, []);
      const canvas = overridePanelsGrid.querySelector(`[data-panel-canvas="${panelIndex}"]`);
      if (canvas) {
        initializePanelCanvas(canvas, getVisiblePanelByIndex(panelIndex));
      }
      const panel = getVisiblePanelByIndex(panelIndex);
      if (panel) {
        syncOverridesFromRegions(panelIndex, panel.bbox[2] - panel.bbox[0], panel.bbox[3] - panel.bbox[1]);
      }
      renderPanels();
    });
  }

  const deleteButtons = overridePanelsGrid.querySelectorAll("[data-delete-region]");
  for (const button of deleteButtons) {
    button.addEventListener("click", (event) => {
      const panelIndex = event.currentTarget.dataset.panelIndex;
      const regionIndex = Number(event.currentTarget.dataset.deleteRegion);
      if (!panelIndex || Number.isNaN(regionIndex)) {
        return;
      }
      const regions = getPanelRegions(panelIndex).slice();
      regions.splice(regionIndex, 1);
      setPanelRegions(panelIndex, regions);
      const panel = getVisiblePanelByIndex(panelIndex);
      if (panel) {
        syncOverridesFromRegions(panelIndex, panel.bbox[2] - panel.bbox[0], panel.bbox[3] - panel.bbox[1]);
      }
      renderPanels();
    });
  }

  const canvases = overridePanelsGrid.querySelectorAll("[data-panel-canvas]");
  for (const canvas of canvases) {
    const panel = getVisiblePanelByIndex(canvas.dataset.panelCanvas);
    initializePanelCanvas(canvas, panel);
  }

  const modelCanvases = overridePanelsGrid.querySelectorAll("[data-model-panel-canvas]");
  for (const canvas of modelCanvases) {
    const panel = getVisiblePanelByIndex(canvas.dataset.modelPanelCanvas);
    initializeModelPanelCanvas(canvas, panel);
  }
}

function renderPanels() {
  if (!latestPanelResult) {
    overrideMeta.hidden = true;
    panelAnnotationSection.hidden = true;
    overridePanelsGrid.innerHTML = '<p class="empty-state">Run Phase 1 on the review page, then open this workspace to correct the results.</p>';
    return;
  }

  overrideMeta.hidden = false;
  panelAnnotationSection.hidden = false;
  overrideMeta.innerHTML = `
    <span class="meta-chip">${latestPanelResult.panels} panels detected</span>
    <span class="meta-chip">${latestPanelResult.filename}</span>
    <span class="meta-chip">Request ${latestPanelResult.request_id}</span>
    <span class="meta-chip">Bubble mode ${latestBubbleMode}</span>
    ${loadedOverridePath ? `<span class="meta-chip">Saved override found</span>` : ""}
  `;

  const visiblePanelResult = getDisplayPanelResult();
  overridePanelsGrid.innerHTML = visiblePanelResult.panel_boxes
    .map((panel, index) => {
      const override = panelOverrides[panel.index] || {};
      const caption = getCaptionForPanel(panel.index, index);
      const dialogue = getDialogueForPanel(panel.index, index);
      const panelText = dialogue.text || "[no dialogue detected]";
      const bubbleCandidates = caption.bubble_candidates || "0";
      const modelBubbleCount = caption.bubble_count || "0";
      const bubbleSequence = caption.bubble_sequence || "none";
      const speechCount = override.speech_count ?? caption.speech_count ?? "0";
      const narrationCount = override.narration_count ?? caption.narration_count ?? "0";
      const sfxCount = override.sfx_count ?? caption.sfx_count ?? "0";
      const correctedBubbleCount = override.bubble_count ?? modelBubbleCount;
      const correctedBubbleSequence = override.bubble_sequence ?? bubbleSequence;
      const textRegionCount = dialogue.text_regions || "0";
      const textRole = dialogue.text_role || "ambient";
      const modelRegions = getModelRegions(panel.index, index);
      const annotationRegions = getPanelRegions(panel.index);
      const activeClass = getActiveAnnotationClass(panel.index);
      return `
        <article class="panel-card">
          <div class="panel-annotator">
            <section class="panel-stage">
              <div class="panel-stage-head">
                <span class="panel-stage-kicker">Step 1</span>
                <strong>Default Bubble Annotation</strong>
              </div>
              <p class="annotation-hint">Current model-detected speech, narration, and SFX regions for this panel.</p>
              <div class="panel-image-frame">
                <canvas class="override-annotation-canvas model-detection-canvas" data-model-panel-canvas="${panel.index}" data-image-url="${escapeAttribute(panel.image_url || latestSourceImageUrl)}"></canvas>
              </div>
              <div class="stack-list override-region-list model-region-list">
                ${
                  modelRegions.length
                    ? modelRegions
                        .map(
                          (region, regionIndex) => `
                            <article class="stack-item annotation-item model-region-item">
                              <div class="annotation-item-main">
                                <strong><span class="annotation-index-badge">#${regionIndex + 1}</span>${escapeHtml(region.class_name)}</strong>
                                <p>[${region.bbox.map((value) => Math.round(value)).join(", ")}]</p>
                              </div>
                            </article>
                          `,
                        )
                        .join("")
                    : '<p class="empty-state">No model-detected bubble regions for this panel.</p>'
                }
              </div>
            </section>
            <section class="panel-stage panel-stage-annotation">
              <div class="panel-stage-head">
                <span class="panel-stage-kicker">Step 2</span>
                <strong>Manual Bubble Annotation</strong>
              </div>
              <p class="annotation-hint">Choose a class below, then drag directly on the panel image to add or correct a region.</p>
              <div class="panel-image-frame panel-image-frame-secondary">
                <canvas class="override-annotation-canvas" data-panel-canvas="${panel.index}" data-image-url="${escapeAttribute(panel.image_url || latestSourceImageUrl)}"></canvas>
              </div>
              <div class="override-annotation-toolbar">
                <button type="button" class="secondary-button" data-clear-annotations="${panel.index}">Clear Boxes</button>
              </div>
              <div class="annotation-class-switcher">
                <button type="button" class="annotation-chip speech ${activeClass === "speech" ? "active" : ""}" data-panel-index="${panel.index}" data-annotation-class-button="speech">Speech</button>
                <button type="button" class="annotation-chip narration ${activeClass === "narration" ? "active" : ""}" data-panel-index="${panel.index}" data-annotation-class-button="narration">Narration</button>
                <button type="button" class="annotation-chip sfx ${activeClass === "sfx" ? "active" : ""}" data-panel-index="${panel.index}" data-annotation-class-button="sfx">SFX</button>
              </div>
              <div class="stack-list override-region-list">
                ${
                  annotationRegions.length
                    ? annotationRegions
                        .map(
                          (region, regionIndex) => `
                            <article class="stack-item annotation-item">
                              <div class="annotation-item-main">
                                <strong><span class="annotation-index-badge">#${regionIndex + 1}</span>${escapeHtml(region.class_name)}</strong>
                                <p>[${region.bbox.map((value) => Math.round(value)).join(", ")}]</p>
                              </div>
                              <button type="button" class="secondary-button" data-panel-index="${panel.index}" data-delete-region="${regionIndex}">Delete</button>
                            </article>
                          `,
                        )
                        .join("")
                    : '<p class="empty-state">No regions yet. Drag on the image above to annotate.</p>'
                }
              </div>
            </section>
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
              <p>Speech regions: ${escapeHtml(caption.speech_count || "0")}</p>
              <p>Narration regions: ${escapeHtml(caption.narration_count || "0")}</p>
              <p>SFX regions: ${escapeHtml(caption.sfx_count || "0")}</p>
              <p>Detected boxes shown on image: ${escapeHtml(modelRegions.length)}</p>
              <p>OCR text regions: ${escapeHtml(textRegionCount)}</p>
              <p>Text role: ${escapeHtml(textRole)}</p>
            </div>
            <div class="panel-ocr">
              <span class="panel-ocr-label">Annotated Summary</span>
              <p>Speech regions: ${escapeHtml(speechCount)}</p>
              <p>Narration regions: ${escapeHtml(narrationCount)}</p>
              <p>SFX regions: ${escapeHtml(sfxCount)}</p>
              <p>Bubble count: ${escapeHtml(correctedBubbleCount)}</p>
              <p>Bubble sequence: ${escapeHtml(correctedBubbleSequence)}</p>
            </div>
          </div>
        </article>
      `;
    })
    .join("");

  attachOverrideListeners();
  renderPanelBoxList();
  initializeFullPagePanelCanvas();
}

async function applyOverrides() {
  if (!latestPanelResult) {
    setStatus("Run Phase 1 on the review page first.");
    return;
  }
  try {
    const sourceImage = await loadImage(latestSourceImageUrl);
    const effectiveBoxes = getEffectivePanelBoxes()
      .filter((item) => (item.role || "panel") === "panel")
      .map((item, index) => ({
        index: item.index ?? index + 1,
        bbox: item.bbox.slice(),
        image_path: "",
        image_url: cropImageToDataUrl(sourceImage, item.bbox),
        role: item.role || "panel",
      }));
    displayPanelResult = {
      ...latestPanelResult,
      panels: effectiveBoxes.length,
      panel_boxes: effectiveBoxes,
    };
    renderPanels();
    overrideJson.hidden = false;
    overrideJson.textContent = JSON.stringify(
      {
        request_id: latestPanelResult.request_id,
        overrides: panelOverrides,
        panel_boxes: getSerializablePanelBoxes(),
        panel_regions: getSerializablePanelRegions(),
      },
      null,
      2,
    );
    setStatus("Applied corrected panel boxes and rebuilt panel crops in the workspace.");
  } catch (error) {
    setStatus(error.message || "Failed to rebuild corrected panel crops.");
  }
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
      panel_boxes: getSerializablePanelBoxes(),
      panel_regions: getSerializablePanelRegions(),
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
        panel_boxes: getSerializablePanelBoxes(),
        panel_regions: getSerializablePanelRegions(),
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
    overrideJson.textContent = JSON.stringify(
      {
        request_id: latestPanelResult.request_id,
        overrides: panelOverrides,
        panel_boxes: getSerializablePanelBoxes(),
        panel_regions: getSerializablePanelRegions(),
      },
      null,
      2,
    );
    setStatus(`Overrides saved to ${payload.overrides_path}`);
  } catch (error) {
    setStatus(error.message || "Failed to save overrides.");
  }
}

async function bootstrap() {
  const payload = readSessionPayload();
  if (!payload?.panelResult) {
    renderPanels();
    return;
  }
  latestPanelResult = payload.panelResult;
  displayPanelResult = null;
  latestDialogue = payload.dialogue || [];
  latestCaptions = payload.captions || [];
  latestBubbleMode = payload.bubbleMode || "heuristic";
  latestSourceImageUrl = payload.sourceImageUrl || payload.panelResult?.source_image_url || "";
  try {
    const saved = await fetchSavedOverrides(latestPanelResult.request_id);
    if (saved.exists) {
      panelOverrides = saved.overrides || {};
      panelBoxOverrides = (saved.panel_boxes || []).map((item) => ({
        index: item.index ?? null,
        bbox: item.bbox,
        points: item.points ?? null,
        role: item.role || "panel",
      })).map((item, index) => ({
        ...item,
        index: item.index ?? index + 1,
      }));
      panelRegionOverrides = saved.panel_regions || {};
      loadedOverridePath = saved.overrides_path || "";
      setStatus(`Loaded saved overrides from ${loadedOverridePath}`);
    }
  } catch (error) {
    setStatus(error.message || "Could not load saved overrides.");
  }
  renderPanels();
  if (!loadedOverridePath) {
    setStatus(`Loaded the latest Phase 1 run in ${latestBubbleMode} mode. You can now correct the default model output.`);
  }
}

function renderPanelBoxList() {
  const boxes = getEffectivePanelBoxes();
  panelBoxList.innerHTML = boxes.length
    ? boxes
        .map(
          (item, index) => `
            <article class="stack-item annotation-item">
              <div class="annotation-item-main">
                                <strong><span class="annotation-index-badge">#${index + 1}</span>${item.points?.length ? "polygon region" : "panel region"}</strong>
                                <p>ID: ${Math.round(item.index ?? index + 1)}</p>
                                <p>[${item.bbox.map((value) => Math.round(value)).join(", ")}]</p>
                              </div>
              <label class="override-field">
                <span>Role</span>
                <select data-panel-role-index="${index}">
                  <option value="panel" ${((item.role || "panel") === "panel") ? "selected" : ""}>panel</option>
                  <option value="background" ${((item.role || "panel") === "background") ? "selected" : ""}>background</option>
                  <option value="ignore" ${((item.role || "panel") === "ignore") ? "selected" : ""}>ignore</option>
                </select>
              </label>
              <button type="button" class="secondary-button" data-delete-panel-box="${index}">Delete</button>
            </article>
          `,
        )
        .join("")
    : '<p class="empty-state">Draw panel boxes on the full page to correct the split.</p>';

  for (const button of panelBoxList.querySelectorAll("[data-delete-panel-box]")) {
    button.addEventListener("click", (event) => {
      const index = Number(event.currentTarget.dataset.deletePanelBox);
      if (Number.isNaN(index)) {
        return;
      }
      const next = getEffectivePanelBoxes().slice();
      next.splice(index, 1);
      panelBoxOverrides = next;
      renderPanelBoxList();
      initializeFullPagePanelCanvas();
    });
  }

  for (const select of panelBoxList.querySelectorAll("[data-panel-role-index]")) {
    select.addEventListener("change", (event) => {
      const index = Number(event.currentTarget.dataset.panelRoleIndex);
      if (Number.isNaN(index)) {
        return;
      }
      const next = getEffectivePanelBoxes().slice();
      if (!next[index]) {
        return;
      }
      next[index] = { ...next[index], role: event.currentTarget.value };
      panelBoxOverrides = next;
      initializeFullPagePanelCanvas();
    });
  }
}

function initializeFullPagePanelCanvas() {
  if (!panelAnnotationCanvas || !latestSourceImageUrl || !latestPanelResult) {
    return;
  }
  const ctx = panelAnnotationCanvas.getContext("2d");
  const image = new Image();
  let scale = 1;
  let dragStart = null;
  let dragCurrent = null;

  function pointerPosition(event) {
    const rect = panelAnnotationCanvas.getBoundingClientRect();
    const scaleX = rect.width ? panelAnnotationCanvas.width / rect.width : 1;
    const scaleY = rect.height ? panelAnnotationCanvas.height / rect.height : 1;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  }

  function draw() {
    if (!image.complete) {
      return;
    }
    ctx.clearRect(0, 0, panelAnnotationCanvas.width, panelAnnotationCanvas.height);
    ctx.drawImage(image, 0, 0, panelAnnotationCanvas.width, panelAnnotationCanvas.height);
    ctx.strokeStyle = "#d04e23";
    ctx.lineWidth = 3;
    ctx.font = "bold 14px Space Grotesk";
    const boxes = getEffectivePanelBoxes();
    boxes.forEach((item, index) => {
      if (item.points?.length >= 3) {
        ctx.beginPath();
        item.points.forEach((point, pointIndex) => {
          const px = point[0] * scale;
          const py = point[1] * scale;
          if (pointIndex === 0) {
            ctx.moveTo(px, py);
          } else {
            ctx.lineTo(px, py);
          }
        });
        ctx.closePath();
        ctx.stroke();
      } else {
        const [x1, y1, x2, y2] = item.bbox;
        const x = x1 * scale;
        const y = y1 * scale;
        const w = (x2 - x1) * scale;
        const h = (y2 - y1) * scale;
        ctx.strokeRect(x, y, w, h);
      }
      const [x1, y1] = item.bbox;
      const x = x1 * scale;
      const y = y1 * scale;
      const label = `#${index + 1} ${item.role || "panel"}`;
      const labelWidth = ctx.measureText(label).width;
      ctx.fillStyle = "#d04e23";
      ctx.fillRect(x, y, labelWidth + 14, 22);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 7, y + 16);
      ctx.fillStyle = "#d04e23";
    });

    if (dragStart && dragCurrent) {
      const x = Math.min(dragStart.x, dragCurrent.x);
      const y = Math.min(dragStart.y, dragCurrent.y);
      const w = Math.abs(dragStart.x - dragCurrent.x);
      const h = Math.abs(dragStart.y - dragCurrent.y);
      ctx.strokeStyle = "#315fb3";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
    }

    if (currentPanelPolygonPoints.length) {
      ctx.strokeStyle = "#315fb3";
      ctx.fillStyle = "#315fb3";
      ctx.lineWidth = 2;
      currentPanelPolygonPoints.forEach((point, index) => {
        const px = point.x;
        const py = point.y;
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fill();
      });
      if (currentPanelPolygonPoints.length > 1) {
        ctx.beginPath();
        currentPanelPolygonPoints.forEach((point, index) => {
          if (index === 0) {
            ctx.moveTo(point.x, point.y);
          } else {
            ctx.lineTo(point.x, point.y);
          }
        });
        ctx.stroke();
      }
    }
  }

  function commitDrag() {
    if (!dragStart || !dragCurrent) {
      return;
    }
    const x1 = Math.round(Math.min(dragStart.x, dragCurrent.x) / scale);
    const y1 = Math.round(Math.min(dragStart.y, dragCurrent.y) / scale);
    const x2 = Math.round(Math.max(dragStart.x, dragCurrent.x) / scale);
    const y2 = Math.round(Math.max(dragStart.y, dragCurrent.y) / scale);
    dragStart = null;
    dragCurrent = null;
    if ((x2 - x1) >= 16 && (y2 - y1) >= 16) {
      panelBoxOverrides = [
        ...getEffectivePanelBoxes(),
        { index: getNextPanelBoxIndex(), bbox: [x1, y1, x2, y2], points: null, role: "panel" },
      ];
      renderPanelBoxList();
      draw();
      return;
    }
    draw();
  }

  function commitPolygon() {
    if (currentPanelPolygonPoints.length < 3) {
      return;
    }
    const normalized = currentPanelPolygonPoints.map((point) => [
      Math.round(point.x / currentPanelAnnotationScale),
      Math.round(point.y / currentPanelAnnotationScale),
    ]);
    const xs = normalized.map((point) => point[0]);
    const ys = normalized.map((point) => point[1]);
    const bbox = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    panelBoxOverrides = [
      ...getEffectivePanelBoxes(),
      { index: getNextPanelBoxIndex(), bbox, points: normalized, role: "panel" },
    ];
    currentPanelPolygonPoints = [];
    renderPanelBoxList();
    draw();
  }

  image.onload = () => {
    const shell = panelAnnotationCanvas.parentElement;
    const availableWidth = shell ? Math.max(320, shell.clientWidth - 8) : 760;
    scale = Math.min(1, availableWidth / image.naturalWidth);
    currentPanelAnnotationScale = scale;
    panelAnnotationCanvas.width = Math.round(image.naturalWidth * scale);
    panelAnnotationCanvas.height = Math.round(image.naturalHeight * scale);
    draw();
  };
  image.src = latestSourceImageUrl;

  panelAnnotationCanvas.onpointerdown = (event) => {
    const mode = panelShapeModeSelect?.value || "rect";
    if (mode === "polygon") {
      currentPanelPolygonPoints = [...currentPanelPolygonPoints, pointerPosition(event)];
      draw();
      return;
    }
    dragStart = pointerPosition(event);
    dragCurrent = dragStart;
    draw();
  };
  panelAnnotationCanvas.onpointermove = (event) => {
    const mode = panelShapeModeSelect?.value || "rect";
    if (mode === "polygon") {
      return;
    }
    if (!dragStart) {
      return;
    }
    dragCurrent = pointerPosition(event);
    draw();
  };
  panelAnnotationCanvas.onpointerup = () => {
    const mode = panelShapeModeSelect?.value || "rect";
    if (mode === "polygon") {
      return;
    }
    commitDrag();
  };
  panelAnnotationCanvas.onpointerleave = () => {
    const mode = panelShapeModeSelect?.value || "rect";
    if (mode === "polygon") {
      return;
    }
    if (dragStart) {
      commitDrag();
    }
  };
}

function initializePanelCanvas(canvas, panel) {
  if (!canvas || !panel) {
    return;
  }
  const imageUrl = canvas.dataset.imageUrl;
  const ctx = canvas.getContext("2d");
  const image = new Image();
  const panelIndex = String(panel.index);
  let scale = 1;
  let dragStart = null;
  let dragCurrent = null;
  const panelWidth = panel.bbox[2] - panel.bbox[0];
  const panelHeight = panel.bbox[3] - panel.bbox[1];
  const modelRegions = getModelRegions(panel.index);

  function currentClass() {
    return getActiveAnnotationClass(panelIndex);
  }

  function pointerPosition(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = rect.width ? canvas.width / rect.width : 1;
    const scaleY = rect.height ? canvas.height / rect.height : 1;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  }

  function draw() {
    if (!image.complete) {
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const colors = { speech: "#d04e23", narration: "#1c7a5f", sfx: "#315fb3" };
    ctx.setLineDash([8, 6]);
    for (const [regionIndex, region] of modelRegions.entries()) {
      const [x, y, w, h] = region.bbox;
      const scaledX = x * scale;
      const scaledY = y * scale;
      const scaledW = w * scale;
      const scaledH = h * scale;
      const color = colors[region.class_name] || "#666";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
      ctx.fillStyle = color;
      ctx.font = "bold 13px Space Grotesk";
      const label = `model ${regionIndex + 1} ${region.class_name}`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillRect(scaledX, scaledY, textWidth + 14, 20);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, scaledX + 7, scaledY + 14);
    }
    ctx.setLineDash([]);
    for (const [regionIndex, region] of getPanelRegions(panelIndex).entries()) {
      const [x, y, w, h] = region.bbox;
      const scaledX = x * scale;
      const scaledY = y * scale;
      const scaledW = w * scale;
      const scaledH = h * scale;
      const color = colors[region.class_name] || "#d04e23";
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
      ctx.fillStyle = color;
      ctx.font = "bold 14px Space Grotesk";
      const label = `#${regionIndex + 1} ${region.class_name}`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillRect(scaledX, scaledY, textWidth + 14, 22);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, scaledX + 7, scaledY + 16);
    }

    if (dragStart && dragCurrent) {
      const x = Math.min(dragStart.x, dragCurrent.x);
      const y = Math.min(dragStart.y, dragCurrent.y);
      const w = Math.abs(dragStart.x - dragCurrent.x);
      const h = Math.abs(dragStart.y - dragCurrent.y);
      ctx.strokeStyle = colors[currentClass()] || "#d04e23";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
    }
  }

  function commitDrag() {
    if (!dragStart || !dragCurrent) {
      return;
    }
    const x = Math.min(dragStart.x, dragCurrent.x) / scale;
    const y = Math.min(dragStart.y, dragCurrent.y) / scale;
    const w = Math.abs(dragStart.x - dragCurrent.x) / scale;
    const h = Math.abs(dragStart.y - dragCurrent.y) / scale;
    if (w >= 8 && h >= 8) {
      const next = getPanelRegions(panelIndex).slice();
      next.push({
        class_name: currentClass(),
        bbox: [x, y, w, h],
      });
      setPanelRegions(panelIndex, next);
      syncOverridesFromRegions(panelIndex, panelWidth, panelHeight);
      renderPanels();
      return;
    }
    dragStart = null;
    dragCurrent = null;
    draw();
  }

  image.onload = () => {
    const shell = canvas.parentElement;
    const availableWidth = shell ? Math.max(220, shell.clientWidth - 24) : 360;
    const availableHeight = 520;
    scale = Math.min(1, availableWidth / image.naturalWidth, availableHeight / image.naturalHeight);
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);
    draw();
  };
  image.src = imageUrl;

  canvas.onpointerdown = (event) => {
    dragStart = pointerPosition(event);
    dragCurrent = dragStart;
    draw();
  };
  canvas.onpointermove = (event) => {
    if (!dragStart) {
      return;
    }
    dragCurrent = pointerPosition(event);
    draw();
  };
  canvas.onpointerup = () => {
    commitDrag();
  };
  canvas.onpointerleave = () => {
    if (dragStart) {
      commitDrag();
    }
  };
}

function initializeModelPanelCanvas(canvas, panel) {
  if (!canvas || !panel) {
    return;
  }
  const imageUrl = canvas.dataset.imageUrl;
  const ctx = canvas.getContext("2d");
  const image = new Image();
  const modelRegions = getModelRegions(panel.index);
  let scale = 1;

  function draw() {
    if (!image.complete) {
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const colors = { speech: "#d04e23", narration: "#1c7a5f", sfx: "#315fb3" };
    ctx.setLineDash([8, 6]);
    for (const [regionIndex, region] of modelRegions.entries()) {
      const [x, y, w, h] = region.bbox;
      const scaledX = x * scale;
      const scaledY = y * scale;
      const scaledW = w * scale;
      const scaledH = h * scale;
      const color = colors[region.class_name] || "#666";
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
      ctx.fillStyle = color;
      ctx.font = "bold 13px Space Grotesk";
      const label = `#${regionIndex + 1} ${region.class_name}`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillRect(scaledX, scaledY, textWidth + 14, 20);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, scaledX + 7, scaledY + 14);
    }
    ctx.setLineDash([]);
  }

  image.onload = () => {
    const shell = canvas.parentElement;
    const availableWidth = shell ? Math.max(220, shell.clientWidth - 24) : 360;
    const availableHeight = 520;
    scale = Math.min(1, availableWidth / image.naturalWidth, availableHeight / image.naturalHeight);
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);
    draw();
  };
  image.src = imageUrl;
}

applyOverridesButton?.addEventListener("click", applyOverrides);
saveOverridesButton?.addEventListener("click", saveOverrides);
exportOverridesButton?.addEventListener("click", exportOverrides);
clearPanelBoxesButton?.addEventListener("click", () => {
  panelBoxOverrides = [];
  renderPanelBoxList();
  initializeFullPagePanelCanvas();
  setStatus("Reset panel boxes to the detector output.");
});

panelShapeModeSelect?.addEventListener("change", () => {
  currentPanelPolygonPoints = [];
  initializeFullPagePanelCanvas();
  const mode = panelShapeModeSelect.value === "polygon" ? "Polygon" : "Rectangle";
  setStatus(`Panel annotation mode set to ${mode}.`);
});

undoPanelPointButton?.addEventListener("click", () => {
  if (!currentPanelPolygonPoints.length) {
    return;
  }
  currentPanelPolygonPoints = currentPanelPolygonPoints.slice(0, -1);
  initializeFullPagePanelCanvas();
  setStatus("Removed the last polygon point.");
});

finishPanelShapeButton?.addEventListener("click", () => {
  if ((panelShapeModeSelect?.value || "rect") !== "polygon") {
    setStatus("Switch to Polygon mode to finish a multi-point panel shape.");
    return;
  }
  if (currentPanelPolygonPoints.length < 3) {
    setStatus("Add at least 3 points before finishing the panel shape.");
    return;
  }
  const ctxCanvas = panelAnnotationCanvas;
  if (!ctxCanvas) {
    return;
  }
  const imageWidth = ctxCanvas.width || 1;
  if (!imageWidth) {
    return;
  }
  const scaleGuess = latestSourceImageUrl ? (ctxCanvas.width / (ctxCanvas.width / 1)) : 1;
  void scaleGuess;
  const eventlessCommit = true;
  if (eventlessCommit) {
    const normalized = currentPanelPolygonPoints.map((point) => [
      Math.round(point.x / currentPanelAnnotationScale),
      Math.round(point.y / currentPanelAnnotationScale),
    ]);
    const xs = normalized.map((point) => point[0]);
    const ys = normalized.map((point) => point[1]);
    const bbox = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    panelBoxOverrides = [...getEffectivePanelBoxes(), { index: getNextPanelBoxIndex(), bbox, points: normalized, role: "panel" }];
    currentPanelPolygonPoints = [];
    renderPanelBoxList();
    initializeFullPagePanelCanvas();
    setStatus("Polygon panel shape added.");
  }
});

bootstrap();
