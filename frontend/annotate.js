const datasetSelect = document.getElementById("dataset-root");
const splitSelect = document.getElementById("dataset-split");
const classSelect = document.getElementById("annotation-class");
const shapeModeSelect = document.getElementById("annotation-shape-mode");
const viewModeSelect = document.getElementById("annotation-view-mode");
const queueItemSelect = document.getElementById("queue-item-select");
const prevImageButton = document.getElementById("prev-image-button");
const nextImageButton = document.getElementById("next-image-button");
const saveAnnotationsButton = document.getElementById("save-annotations-button");
const exportDatasetButton = document.getElementById("export-dataset-button");
const exportValidatedBubbleButton = document.getElementById("export-validated-bubble-button");
const undoAnnotationPointButton = document.getElementById("undo-annotation-point-button");
const finishAnnotationShapeButton = document.getElementById("finish-annotation-shape-button");
const annotationStatus = document.getElementById("annotation-status");
const annotationMeta = document.getElementById("annotation-meta");
const annotationInspector = document.getElementById("annotation-inspector");
const annotationList = document.getElementById("annotation-list");
const canvas = document.getElementById("annotation-canvas");
const ctx = canvas.getContext("2d");

const classColors = {
  speech_bubble: "#d04e23",
  narration_box: "#1c7a5f",
  sfx: "#315fb3",
};
const preferredValidationDataset = "new object training data.v1.coco";

let datasetSummary = { total: 0 };
let currentItem = null;
let currentImage = null;
let currentScale = 1;
let currentAnnotations = [];
let currentIndex = 0;
let dragStart = null;
let dragCurrent = null;
let currentPolygonPoints = [];
let selectedAnnotationIndex = null;
let availableDatasets = [];
let reviewQueue = [];
let reviewQueuePosition = -1;
let hasAppliedUrlState = false;

function currentQuery() {
  return new URLSearchParams(window.location.search);
}

function setStatus(message) {
  annotationStatus.textContent = message;
}

function currentDatasetKey() {
  return datasetSelect.value || "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Request failed for ${url}`;
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

async function loadSummary() {
  datasetSummary = await fetchJson(
    `/api/annotation/images?dataset=${encodeURIComponent(currentDatasetKey())}&split=${encodeURIComponent(splitSelect.value)}&offset=0&limit=1`,
  );
  availableDatasets = datasetSummary.available_datasets || availableDatasets;
  renderDatasetOptions();
}

function currentViewMode() {
  return viewModeSelect.value || "all";
}

async function loadReviewQueue() {
  const payload = await fetchJson(`/api/annotation/review-queue?dataset=${encodeURIComponent(currentDatasetKey())}`);
  reviewQueue = payload.items || [];
  renderQueueOptions();
}

async function loadItem(index, splitOverride = null) {
  const effectiveSplit = splitOverride || splitSelect.value;
  currentIndex = Math.max(0, index);
  splitSelect.value = effectiveSplit;
  await loadSummary();
  const payload = await fetchJson(
    `/api/annotation/item?dataset=${encodeURIComponent(currentDatasetKey())}&split=${encodeURIComponent(effectiveSplit)}&index=${currentIndex}`,
  );
  currentItem = payload;
  currentAnnotations = (payload.annotations || []).map((item, idx) => ({
    id: item.id || String(idx + 1),
    class_name: item.class_name,
    bbox: [...item.bbox],
    points: Array.isArray(item.points) ? item.points.map((point) => [...point]) : null,
  }));
  currentPolygonPoints = [];
  selectedAnnotationIndex = currentAnnotations.length ? 0 : null;
  await loadImage(payload.image_url);
  renderMeta();
  renderInspector();
  renderAnnotationList();
  drawCanvas();
  const queuePrefix = reviewQueuePosition >= 0 ? `Queue ${reviewQueuePosition + 1}/${reviewQueue.length}. ` : "";
  setStatus(`${queuePrefix}Loaded ${payload.file_name} from ${payload.annotation_source} annotations.`);
  updateNavButtons();
  syncUrlState();
}

function updateModeControls() {
  const inReviewMode = currentViewMode() === "review";
  splitSelect.disabled = inReviewMode;
  queueItemSelect.disabled = inReviewMode ? !reviewQueue.length : true;
}

function renderDatasetOptions() {
  if (!availableDatasets.length) {
    return;
  }
  const currentValue = currentDatasetKey() || availableDatasets[0].key;
  datasetSelect.innerHTML = availableDatasets
    .map((item) => `<option value="${escapeHtml(item.key)}" ${item.key === currentValue ? "selected" : ""}>${escapeHtml(item.name)}</option>`)
    .join("");
}

async function loadImage(url) {
  currentImage = await new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load dataset image."));
    image.src = url;
  });
  const maxWidth = Math.min(window.innerWidth * 0.65, 900);
  currentScale = Math.min(1, maxWidth / currentImage.naturalWidth);
  canvas.width = Math.round(currentImage.naturalWidth * currentScale);
  canvas.height = Math.round(currentImage.naturalHeight * currentScale);
}

function renderMeta() {
  if (!currentItem) {
    annotationMeta.innerHTML = "";
    return;
  }
  annotationMeta.innerHTML = `
    <span class="meta-chip">${escapeHtml(currentItem.dataset)}</span>
    <span class="meta-chip">${currentItem.split}</span>
    <span class="meta-chip">Image ${currentItem.index + 1} / ${datasetSummary.total}</span>
    ${reviewQueuePosition >= 0 ? `<span class="meta-chip">Queue ${reviewQueuePosition + 1} / ${reviewQueue.length}</span>` : ""}
    <span class="meta-chip">${escapeHtml(currentItem.file_name)}</span>
    <span class="meta-chip">${currentItem.annotation_source}</span>
    <span class="meta-chip">${currentAnnotations.length} labels</span>
  `;
}

function renderQueueOptions() {
  if (!reviewQueue.length) {
    queueItemSelect.innerHTML = '<option value="">No queued files</option>';
    queueItemSelect.disabled = true;
    return;
  }

  queueItemSelect.disabled = currentViewMode() !== "review";
  queueItemSelect.innerHTML = reviewQueue
    .map((item) => {
      const reasons = item.reasons.slice(0, 2).join(", ");
      const classText = item.class_names.length ? item.class_names.join(", ") : "unlabeled";
      const label = `#${item.queue_index + 1} ${item.split} / ${item.file_name} [${classText}] ${reasons}`;
      return `<option value="${item.queue_index}" ${item.queue_index === reviewQueuePosition ? "selected" : ""}>${escapeHtml(label)}</option>`;
    })
    .join("");
}

function renderAnnotationList() {
  if (!currentAnnotations.length) {
    annotationList.innerHTML = '<p class="empty-state">No annotations yet for this image.</p>';
    return;
  }
  annotationList.innerHTML = currentAnnotations
    .map(
      (item, index) => `
        <article class="stack-item annotation-item ${index === selectedAnnotationIndex ? "is-selected" : ""}" data-annotation-select="${index}">
          <div class="annotation-item-main">
            <strong><span class="annotation-index-badge">#${index + 1}</span>${escapeHtml(item.class_name)}</strong>
            <p>${item.points?.length ? "polygon" : "rectangle"}</p>
            <p>[${item.bbox.map((value) => Math.round(value)).join(", ")}]</p>
          </div>
          <button type="button" class="secondary-button annotation-delete-button" data-annotation-index="${index}">Delete</button>
        </article>
      `,
    )
    .join("");

  for (const row of annotationList.querySelectorAll("[data-annotation-select]")) {
    row.addEventListener("click", (event) => {
      if (event.target.closest("[data-annotation-index]")) {
        return;
      }
      selectAnnotation(Number(row.dataset.annotationSelect));
    });
  }

  for (const button of annotationList.querySelectorAll("[data-annotation-index]")) {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const index = Number(event.currentTarget.dataset.annotationIndex);
      currentAnnotations.splice(index, 1);
      if (selectedAnnotationIndex === index) {
        selectedAnnotationIndex = currentAnnotations.length ? Math.min(index, currentAnnotations.length - 1) : null;
      } else if (selectedAnnotationIndex !== null && index < selectedAnnotationIndex) {
        selectedAnnotationIndex -= 1;
      }
      renderMeta();
      renderInspector();
      renderAnnotationList();
      drawCanvas();
    });
  }
}

function renderInspector() {
  if (selectedAnnotationIndex === null || !currentAnnotations[selectedAnnotationIndex]) {
    annotationInspector.innerHTML = '<p class="empty-state">Select a box on the canvas or from the list to edit it.</p>';
    return;
  }

  const item = currentAnnotations[selectedAnnotationIndex];
  annotationInspector.innerHTML = `
    <div class="annotation-inspector-header">
      <div>
        <p class="section-kicker">Selected Label</p>
        <h3>#${selectedAnnotationIndex + 1} ${escapeHtml(item.class_name)}</h3>
      </div>
      <span class="meta-chip">${item.points?.length ? "polygon" : "rectangle"} [${item.bbox.map((value) => Math.round(value)).join(", ")}]</span>
    </div>
    <p class="annotation-inspector-help">Click another box to switch selection. Change the class here without redrawing the box.</p>
    <div class="annotation-inspector-controls">
      <label class="override-field">
        <span>Class</span>
        <select id="selected-annotation-class">
          <option value="speech_bubble" ${item.class_name === "speech_bubble" ? "selected" : ""}>speech_bubble</option>
          <option value="narration_box" ${item.class_name === "narration_box" ? "selected" : ""}>narration_box</option>
          <option value="sfx" ${item.class_name === "sfx" ? "selected" : ""}>sfx</option>
        </select>
      </label>
      <div class="annotation-inspector-actions">
        <button type="button" id="apply-annotation-class-button">Apply Class</button>
        <button type="button" id="delete-selected-annotation-button" class="secondary-button">Delete Selected</button>
      </div>
    </div>
  `;

  annotationInspector.querySelector("#apply-annotation-class-button").addEventListener("click", () => {
    const nextClass = annotationInspector.querySelector("#selected-annotation-class").value;
    currentAnnotations[selectedAnnotationIndex].class_name = nextClass;
    classSelect.value = nextClass;
    renderInspector();
    renderAnnotationList();
    drawCanvas();
  });

  annotationInspector.querySelector("#delete-selected-annotation-button").addEventListener("click", () => {
    currentAnnotations.splice(selectedAnnotationIndex, 1);
    selectedAnnotationIndex = currentAnnotations.length ? Math.min(selectedAnnotationIndex, currentAnnotations.length - 1) : null;
    renderMeta();
    renderInspector();
    renderAnnotationList();
    drawCanvas();
  });
}

function selectAnnotation(index) {
  if (index < 0 || index >= currentAnnotations.length) {
    return;
  }
  selectedAnnotationIndex = index;
  renderInspector();
  renderAnnotationList();
  drawCanvas();
}

function drawCanvas() {
  if (!currentImage) {
    return;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);

  currentAnnotations.forEach((item, index) => {
    const color = classColors[item.class_name] || "#b44d25";
    const isPolygon = Array.isArray(item.points) && item.points.length >= 3;
    const [x, y, w, h] = item.bbox;
    const scaledX = x * currentScale;
    const scaledY = y * currentScale;
    if (isPolygon) {
      const scaledPoints = item.points.map((point) => [point[0] * currentScale, point[1] * currentScale]);
      if (index === selectedAnnotationIndex) {
        ctx.strokeStyle = "#ffd54f";
        ctx.lineWidth = 6;
        ctx.beginPath();
        scaledPoints.forEach(([px, py], pointIndex) => {
          if (pointIndex === 0) {
            ctx.moveTo(px, py);
          } else {
            ctx.lineTo(px, py);
          }
        });
        ctx.closePath();
        ctx.stroke();
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      scaledPoints.forEach(([px, py], pointIndex) => {
        if (pointIndex === 0) {
          ctx.moveTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      });
      ctx.closePath();
      ctx.stroke();
      ctx.fillStyle = color;
      scaledPoints.forEach(([px, py]) => {
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    } else {
      const scaledW = w * currentScale;
      const scaledH = h * currentScale;
      if (index === selectedAnnotationIndex) {
        ctx.strokeStyle = "#ffd54f";
        ctx.lineWidth = 6;
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
      ctx.fillStyle = color;
    }
    ctx.font = "bold 14px Space Grotesk";
    const label = `#${index + 1} ${item.class_name}`;
    const textWidth = ctx.measureText(label).width;
    const chipWidth = textWidth + 14;
    const chipHeight = 22;
    ctx.fillRect(scaledX, Math.max(0, scaledY), chipWidth, chipHeight);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, scaledX + 7, Math.max(16, scaledY + 16));
  });

  if (dragStart && dragCurrent) {
    const x = Math.min(dragStart.x, dragCurrent.x);
    const y = Math.min(dragStart.y, dragCurrent.y);
    const w = Math.abs(dragStart.x - dragCurrent.x);
    const h = Math.abs(dragStart.y - dragCurrent.y);
    ctx.strokeStyle = classColors[classSelect.value] || "#b44d25";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
  }
  if (currentPolygonPoints.length) {
    ctx.strokeStyle = classColors[classSelect.value] || "#b44d25";
    ctx.lineWidth = 2;
    ctx.beginPath();
    currentPolygonPoints.forEach((point, index) => {
      if (index === 0) {
        ctx.moveTo(point.x, point.y);
      } else {
        ctx.lineTo(point.x, point.y);
      }
    });
    ctx.stroke();
    ctx.fillStyle = classColors[classSelect.value] || "#b44d25";
    currentPolygonPoints.forEach((point) => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }
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

function findAnnotationAtPoint(point) {
  for (let index = currentAnnotations.length - 1; index >= 0; index -= 1) {
    const item = currentAnnotations[index];
    if (Array.isArray(item.points) && item.points.length >= 3) {
      if (pointInPolygon(point, item.points.map((polygonPoint) => ({ x: polygonPoint[0] * currentScale, y: polygonPoint[1] * currentScale })))) {
        return index;
      }
      continue;
    }
    const [x, y, w, h] = item.bbox;
    const scaledX = x * currentScale;
    const scaledY = y * currentScale;
    const scaledW = w * currentScale;
    const scaledH = h * currentScale;
    if (point.x >= scaledX && point.x <= scaledX + scaledW && point.y >= scaledY && point.y <= scaledY + scaledH) {
      return index;
    }
  }
  return null;
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersects = ((yi > point.y) !== (yj > point.y))
      && (point.x < ((xj - xi) * (point.y - yi)) / ((yj - yi) || 1e-9) + xi);
    if (intersects) {
      inside = !inside;
    }
  }
  return inside;
}

function commitDragBox() {
  if (!dragStart || !dragCurrent) {
    return;
  }
  const x = Math.min(dragStart.x, dragCurrent.x) / currentScale;
  const y = Math.min(dragStart.y, dragCurrent.y) / currentScale;
  const w = Math.abs(dragStart.x - dragCurrent.x) / currentScale;
  const h = Math.abs(dragStart.y - dragCurrent.y) / currentScale;
  if (w >= 8 && h >= 8) {
    currentAnnotations.push({
      id: `${Date.now()}-${currentAnnotations.length + 1}`,
      class_name: classSelect.value,
      bbox: [x, y, w, h],
      points: null,
    });
    selectedAnnotationIndex = currentAnnotations.length - 1;
    renderMeta();
    renderInspector();
    renderAnnotationList();
  }
  dragStart = null;
  dragCurrent = null;
  drawCanvas();
}

function commitPolygon() {
  if (currentPolygonPoints.length < 3) {
    return;
  }
  const normalized = currentPolygonPoints.map((point) => [
    Math.round((point.x / currentScale) * 1000) / 1000,
    Math.round((point.y / currentScale) * 1000) / 1000,
  ]);
  const xs = normalized.map((point) => point[0]);
  const ys = normalized.map((point) => point[1]);
  currentAnnotations.push({
    id: `${Date.now()}-${currentAnnotations.length + 1}`,
    class_name: classSelect.value,
    bbox: [Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)],
    points: normalized,
  });
  currentPolygonPoints = [];
  selectedAnnotationIndex = currentAnnotations.length - 1;
  renderMeta();
  renderInspector();
  renderAnnotationList();
  drawCanvas();
}

async function saveAnnotations() {
  if (!currentItem) {
    setStatus("Load an image first.");
    return;
  }
  const payload = {
    dataset: currentItem.dataset,
    split: currentItem.split,
    image_id: currentItem.image_id,
    file_name: currentItem.file_name,
    width: currentItem.width,
    height: currentItem.height,
    annotations: currentAnnotations,
  };
  const result = await fetchJson("/api/annotation/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  currentItem.annotation_source = "override";
  renderMeta();
  if (reviewQueuePosition >= 0 && reviewQueue[reviewQueuePosition]) {
    reviewQueue[reviewQueuePosition].override_exists = true;
    reviewQueue[reviewQueuePosition].annotation_source = "override";
    reviewQueue[reviewQueuePosition].annotation_count = currentAnnotations.length;
    reviewQueue[reviewQueuePosition].class_names = [...new Set(currentAnnotations.map((item) => item.class_name))].sort();
    renderQueueOptions();
  }
  setStatus(`Saved annotations to ${result.annotation_path}`);
}

async function exportDataset(exportMode = "full") {
  const formData = new FormData();
  formData.set("dataset", currentDatasetKey());
  formData.set("export_mode", exportMode);
  const result = await fetchJson("/api/annotation/export", { method: "POST", body: formData });
  const label = exportMode === "validated_bubble_only" ? "validated bubble-only" : "full";
  setStatus(`Exported ${label} retraining dataset to ${result.output_dir}`);
}

function updateNavButtons() {
  if (currentViewMode() === "review") {
    prevImageButton.disabled = reviewQueuePosition <= 0;
    nextImageButton.disabled = reviewQueuePosition >= Math.max(0, reviewQueue.length - 1);
    return;
  }
  prevImageButton.disabled = currentIndex <= 0;
  nextImageButton.disabled = currentIndex >= Math.max(0, datasetSummary.total - 1);
}

function syncUrlState() {
  const params = new URLSearchParams(window.location.search);
  if (datasetSelect.value) {
    params.set("dataset", datasetSelect.value);
  }
  params.set("view", currentViewMode());
  params.set("split", splitSelect.value);
  params.set("index", String(currentIndex));
  if (reviewQueuePosition >= 0) {
    params.set("queueIndex", String(reviewQueuePosition));
  } else {
    params.delete("queueIndex");
  }
  window.history.replaceState({}, "", `${window.location.pathname}?${params.toString()}`);
}

async function loadQueuePosition(position) {
  if (position < 0 || position >= reviewQueue.length) {
    return;
  }
  reviewQueuePosition = position;
  const item = reviewQueue[position];
  queueItemSelect.value = String(position);
  await loadItem(item.index, item.split);
}

async function bootstrapFromMode() {
  if (currentViewMode() === "review") {
    await loadReviewQueue();
    updateModeControls();
    if (!reviewQueue.length) {
      reviewQueuePosition = -1;
      updateNavButtons();
      return;
    }
    const requestedQueueIndex = Number.parseInt(currentQuery().get("queueIndex") || "", 10);
    const nextQueueIndex = Number.isFinite(requestedQueueIndex)
      ? Math.max(0, Math.min(requestedQueueIndex, reviewQueue.length - 1))
      : 0;
    await loadQueuePosition(nextQueueIndex);
    return;
  }

  reviewQueuePosition = -1;
  renderQueueOptions();
  updateModeControls();
  const requestedIndex = Number.parseInt(currentQuery().get("index") || "", 10);
  const nextIndex = Number.isFinite(requestedIndex) ? requestedIndex : 0;
  await loadItem(nextIndex);
}

canvas.addEventListener("pointerdown", (event) => {
  const point = pointerPosition(event);
  if ((shapeModeSelect?.value || "rect") === "polygon") {
    const hitIndex = findAnnotationAtPoint(point);
    if (hitIndex !== null) {
      selectAnnotation(hitIndex);
      return;
    }
    currentPolygonPoints = [...currentPolygonPoints, point];
    drawCanvas();
    return;
  }
  const hitIndex = findAnnotationAtPoint(point);
  if (hitIndex !== null) {
    dragStart = null;
    dragCurrent = null;
    selectAnnotation(hitIndex);
    return;
  }
  dragStart = point;
  dragCurrent = dragStart;
  drawCanvas();
});

canvas.addEventListener("pointermove", (event) => {
  if ((shapeModeSelect?.value || "rect") === "polygon") {
    return;
  }
  if (!dragStart) {
    return;
  }
  dragCurrent = pointerPosition(event);
  drawCanvas();
});

canvas.addEventListener("pointerup", () => {
  if ((shapeModeSelect?.value || "rect") === "polygon") {
    return;
  }
  commitDragBox();
});
canvas.addEventListener("pointerleave", () => {
  if ((shapeModeSelect?.value || "rect") === "polygon") {
    return;
  }
  if (dragStart) {
    commitDragBox();
  }
});

datasetSelect.addEventListener("change", async () => {
  await bootstrap();
});

splitSelect.addEventListener("change", async () => {
  if (currentViewMode() === "review") {
    return;
  }
  await bootstrap();
});

prevImageButton.addEventListener("click", async () => {
  if (currentViewMode() === "review") {
    await loadQueuePosition(reviewQueuePosition - 1);
    return;
  }
  await loadItem(currentIndex - 1);
});

nextImageButton.addEventListener("click", async () => {
  if (currentViewMode() === "review") {
    await loadQueuePosition(reviewQueuePosition + 1);
    return;
  }
  await loadItem(currentIndex + 1);
});

viewModeSelect.addEventListener("change", async () => {
  updateModeControls();
  await bootstrap();
});

shapeModeSelect?.addEventListener("change", () => {
  dragStart = null;
  dragCurrent = null;
  currentPolygonPoints = [];
  const mode = shapeModeSelect.value === "polygon" ? "Polygon" : "Rectangle";
  drawCanvas();
  setStatus(`Annotation shape mode set to ${mode}.`);
});

undoAnnotationPointButton?.addEventListener("click", () => {
  if (!currentPolygonPoints.length) {
    return;
  }
  currentPolygonPoints = currentPolygonPoints.slice(0, -1);
  drawCanvas();
  setStatus("Removed the last annotation point.");
});

finishAnnotationShapeButton?.addEventListener("click", () => {
  if ((shapeModeSelect?.value || "rect") !== "polygon") {
    setStatus("Switch to Polygon mode to finish a multi-point annotation shape.");
    return;
  }
  if (currentPolygonPoints.length < 3) {
    setStatus("Add at least 3 points before finishing the polygon.");
    return;
  }
  commitPolygon();
  setStatus("Polygon annotation added.");
});

queueItemSelect.addEventListener("change", async () => {
  if (currentViewMode() !== "review") {
    return;
  }
  const queueIndex = Number.parseInt(queueItemSelect.value || "", 10);
  if (!Number.isFinite(queueIndex)) {
    return;
  }
  await loadQueuePosition(queueIndex);
});

saveAnnotationsButton.addEventListener("click", saveAnnotations);
exportDatasetButton.addEventListener("click", async () => {
  await exportDataset("full");
});
exportValidatedBubbleButton.addEventListener("click", async () => {
  await exportDataset("validated_bubble_only");
});

window.addEventListener("resize", () => {
  if (!currentItem) {
    return;
  }
  loadImage(currentItem.image_url)
    .then(drawCanvas)
    .catch(() => {});
});

async function bootstrap() {
  try {
    setStatus("Loading dataset...");
    const query = currentQuery();
    if (!availableDatasets.length) {
      const datasetPayload = await fetchJson("/api/annotation/datasets");
      availableDatasets = datasetPayload.datasets || [];
      if (!datasetSelect.value && datasetPayload.default_dataset) {
        datasetSelect.innerHTML = "";
      }
      renderDatasetOptions();
      const requestedDataset = query.get("dataset");
      if (requestedDataset) {
        datasetSelect.value = requestedDataset;
      } else if (availableDatasets.some((item) => item.key === preferredValidationDataset)) {
        datasetSelect.value = preferredValidationDataset;
      } else if (!datasetSelect.value && datasetPayload.default_dataset) {
        datasetSelect.value = datasetPayload.default_dataset;
      }
      if (!datasetSelect.value && availableDatasets.length) {
        datasetSelect.value = availableDatasets[0].key;
      }
    }
    if (!hasAppliedUrlState) {
      const requestedSplit = query.get("split");
      if (requestedSplit && ["train", "valid", "test"].includes(requestedSplit)) {
        splitSelect.value = requestedSplit;
      }
      const requestedView = query.get("view");
      if (requestedView && ["all", "review"].includes(requestedView)) {
        viewModeSelect.value = requestedView;
      }
      hasAppliedUrlState = true;
    }
    await bootstrapFromMode();
  } catch (error) {
    setStatus(error.message || "Failed to load annotation workspace.");
  }
}

bootstrap();
