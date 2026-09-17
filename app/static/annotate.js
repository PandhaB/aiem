// Polygon editor on a canvas. Vertices are stored in native image pixels (COCO).
// Annotate mode writes project annotations.json. Refine mode writes a run's predictions.json.
(() => {
  function start() {
    const canvas = document.getElementById("canvas");
    if (!canvas) return;
    const editor = {
      mode: canvas.dataset.mode || "annotate",
      projectId: canvas.dataset.projectId || window.PROJECT_ID,
      runId: canvas.dataset.runId || null,
      fileName: canvas.dataset.fileName || "predictions.json",
    };
    Object.assign(editor, window.POLYGON_EDITOR || {});
    if (!editor.projectId) return;
    boot(canvas, editor);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }

  function boot(canvas, editor) {
  const projectId = editor.projectId;
  const ctx = canvas.getContext("2d");
  const imageList = document.getElementById("image-list");
  const classSelect = document.getElementById("class-select");
  const saveStatus = document.getElementById("save-status");
  const saveBtn = document.getElementById("save-btn");
  const saveCopyBtn = document.getElementById("save-copy-btn");
  const saveOverwriteBtn = document.getElementById("save-overwrite-btn");
  const deleteBtn = document.getElementById("delete-instance");
  const smoothBtn = document.getElementById("smooth-btn");
  const smoothUndoBtn = document.getElementById("smooth-undo-btn");
  const imagePrevBtn = document.getElementById("image-prev");
  const imageNextBtn = document.getElementById("image-next");
  const zoomCanvas = document.getElementById("zoom-canvas");
  const HANDLE_PX = 8;
  const EDGE_PX = 7;
  const ZOOM_MAG = 4;
  const SHIFT_GAIN = 0.25;

  let coco = null;
  let currentImage = null;
  let smoothUndo = null;
  let bitmap = null;
  let scale = 1;
  let offsetX = 0;
  let offsetY = 0;
  let draft = [];
  let selectedAnnId = null;
  let selectedVertex = null;
  let dragVertex = null;
  let suppressClick = false;
  let dirty = false;
  let imageLoadSeq = 0;
  let hoverPoint = null;
  let lastRaw = null;
  let lastClient = null;
  let shiftPrecision = false;

  function setDirty(value) {
    dirty = value;
    window.polygonEditorDirty = value;
  }

  function isRefine() {
    return editor.mode === "refine" && editor.runId;
  }

  function cloneCoco(data) {
    return JSON.parse(JSON.stringify(data));
  }

  function setSmoothUndo(snapshot) {
    smoothUndo = snapshot;
    if (smoothUndoBtn) smoothUndoBtn.disabled = !smoothUndo;
  }

  function currentImageIndex() {
    if (!coco || !currentImage) return -1;
    return coco.images.findIndex((item) => item.id === currentImage.id);
  }

  function updateImageCaption() {
    const node = document.getElementById("image-caption");
    if (!node || !currentImage || !coco) return;
    const index = currentImageIndex();
    node.textContent = `${currentImage.file_name} · ${index + 1} / ${coco.images.length}`;
  }

  function stepImage(delta) {
    if (!coco || coco.images.length < 2 || !currentImage) return;
    const index = currentImageIndex();
    if (index < 0) return;
    const next = (index + delta + coco.images.length) % coco.images.length;
    loadImage(coco.images[next]);
  }

  function cocoUrl(fileName) {
    if (isRefine()) {
      const name = fileName || editor.fileName || "predictions.json";
      return `/api/projects/${projectId}/runs/${encodeURIComponent(editor.runId)}/predictions?file=${encodeURIComponent(name)}`;
    }
    return `/api/projects/${projectId}/annotations`;
  }

  function imageUrl(fileName) {
    if (isRefine()) {
      return `/api/projects/${projectId}/runs/${encodeURIComponent(editor.runId)}/inputs/${encodeURIComponent(fileName)}`;
    }
    return `/api/projects/${projectId}/images/${encodeURIComponent(fileName)}`;
  }

  function currentClass() {
    const option = classSelect.selectedOptions[0];
    return {
      id: Number(classSelect.value),
      color: option ? option.dataset.color : "#e63946",
    };
  }

  function annsForImage(imageId) {
    return coco.annotations.filter((item) => item.image_id === imageId);
  }

  function selectedAnnotation() {
    if (selectedAnnId === null || !coco) return null;
    return coco.annotations.find((item) => item.id === selectedAnnId) || null;
  }

  function fitImage() {
    if (!bitmap) return;
    const pad = 8;
    const availW = canvas.width - pad * 2;
    const availH = canvas.height - pad * 2;
    scale = Math.min(availW / bitmap.width, availH / bitmap.height);
    offsetX = (canvas.width - bitmap.width * scale) / 2;
    offsetY = (canvas.height - bitmap.height * scale) / 2;
  }

  function toImage(event) {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (canvas.width / rect.width);
    const y = (event.clientY - rect.top) * (canvas.height / rect.height);
    return {
      x: (x - offsetX) / scale,
      y: (y - offsetY) / scale,
    };
  }

  function toCanvas(x, y) {
    return [offsetX + x * scale, offsetY + y * scale];
  }

  function isFormField(event) {
    return event.target && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName);
  }

  function pointerOverCanvas(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    return clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom;
  }

  function handlePointerMove(event) {
    if (!currentImage) return;
    lastClient = { x: event.clientX, y: event.clientY };
    const raw = toImage(event);
    logicalPointFromEvent(event);
    if (dragVertex) {
      applyHoverToDrag();
      canvas.style.cursor = "grabbing";
      redraw();
      return;
    }
    if (pointerOverCanvas(event.clientX, event.clientY) && !draft.length) {
      canvas.style.cursor = cursorFor(raw);
    }
    if (shiftPrecision) redraw();
    else updateZoom();
  }

  function logicalPointFromEvent(event) {
    const raw = toImage(event);
    if (event.shiftKey && lastRaw && hoverPoint) {
      hoverPoint = {
        x: hoverPoint.x + (raw.x - lastRaw.x) * SHIFT_GAIN,
        y: hoverPoint.y + (raw.y - lastRaw.y) * SHIFT_GAIN,
      };
    } else {
      hoverPoint = { x: raw.x, y: raw.y };
    }
    lastRaw = raw;
    shiftPrecision = !!event.shiftKey;
    return hoverPoint;
  }

  function applyHoverToDrag() {
    if (!dragVertex || !hoverPoint || !coco) return;
    const annotation = coco.annotations.find((item) => item.id === dragVertex.annId);
    if (!annotation) return;
    const clamped = clampToImage(hoverPoint.x, hoverPoint.y);
    annotation.segmentation[0][dragVertex.index] = clamped.x;
    annotation.segmentation[0][dragVertex.index + 1] = clamped.y;
    setDirty(true);
  }

  function drawPrecisionCursor() {
    if (!hoverPoint || !shiftPrecision) return;
    const [cx, cy] = toCanvas(hoverPoint.x, hoverPoint.y);
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy);
    ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8);
    ctx.lineTo(cx, cy + 8);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(178, 58, 47, 0.95)";
    ctx.stroke();
    ctx.restore();
  }

  function sizeZoomCanvas() {
    if (!zoomCanvas) return;
    const rect = zoomCanvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    if (zoomCanvas.width !== width || zoomCanvas.height !== height) {
      zoomCanvas.width = width;
      zoomCanvas.height = height;
    }
  }

  function updateZoom() {
    if (!zoomCanvas) return;
    sizeZoomCanvas();
    const zctx = zoomCanvas.getContext("2d");
    zctx.setTransform(1, 0, 0, 1, 0, 0);
    zctx.fillStyle = "#111";
    zctx.fillRect(0, 0, zoomCanvas.width, zoomCanvas.height);
    if (!bitmap || !hoverPoint) return;
    const [cx, cy] = toCanvas(hoverPoint.x, hoverPoint.y);
    zctx.imageSmoothingEnabled = false;
    zctx.setTransform(
      ZOOM_MAG,
      0,
      0,
      ZOOM_MAG,
      zoomCanvas.width / 2 - cx * ZOOM_MAG,
      zoomCanvas.height / 2 - cy * ZOOM_MAG
    );
    zctx.drawImage(canvas, 0, 0);
    zctx.setTransform(1, 0, 0, 1, 0, 0);
    zctx.strokeStyle = "rgba(255,255,255,0.75)";
    zctx.lineWidth = 1;
    zctx.beginPath();
    zctx.moveTo(zoomCanvas.width / 2, 0);
    zctx.lineTo(zoomCanvas.width / 2, zoomCanvas.height);
    zctx.moveTo(0, zoomCanvas.height / 2);
    zctx.lineTo(zoomCanvas.width, zoomCanvas.height / 2);
    zctx.stroke();
  }

  function clampToImage(x, y) {
    if (!currentImage) return { x, y };
    return {
      x: Math.max(0, Math.min(currentImage.width, x)),
      y: Math.max(0, Math.min(currentImage.height, y)),
    };
  }

  function hexToRgba(hex, alpha) {
    const text = hex.replace("#", "");
    const r = parseInt(text.slice(0, 2), 16);
    const g = parseInt(text.slice(2, 4), 16);
    const b = parseInt(text.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function colorForCategory(categoryId) {
    const option = [...classSelect.options].find((item) => Number(item.value) === categoryId);
    return option ? option.dataset.color : "#e63946";
  }

  function drawPolygon(flat, color, selected) {
    if (flat.length < 6) return;
    ctx.beginPath();
    const start = toCanvas(flat[0], flat[1]);
    ctx.moveTo(start[0], start[1]);
    for (let i = 2; i < flat.length; i += 2) {
      const point = toCanvas(flat[i], flat[i + 1]);
      ctx.lineTo(point[0], point[1]);
    }
    ctx.closePath();
    ctx.fillStyle = hexToRgba(color, selected ? 0.45 : 0.28);
    ctx.strokeStyle = color;
    ctx.lineWidth = selected ? 3 : 2;
    ctx.fill();
    ctx.stroke();
  }

  function drawHandles(flat, color) {
    for (let i = 0; i < flat.length; i += 2) {
      const point = toCanvas(flat[i], flat[i + 1]);
      ctx.beginPath();
      ctx.arc(point[0], point[1], i === selectedVertex ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  function redraw() {
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (bitmap) {
      ctx.drawImage(bitmap, offsetX, offsetY, bitmap.width * scale, bitmap.height * scale);
    }
    if (!currentImage) {
      updateZoom();
      return;
    }
    for (const annotation of annsForImage(currentImage.id)) {
      const selected = annotation.id === selectedAnnId;
      const color = colorForCategory(annotation.category_id);
      drawPolygon(annotation.segmentation[0], color, selected);
      if (selected) drawHandles(annotation.segmentation[0], color);
    }
    if (draft.length) {
      const color = currentClass().color;
      ctx.beginPath();
      const start = toCanvas(draft[0], draft[1]);
      ctx.moveTo(start[0], start[1]);
      for (let i = 2; i < draft.length; i += 2) {
        const point = toCanvas(draft[i], draft[i + 1]);
        ctx.lineTo(point[0], point[1]);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
      for (let i = 0; i < draft.length; i += 2) {
        const point = toCanvas(draft[i], draft[i + 1]);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(point[0], point[1], 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    updateZoom();
    drawPrecisionCursor();
  }

  function closeDraft() {
    if (draft.length < 6) {
      draft = [];
      redraw();
      return;
    }
    const nextId = coco.annotations.reduce((max, item) => Math.max(max, item.id), 0) + 1;
    const annotation = {
      id: nextId,
      image_id: currentImage.id,
      category_id: currentClass().id,
      segmentation: [draft.slice()],
      iscrowd: 0,
    };
    if (isRefine()) annotation.score = 1.0;
    coco.annotations.push(annotation);
    selectedAnnId = nextId;
    selectedVertex = null;
    draft = [];
    setDirty(true);
    redraw();
  }

  function hitTest(x, y) {
    const annotations = annsForImage(currentImage.id).slice().reverse();
    for (const annotation of annotations) {
      const flat = annotation.segmentation[0];
      let inside = false;
      for (let i = 0, j = flat.length - 2; i < flat.length; j = i, i += 2) {
        const xi = flat[i], yi = flat[i + 1];
        const xj = flat[j], yj = flat[j + 1];
        const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
        if (intersect) inside = !inside;
      }
      if (inside) return annotation.id;
    }
    return null;
  }

  function hitVertex(annotation, imagePoint) {
    if (!annotation) return null;
    const [mx, my] = toCanvas(imagePoint.x, imagePoint.y);
    const flat = annotation.segmentation[0];
    for (let i = 0; i < flat.length; i += 2) {
      const point = toCanvas(flat[i], flat[i + 1]);
      if (Math.hypot(mx - point[0], my - point[1]) <= HANDLE_PX) return i;
    }
    return null;
  }

  function hitEdge(annotation, imagePoint) {
    if (!annotation) return null;
    const [mx, my] = toCanvas(imagePoint.x, imagePoint.y);
    const flat = annotation.segmentation[0];
    let best = null;
    const count = flat.length;
    for (let i = 0; i < count; i += 2) {
      const j = (i + 2) % count;
      const a = toCanvas(flat[i], flat[i + 1]);
      const b = toCanvas(flat[j], flat[j + 1]);
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const len2 = dx * dx + dy * dy;
      let t = len2 === 0 ? 0 : ((mx - a[0]) * dx + (my - a[1]) * dy) / len2;
      t = Math.max(0, Math.min(1, t));
      const qx = a[0] + t * dx;
      const qy = a[1] + t * dy;
      const dist = Math.hypot(mx - qx, my - qy);
      if (dist <= EDGE_PX && t > 0.08 && t < 0.92) {
        const insertAt = j === 0 ? count : j;
        if (!best || dist < best.dist) best = { index: insertAt, dist };
      }
    }
    return best;
  }

  function cursorFor(imagePoint) {
    if (draft.length) return "crosshair";
    const selected = selectedAnnotation();
    if (selected && hitVertex(selected, imagePoint) !== null) return "grab";
    if (selected && hitEdge(selected, imagePoint)) return "copy";
    if (hitTest(imagePoint.x, imagePoint.y) !== null) return "pointer";
    return "crosshair";
  }

  function deleteSelected() {
    if (selectedAnnId === null || draft.length) return false;
    const annotation = selectedAnnotation();
    if (!annotation) return false;
    const flat = annotation.segmentation[0];
    if (selectedVertex !== null && flat.length > 6) {
      flat.splice(selectedVertex, 2);
      if (selectedVertex >= flat.length) selectedVertex = 0;
      setDirty(true);
      redraw();
      return true;
    }
    coco.annotations = coco.annotations.filter((item) => item.id !== selectedAnnId);
    selectedAnnId = null;
    selectedVertex = null;
    setDirty(true);
    redraw();
    return true;
  }

  async function loadImage(info) {
    const seq = ++imageLoadSeq;
    currentImage = info;
    selectedAnnId = null;
    selectedVertex = null;
    draft = [];
    updateImageCaption();
    const image = new Image();
    image.src = imageUrl(info.file_name);
    await image.decode();
    if (seq !== imageLoadSeq) return;
    bitmap = image;
    fitImage();
    redraw();
    if (!imageList) return;
    [...imageList.querySelectorAll("button")].forEach((button) => {
      button.classList.toggle("active", button.dataset.name === info.file_name);
    });
  }

  async function loadCoco() {
    if (editor.mode === "refine" && !editor.runId) return;
    const response = await fetch(cocoUrl());
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      if (saveStatus) saveStatus.textContent = payload.detail || "Could not load polygons.";
      return;
    }
    coco = await response.json();
    setSmoothUndo(null);
    if (!imageList) return;
    imageList.innerHTML = "";
    if (!coco.images.length) {
      imageList.innerHTML = "<li class='muted'>No images in this set.</li>";
      return;
    }
    for (const info of coco.images) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.name = info.file_name;
      button.textContent = info.file_name;
      button.addEventListener("click", () => loadImage(info));
      item.appendChild(button);
      imageList.appendChild(item);
    }
    await loadImage(coco.images[0]);
    setDirty(false);
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (!currentImage || event.button !== 0) return;
    if (draft.length) return;
    const point = toImage(event);
    const selected = selectedAnnotation();
    if (selected) {
      const vertex = hitVertex(selected, point);
      if (vertex !== null) {
        selectedVertex = vertex;
        dragVertex = { annId: selected.id, index: vertex };
        suppressClick = true;
        canvas.setPointerCapture(event.pointerId);
        canvas.style.cursor = "grabbing";
        redraw();
        return;
      }
      const edge = hitEdge(selected, point);
      if (edge) {
        const flat = selected.segmentation[0];
        const clamped = clampToImage(point.x, point.y);
        flat.splice(edge.index, 0, clamped.x, clamped.y);
        selectedVertex = edge.index;
        dragVertex = { annId: selected.id, index: edge.index };
        suppressClick = true;
        setDirty(true);
        canvas.setPointerCapture(event.pointerId);
        canvas.style.cursor = "grabbing";
        redraw();
        return;
      }
    }
    const hit = hitTest(point.x, point.y);
    if (hit !== null) {
      selectedAnnId = hit;
      selectedVertex = null;
      suppressClick = true;
      redraw();
    }
  });

  canvas.addEventListener("pointermove", handlePointerMove);

  document.addEventListener("pointermove", (event) => {
    if (event.target === canvas) return;
    if (!event.shiftKey || !hoverPoint || !lastRaw) return;
    handlePointerMove(event);
  });

  function endDrag(event) {
    if (!dragVertex) return;
    dragVertex = null;
    if (event && canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    canvas.style.cursor = "grab";
  }

  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", (event) => {
    if (dragVertex) return;
    if (event.shiftKey || shiftPrecision) return;
    hoverPoint = null;
    lastRaw = null;
    lastClient = null;
    shiftPrecision = false;
    redraw();
  });

  canvas.addEventListener("click", (event) => {
    if (!currentImage) return;
    if (suppressClick) {
      suppressClick = false;
      return;
    }
    const point = toImage(event);
    if (point.x < 0 || point.y < 0 || point.x > currentImage.width || point.y > currentImage.height) {
      return;
    }
    if (!draft.length) {
      const hit = hitTest(point.x, point.y);
      if (hit !== null) {
        selectedAnnId = hit;
        selectedVertex = null;
        redraw();
        return;
      }
    }
    draft.push(point.x, point.y);
    selectedAnnId = null;
    selectedVertex = null;
    redraw();
  });

  canvas.addEventListener("dblclick", (event) => {
    event.preventDefault();
    closeDraft();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Shift") {
      if (isFormField(event) || event.repeat) return;
      shiftPrecision = true;
      if (hoverPoint) redraw();
      return;
    }
    if (isFormField(event)) return;
    if (event.key === "Enter") {
      closeDraft();
    } else if (event.key === "Escape") {
      draft = [];
      selectedAnnId = null;
      selectedVertex = null;
      dragVertex = null;
      redraw();
    } else if (event.key === "Delete" || event.key === "Backspace") {
      if (deleteSelected()) event.preventDefault();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      stepImage(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      stepImage(1);
    }
  });

  document.addEventListener("keyup", (event) => {
    if (event.key !== "Shift") return;
    if (isFormField(event)) return;
    shiftPrecision = false;
    if (lastRaw) {
      hoverPoint = { x: lastRaw.x, y: lastRaw.y };
      applyHoverToDrag();
    }
    const stillOver = lastClient && pointerOverCanvas(lastClient.x, lastClient.y);
    if (!stillOver && !dragVertex) {
      hoverPoint = null;
      lastRaw = null;
      lastClient = null;
    }
    redraw();
  });

  function pointLineDistance(point, start, end) {
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const length = Math.hypot(dx, dy);
    if (length === 0) return Math.hypot(point[0] - start[0], point[1] - start[1]);
    return Math.abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / length;
  }

  function rdp(points, epsilon) {
    if (points.length < 3) return points.slice();
    const keep = new Array(points.length).fill(false);
    keep[0] = true;
    keep[points.length - 1] = true;
    const stack = [[0, points.length - 1]];
    while (stack.length) {
      const [start, end] = stack.pop();
      let maxDist = -1;
      let farthest = start;
      for (let i = start + 1; i < end; i += 1) {
        const dist = pointLineDistance(points[i], points[start], points[end]);
        if (dist > maxDist) {
          maxDist = dist;
          farthest = i;
        }
      }
      if (maxDist > epsilon && farthest !== start) {
        keep[farthest] = true;
        stack.push([start, farthest]);
        stack.push([farthest, end]);
      }
    }
    return points.filter((_, index) => keep[index]);
  }

  function simplifyPolygon(flat, tolerance) {
    const points = [];
    for (let i = 0; i < flat.length - 1; i += 2) {
      points.push([flat[i], flat[i + 1]]);
    }
    if (points.length >= 2 && points[0][0] === points[points.length - 1][0] && points[0][1] === points[points.length - 1][1]) {
      points.pop();
    }
    if (points.length <= 3) return flat.slice();
    const simplified = rdp(points.concat([points[0]]), tolerance);
    if (simplified.length >= 2 && simplified[0][0] === simplified[simplified.length - 1][0] && simplified[0][1] === simplified[simplified.length - 1][1]) {
      simplified.pop();
    }
    if (simplified.length < 3) return flat.slice();
    const out = [];
    simplified.forEach((point) => {
      out.push(point[0], point[1]);
    });
    return out;
  }

  function smoothAll(tolerance) {
    if (!coco) return 0;
    let dropped = 0;
    for (const annotation of coco.annotations) {
      const flat = annotation.segmentation[0];
      const before = flat.length / 2;
      const next = simplifyPolygon(flat, tolerance);
      annotation.segmentation[0] = next;
      dropped += Math.max(0, before - next.length / 2);
    }
    return dropped;
  }

  async function savePredictions(fileName, rebuildVisuals) {
    if (!coco) return;
    saveStatus.textContent = "Saving…";
    const params = new URLSearchParams();
    params.set("file", fileName);
    params.set("rebuild_visuals", rebuildVisuals ? "true" : "false");
    const response = await fetch(`${cocoUrl(fileName).split("?")[0]}?${params.toString()}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(coco),
    });
    if (!response.ok) {
      const payload = await response.json();
      saveStatus.textContent = payload.detail || "Save failed.";
      return false;
    }
    coco = await response.json();
    editor.fileName = fileName;
    window.POLYGON_EDITOR.fileName = fileName;
    canvas.dataset.fileName = fileName;
    setDirty(false);
    setSmoothUndo(null);
    if (selectedAnnId !== null && !coco.annotations.some((item) => item.id === selectedAnnId)) {
      selectedAnnId = null;
      selectedVertex = null;
    }
    redraw();
    return true;
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      saveStatus.textContent = "Saving…";
      const response = await fetch(cocoUrl(), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(coco),
      });
      if (!response.ok) {
        const payload = await response.json();
        saveStatus.textContent = payload.detail || "Save failed.";
        return;
      }
      coco = await response.json();
      setDirty(false);
      setSmoothUndo(null);
      saveStatus.textContent = "Saved.";
      if (selectedAnnId !== null && !coco.annotations.some((item) => item.id === selectedAnnId)) {
        selectedAnnId = null;
        selectedVertex = null;
      }
      redraw();
    });
  }

  if (saveCopyBtn) {
    saveCopyBtn.addEventListener("click", async () => {
      const input = document.getElementById("save-filename");
      const name = (input && input.value.trim()) || "predictions_refined.json";
      const ok = await savePredictions(name, name === "predictions.json");
      if (ok) {
        saveStatus.textContent = `Saved copy as ${name}. The original predictions.json is unchanged.`;
        const select = document.getElementById("file-select");
        if (select && ![...select.options].some((option) => option.value === name)) {
          const option = document.createElement("option");
          option.value = name;
          option.textContent = name;
          select.appendChild(option);
        }
        if (select) select.value = name;
      }
    });
  }

  if (saveOverwriteBtn) {
    saveOverwriteBtn.addEventListener("click", async () => {
      const confirmed = window.confirm(
        "Replace this run's predictions.json and rebuild overlays and masks? The raw model output will be overwritten."
      );
      if (!confirmed) return;
      const ok = await savePredictions("predictions.json", true);
      if (ok) saveStatus.textContent = "Overwrote predictions.json and rebuilt overlays and masks.";
    });
  }

  if (smoothBtn) {
    smoothBtn.addEventListener("click", () => {
      const raw = document.getElementById("smooth-tolerance");
      const tolerance = raw ? Number(raw.value) : 2;
      if (!Number.isFinite(tolerance) || tolerance <= 0) {
        saveStatus.textContent = "Smooth tolerance must be greater than 0.";
        return;
      }
      const snapshot = { coco: cloneCoco(coco), dirty };
      const dropped = smoothAll(tolerance);
      selectedVertex = null;
      if (dropped) {
        setSmoothUndo(snapshot);
        setDirty(true);
      }
      redraw();
      saveStatus.textContent = dropped
        ? `Removed ${dropped} vertices. Undo to restore, or save a copy / overwrite the run to keep this.`
        : "No vertices to drop at this tolerance.";
    });
  }

  if (smoothUndoBtn) {
    smoothUndoBtn.addEventListener("click", () => {
      if (!smoothUndo) return;
      coco = cloneCoco(smoothUndo.coco);
      setDirty(smoothUndo.dirty);
      setSmoothUndo(null);
      selectedVertex = null;
      redraw();
      if (saveStatus) {
        saveStatus.textContent = "Restored contours from before the last Smooth. You can try another tolerance.";
      }
    });
  }

  if (imagePrevBtn) imagePrevBtn.addEventListener("click", () => stepImage(-1));
  if (imageNextBtn) imageNextBtn.addEventListener("click", () => stepImage(1));

  if (zoomCanvas && window.ResizeObserver) {
    new ResizeObserver(() => updateZoom()).observe(zoomCanvas);
  }

  if (deleteBtn) {
    deleteBtn.addEventListener("click", () => {
      if (!deleteSelected() && saveStatus) {
        saveStatus.textContent = "Select a mask first.";
      }
    });
  }

  loadCoco().catch((err) => {
    if (saveStatus) saveStatus.textContent = err.message || "Could not load polygons.";
  });
  }
})();
