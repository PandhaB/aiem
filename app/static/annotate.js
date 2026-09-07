(() => {
  const projectId = window.PROJECT_ID;
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const imageList = document.getElementById("image-list");
  const classSelect = document.getElementById("class-select");
  const saveStatus = document.getElementById("save-status");

  let coco = null;
  let currentImage = null;
  let bitmap = null;
  let scale = 1;
  let offsetX = 0;
  let offsetY = 0;
  let draft = [];
  let selectedAnnId = null;

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

  function redraw() {
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (bitmap) {
      ctx.drawImage(bitmap, offsetX, offsetY, bitmap.width * scale, bitmap.height * scale);
    }
    if (!currentImage) return;
    for (const annotation of annsForImage(currentImage.id)) {
      drawPolygon(
        annotation.segmentation[0],
        colorForCategory(annotation.category_id),
        annotation.id === selectedAnnId
      );
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
  }

  function closeDraft() {
    if (draft.length < 6) {
      draft = [];
      redraw();
      return;
    }
    const nextId = coco.annotations.reduce((max, item) => Math.max(max, item.id), 0) + 1;
    coco.annotations.push({
      id: nextId,
      image_id: currentImage.id,
      category_id: currentClass().id,
      segmentation: [draft.slice()],
      iscrowd: 0,
    });
    selectedAnnId = nextId;
    draft = [];
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

  async function loadImage(info) {
    currentImage = info;
    selectedAnnId = null;
    draft = [];
    const image = new Image();
    image.src = `/api/projects/${projectId}/images/${encodeURIComponent(info.file_name)}`;
    await image.decode();
    bitmap = image;
    fitImage();
    redraw();
    [...imageList.querySelectorAll("button")].forEach((button) => {
      button.classList.toggle("active", button.dataset.name === info.file_name);
    });
  }

  async function loadCoco() {
    coco = await (await fetch(`/api/projects/${projectId}/annotations`)).json();
    imageList.innerHTML = "";
    if (!coco.images.length) {
      imageList.innerHTML = "<li class='muted'>No images yet. Import some from the project page.</li>";
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
  }

  canvas.addEventListener("click", (event) => {
    if (!currentImage) return;
    const point = toImage(event);
    if (point.x < 0 || point.y < 0 || point.x > currentImage.width || point.y > currentImage.height) {
      return;
    }
    if (!draft.length) {
      const hit = hitTest(point.x, point.y);
      if (hit !== null) {
        selectedAnnId = hit;
        redraw();
        return;
      }
    }
    draft.push(point.x, point.y);
    selectedAnnId = null;
    redraw();
  });

  canvas.addEventListener("dblclick", (event) => {
    event.preventDefault();
    closeDraft();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      closeDraft();
    } else if (event.key === "Escape") {
      draft = [];
      selectedAnnId = null;
      redraw();
    } else if (event.key === "Delete" || event.key === "Backspace") {
      if (selectedAnnId !== null && !draft.length) {
        coco.annotations = coco.annotations.filter((item) => item.id !== selectedAnnId);
        selectedAnnId = null;
        redraw();
      }
    }
  });

  document.getElementById("save-btn").addEventListener("click", async () => {
    saveStatus.textContent = "Saving…";
    const response = await fetch(`/api/projects/${projectId}/annotations`, {
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
    saveStatus.textContent = "Saved.";
  });

  loadCoco();
})();
