window.pollJob = async function pollJob(jobId, ui) {
  ui.box.hidden = false;
  const tick = async () => {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    ui.status.textContent = job.status;
    ui.message.textContent = job.message || "";
    ui.progress.value = job.progress || 0;
    if (ui.log) window.renderJobLog(ui.log, job.log || []);
    if (ui.onTick) ui.onTick(job);
    if (job.status === "failed") {
      ui.error.hidden = false;
      ui.error.textContent = job.error || "Job failed.";
      if (ui.onDone) ui.onDone(job);
      return;
    }
    if (job.status === "completed" || job.status === "stopped") {
      if (ui.onComplete) ui.onComplete(job);
      if (ui.onDone) ui.onDone(job);
      return;
    }
    window.setTimeout(tick, 400);
  };
  tick();
};

window.renderJobLog = function renderJobLog(node, lines) {
  const text = (lines || []).join("\n");
  const stick =
    node.scrollHeight - node.scrollTop - node.clientHeight < 48;
  node.textContent = text;
  if (stick || node.dataset.stick !== "0") node.scrollTop = node.scrollHeight;
};

window.drawLossCurves = function drawLossCurves(canvas, history) {
  const empty = document.getElementById("curves-empty");
  const legend = document.getElementById("curves-legend");
  const points = (history || []).filter((row) => Number.isFinite(Number(row.total_loss)));
  if (!canvas) return;
  if (points.length === 0) {
    canvas.hidden = true;
    if (empty) empty.hidden = false;
    if (legend) legend.hidden = true;
    return;
  }
  canvas.hidden = false;
  if (empty) empty.hidden = true;
  if (legend) legend.hidden = false;

  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvas.getContext("2d");
  const pad = { left: 48, right: 16, top: 16, bottom: 36 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const xs = points.map((row) => Number(row.iter) || 0);
  const series = [
    { key: "total_loss", color: "#b23a2f" },
    { key: "loss_mask", color: "#2a9d8f" },
  ];
  let yMin = Infinity;
  let yMax = -Infinity;
  series.forEach((item) => {
    points.forEach((row) => {
      const value = Number(row[item.key]);
      if (!Number.isFinite(value)) return;
      yMin = Math.min(yMin, value);
      yMax = Math.max(yMax, value);
    });
  });
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) return;
  if (yMin === yMax) {
    yMin = Math.max(0, yMin - 0.1);
    yMax += 0.1;
  }
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const xSpan = Math.max(xMax - xMin, 1);
  const ySpan = yMax - yMin;

  const xOf = (iter) => pad.left + ((iter - xMin) / xSpan) * plotW;
  const yOf = (value) => pad.top + plotH - ((value - yMin) / ySpan) * plotH;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fffdf8";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d7cfc2";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  ctx.fillStyle = "#5c6778";
  ctx.font = "12px ui-monospace, Menlo, Consolas, monospace";
  ctx.fillText(yMax.toFixed(3), 4, pad.top + 10);
  ctx.fillText(yMin.toFixed(3), 4, pad.top + plotH);
  ctx.fillText(String(xMin), pad.left, height - 8);
  ctx.fillText(String(xMax), pad.left + plotW - 24, height - 8);
  ctx.fillText("iteration", pad.left + plotW / 2 - 24, height - 8);

  series.forEach((item) => {
    const usable = points.filter((row) => Number.isFinite(Number(row[item.key])));
    if (usable.length === 0) return;
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    usable.forEach((row, index) => {
      const x = xOf(Number(row.iter) || 0);
      const y = yOf(Number(row[item.key]));
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
};
