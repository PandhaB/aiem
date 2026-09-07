window.pollJob = async function pollJob(jobId, ui) {
  ui.box.hidden = false;
  const tick = async () => {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    ui.status.textContent = job.status;
    ui.message.textContent = job.message || "";
    ui.progress.value = job.progress || 0;
    if (job.status === "failed") {
      ui.error.hidden = false;
      ui.error.textContent = job.error || "Job failed.";
      return;
    }
    if (job.status === "completed") {
      if (ui.onComplete) ui.onComplete(job);
      return;
    }
    window.setTimeout(tick, 400);
  };
  tick();
};
