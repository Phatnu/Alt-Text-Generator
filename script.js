/* Alt Text Studio
   Caption model runs locally via Transformers.js v3 (WebGPU when available,
   WASM/CPU otherwise). Every image gets filename-based alt text immediately,
   then the model upgrades it in place. */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
const cacheAvailable = (() => { try { return typeof caches !== 'undefined' && !!caches.open; } catch { return false; } })();
env.useBrowserCache = cacheAvailable;
if (!cacheAvailable) {
  console.warn('[altgen] Browser cache unavailable — if you opened this as a local file, serve it over http://localhost so the model is not re-downloaded each visit.');
}

const MAX_IMAGES = 10;
const TARGET_BYTES = 300 * 1024;   // 300 KB
const MAX_DIM = 1920;
const MODEL_ID = 'Xenova/vit-gpt2-image-captioning';
const CAPTION_CONCURRENCY = 2;

/* ── dom ──────────────────────────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);
const fileInput = $('file');
const drop = $('drop');
const queueEl = $('queue');
const genBtn = $('genBtn');
const clearBtn = $('clearBtn');
const statusEl = $('status');
const altEl = $('alt');
const titleEl = $('title');
const snippetEl = $('snippet');
const countEl = $('count');
const metaEl = $('fileMeta');
const badge = $('modelBadge');
const urlInput = $('url');
const batchListEl = $('batchList');
const doneCountEl = $('doneCount');
const selLabel = $('selLabel');
const sizeInfo = $('sizeInfo');
const backendSel = $('backend');
const previewEl = $('preview');
const previewImg = $('previewImg');
const previewEmpty = $('previewEmpty');
const progressBar = $('progressBar');

/* ── state ────────────────────────────────────────────────────────────── */
let captioner = null;
let loadingPromise = null;
let captionDevice = 'wasm';
let loadFailed = false;
let backendPref = 'auto';
try { backendPref = localStorage.getItem('altgen-backend') || 'auto'; } catch {}

let items = [];       // { id, name, origSize, size, src, objectUrl, blob, mime, w, h, status, alt, title, didCompress, editedAlt, editedTitle, ms }
let selectedId = null;
let processing = false;
let nextId = 1;

const STATUS_TEXT = { compressing: 'preparing', draft: 'draft', working: 'writing', done: 'captioned', fallback: 'needs review' };

/* ── small helpers ────────────────────────────────────────────────────── */
function log(m) { statusEl.textContent = m; }
function setBadge(state, text) { badge.className = 'badge ' + state; badge.textContent = text; }
function setProgress(pct) { progressBar.style.width = Math.max(0, Math.min(100, pct)) + '%'; }
function updateProgress() {
  const total = items.length;
  if (!total) { setProgress(0); return; }
  const done = items.filter((i) => i.status === 'done' || i.status === 'fallback').length;
  setProgress((done / total) * 100);
}

function filenameToAlt(name) {
  return (name || '').replace(/\.[a-z0-9]+$/i, '').replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim() || 'Descriptive image';
}
function formatBytes(n) {
  if (n == null) return '—';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return Math.round(n / 1024) + ' KB';
  return (n / (1024 * 1024)).toFixed(1) + ' MB';
}
function slugify(text, maxLen = 60) {
  const s = (text || '').toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, maxLen).replace(/-+$/g, '');
  return s || 'image';
}
function extForMime(mime) {
  return (mime || '').toLowerCase().includes('png') ? 'png' : 'jpg';
}
function normalizedOutputMime(mime) {
  return (mime || '').toLowerCase().includes('png') ? 'image/png' : 'image/jpeg';
}
function baseOf(name) { return (name || 'image').replace(/\.[a-z0-9]+$/i, ''); }

function downloadNames() {
  const used = new Set();
  const map = new Map();
  for (const it of items) {
    const ext = extForMime(it.mime);
    const stem = slugify(it.alt || it.title || baseOf(it.name));
    let name = `${stem}.${ext}`;
    let k = 2;
    while (used.has(name)) name = `${stem}-${k++}.${ext}`;
    used.add(name);
    map.set(it.id, name);
  }
  return map;
}

function escapeAttr(s) { return String(s ?? '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function escapeHtml(s) { return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function csvCell(s) { return '"' + String(s ?? '').replace(/"/g, '""') + '"'; }

/* ── engine ───────────────────────────────────────────────────────────── */
async function detectWebGPU() {
  try {
    if (typeof navigator === 'undefined' || !navigator.gpu) return { ok: false };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false };
    return { ok: true, fp16: adapter.features.has('shader-f16') };
  } catch { return { ok: false }; }
}

function modelProgress(p) {
  if (p.status === 'downloading' || p.status === 'progress') {
    const pct = p.progress ? Math.round(p.progress) : 0;
    log(`Fetching the caption model — ${p.file || 'weights'} ${pct}%`);
    setProgress(pct);
  }
}

async function loadCaptioner() {
  const candidates = [];
  const gpu = await detectWebGPU();

  if (backendPref === 'cpu') {
    candidates.push({ device: 'wasm', dtype: 'q8' });
  } else if (backendPref === 'gpu') {
    if (gpu.ok) candidates.push(gpu.fp16 ? { device: 'webgpu', dtype: 'fp16' } : { device: 'webgpu', dtype: 'fp32' });
    candidates.push({ device: 'wasm', dtype: 'q8' });
  } else {
    if (gpu.ok && gpu.fp16) candidates.push({ device: 'webgpu', dtype: 'fp16' });
    candidates.push({ device: 'wasm', dtype: 'q8' });
  }

  let lastErr = null;
  for (const c of candidates) {
    try {
      log(`Warming up the caption model on ${c.device === 'webgpu' ? 'the GPU' : 'the CPU'}…`);
      const pipe = await pipeline('image-to-text', MODEL_ID, {
        device: c.device,
        dtype: c.dtype,
        progress_callback: modelProgress,
      });
      captionDevice = c.device;
      return pipe;
    } catch (e) {
      lastErr = e;
      log(`${c.device}/${c.dtype} unavailable — trying the next backend…`);
    }
  }
  throw lastErr || new Error('No caption backend available');
}

async function getCaptioner() {
  if (captioner) return captioner;
  if (loadingPromise) return loadingPromise;
  if (loadFailed) throw new Error('Caption model unavailable');
  setBadge('busy', 'model loading');
  loadingPromise = loadCaptioner().then((c) => {
    captioner = c;
    setBadge('done', captionDevice === 'webgpu' ? 'model ready (GPU)' : 'model ready (CPU)');
    log(captionDevice === 'webgpu'
      ? 'Caption model ready on the GPU.'
      : 'Caption model ready on the CPU.');
    return c;
  }).catch((e) => {
    loadingPromise = null;
    loadFailed = true;
    setBadge('idle', 'model failed');
    log('Could not load the caption model: ' + (e && e.message ? e.message : e) + '\nFilenames will be used as alt text instead.');
    throw e;
  });
  return loadingPromise;
}

// Preload in the background.
getCaptioner().catch(() => {});

/* ── image compression ────────────────────────────────────────────────── */
function loadBitmap(blob) {
  if (typeof createImageBitmap === 'function') {
    return createImageBitmap(blob).then(
      (bmp) => ({ draw: (ctx, w, h) => ctx.drawImage(bmp, 0, 0, w, h), w: bmp.width, h: bmp.height, close: () => { try { bmp.close(); } catch {} } }),
    ).catch(() => loadViaImage(blob));
  }
  return loadViaImage(blob);
}
function loadViaImage(blob) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => resolve({ draw: (ctx, w, h) => ctx.drawImage(img, 0, 0, w, h), w: img.naturalWidth || 0, h: img.naturalHeight || 0, close: () => URL.revokeObjectURL(url) });
    img.onerror = (e) => { URL.revokeObjectURL(url); reject(e); };
    img.src = url;
  });
}
function canvasToBlob(canvas, mime, q) {
  return new Promise((resolve) => canvas.toBlob((b) => resolve(b), mime, q));
}
function hasAlpha(src) {
  try {
    const pw = Math.max(1, Math.min(160, src.w));
    const ph = Math.max(1, Math.round((src.h / src.w) * pw)) || 1;
    const c = document.createElement('canvas');
    c.width = pw; c.height = Math.min(ph, 160);
    const x = c.getContext('2d', { willReadFrequently: true });
    x.clearRect(0, 0, c.width, c.height);
    src.draw(x, c.width, c.height);
    const d = x.getImageData(0, 0, c.width, c.height).data;
    for (let i = 3; i < d.length; i += 16) if (d[i] < 250) return true;
    return false;
  } catch { return false; }
}

async function compressImageBlob(origBlob, origType) {
  const rawType = origType || origBlob.type || 'image/jpeg';
  const type = normalizedOutputMime(rawType);
  if (origBlob.size <= TARGET_BYTES && (rawType === 'image/png' || rawType === 'image/jpeg' || rawType === 'image/jpg')) {
    let w = null, h = null;
    try { const d = await loadBitmap(origBlob); w = d.w; h = d.h; d.close(); } catch {}
    return { blob: origBlob, mime: type, w, h, didCompress: false };
  }
  let src;
  try { src = await loadBitmap(origBlob); }
  catch { return { blob: origBlob, mime: type, w: null, h: null, didCompress: false }; }
  if (!src.w || !src.h) { try { src.close(); } catch {} return { blob: origBlob, mime: type, w: null, h: null, didCompress: false }; }

  const transparent = hasAlpha(src);
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  if (transparent) {
    const s0 = Math.min(1, MAX_DIM / Math.max(src.w, src.h));
    const scales = [s0, s0 * 0.8, s0 * 0.62, s0 * 0.45, s0 * 0.32, s0 * 0.22, s0 * 0.15].map((s) => Math.max(s, 0.06));
    let best = null;
    for (const sc of scales) {
      const w = Math.max(1, Math.round(src.w * sc));
      const h = Math.max(1, Math.round(src.h * sc));
      canvas.width = w; canvas.height = h;
      ctx.clearRect(0, 0, w, h);
      src.draw(ctx, w, h);
      const b = await canvasToBlob(canvas, 'image/png');
      if (!b) break;
      if (!best || b.size < best.blob.size) best = { blob: b, mime: 'image/png', w, h };
      if (b.size <= TARGET_BYTES) { try { src.close(); } catch {} return { blob: b, mime: 'image/png', w, h, didCompress: true }; }
    }
    if (best && best.blob.size <= TARGET_BYTES) { try { src.close(); } catch {} return { ...best, didCompress: true }; }
  }

  const s0 = Math.min(1, MAX_DIM / Math.max(src.w, src.h));
  const scales = [s0, s0 * 0.8, s0 * 0.62, s0 * 0.45, s0 * 0.32].map((s) => Math.max(s, 0.08));
  const quals = [0.86, 0.78, 0.7, 0.6, 0.5, 0.4];
  let best = null;
  for (const sc of scales) {
    const w = Math.max(1, Math.round(src.w * sc));
    const h = Math.max(1, Math.round(src.h * sc));
    canvas.width = w; canvas.height = h;
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, w, h);
    src.draw(ctx, w, h);
    for (const q of quals) {
      const b = await canvasToBlob(canvas, 'image/jpeg', q);
      if (!b) break;
      if (!best || b.size < best.blob.size) best = { blob: b, mime: 'image/jpeg', w, h };
      if (b.size <= TARGET_BYTES) { try { src.close(); } catch {} return { blob: b, mime: 'image/jpeg', w, h, didCompress: true }; }
    }
  }
  try { src.close(); } catch {}
  if (best) return { ...best, didCompress: true };
  return { blob: origBlob, mime: type, w: src.w, h: src.h, didCompress: false };
}

/* ── queue ────────────────────────────────────────────────────────────── */
function remainingSlots() { return MAX_IMAGES - items.length; }

function newItem(name, size) {
  return {
    id: nextId++, name, origSize: size, size, src: '', objectUrl: null, blob: null,
    mime: 'image/jpeg', w: null, h: null, status: 'compressing',
    alt: filenameToAlt(name), title: filenameToAlt(name),
    didCompress: false, editedAlt: false, editedTitle: false, ms: undefined,
  };
}

async function addFiles(files) {
  const imgs = Array.from(files || []).filter((f) => f.type.startsWith('image/'));
  if (!imgs.length) return;
  const room = remainingSlots();
  if (room <= 0) { log(`Limit reached — ${MAX_IMAGES} images at a time.`); return; }
  const accepted = imgs.slice(0, room);
  const skipped = imgs.length - accepted.length;
  fileInput.value = '';
  loadFailed = false;

  const queue = [...accepted];
  let ix = 0;
  const workers = [];
  const concurrency = Math.min(2, queue.length);
  log(`Preparing ${accepted.length} image${accepted.length > 1 ? 's' : ''}…`);
  for (let w = 0; w < concurrency; w++) {
    workers.push((async () => {
      while (ix < queue.length) {
        const f = queue[ix++];
        const item = newItem(f.name, f.size);
        item.mime = f.type || 'image/jpeg';
        items.push(item);
        if (selectedId == null) selectedId = item.id;
        render();
        try {
          const r = await compressImageBlob(f, f.type);
          const url = URL.createObjectURL(r.blob);
          Object.assign(item, { blob: r.blob, mime: r.mime, w: r.w, h: r.h, didCompress: r.didCompress, size: r.blob.size, src: url, objectUrl: url, status: 'draft' });
        } catch (e) {
          const url = URL.createObjectURL(f);
          Object.assign(item, { blob: f, size: f.size, src: url, objectUrl: url, status: 'draft' });
        }
        render();
        processQueue().catch(() => {}); // start captioning while other files compress
      }
    })());
  }
  await Promise.all(workers);

  if (skipped > 0) log(`Added ${accepted.length}. Skipped ${skipped} — the limit is ${MAX_IMAGES}.`);
  processQueue().catch(() => {});
}

async function addUrlItem(url) {
  if (remainingSlots() <= 0) { log(`Limit reached — ${MAX_IMAGES} images at a time.`); return; }
  loadFailed = false;
  let name = 'remote-image.jpg';
  try { name = decodeURIComponent(url.split('/').pop().split('?')[0]) || name; } catch {}
  const item = newItem(name, null);
  item.src = url; item.mime = 'image/jpeg';
  items.push(item);
  if (selectedId == null) selectedId = item.id;
  render();
  log(`Fetching ${name}…`);
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const blob = await res.blob();
    if (!blob.type.startsWith('image/')) throw new Error('that URL is not an image');
    const r = await compressImageBlob(blob, blob.type);
    const ourl = URL.createObjectURL(r.blob);
    Object.assign(item, { origSize: blob.size, blob: r.blob, mime: r.mime, w: r.w, h: r.h, didCompress: r.didCompress, size: r.blob.size, src: ourl, objectUrl: ourl, status: 'draft' });
  } catch (e) {
    item.status = 'draft';
    log(`Added the URL without compressing (${e && e.message ? e.message : e}).`);
  }
  render();
  processQueue().catch(() => {});
}

function removeItem(id) {
  const ix = items.findIndex((i) => i.id === id);
  if (ix < 0) return;
  const [gone] = items.splice(ix, 1);
  if (gone.objectUrl) { try { URL.revokeObjectURL(gone.objectUrl); } catch {} }
  if (selectedId === id) selectedId = items.length ? items[Math.min(ix, items.length - 1)].id : null;
  render();
}

function selectItem(id) { selectedId = id; render(); }
function selectedItem() { return items.find((i) => i.id === selectedId) || null; }

/* ── captioning ───────────────────────────────────────────────────────── */
async function captionOne(item) {
  item.status = 'working';
  render();
  const t0 = performance.now();
  try {
    const cap = await getCaptioner();
    const out = await cap(item.src);
    let text = '';
    if (Array.isArray(out) && out[0] && out[0].generated_text) text = out[0].generated_text;
    else if (typeof out === 'string') text = out;
    text = (text || '').trim();
    if (!text) throw new Error('empty caption');
    text = text.charAt(0).toUpperCase() + text.slice(1);
    const prevAlt = item.alt;
    if (!item.editedAlt) item.alt = text;
    if (!item.editedTitle && (!item.title || item.title === prevAlt || item.title === filenameToAlt(item.name))) item.title = item.alt;
    item.status = 'done';
  } catch (e) {
    const fb = filenameToAlt(item.name);
    if (!item.editedAlt) item.alt = fb;
    if (!item.editedTitle) item.title = fb;
    item.status = 'fallback';
  }
  item.ms = Math.round(performance.now() - t0);
  render();
}

function isPending(it) { return it.status === 'draft'; }

async function processQueue() {
  if (processing) return;
  if (!items.length) { render(); return; }
  if (!items.some(isPending)) { render(); return; }
  processing = true;
  genBtn.disabled = true;
  setBadge('busy', 'model working');
  try {
    try { await getCaptioner(); } catch {}
    // Keep draining — fresh images can finish compressing while we work.
    while (items.some(isPending)) {
      const queue = items.filter(isPending);
      let ix = 0;
      const concurrency = Math.min(CAPTION_CONCURRENCY, queue.length);
      const workers = [];
      for (let w = 0; w < concurrency; w++) {
        workers.push((async () => {
          while (ix < queue.length) {
            const item = queue[ix++];
            log(`Writing alt text for ${item.name}…`);
            await captionOne(item);
          }
        })());
      }
      await Promise.all(workers);
    }

    const dones = items.filter((i) => i.status === 'done').length;
    const fbs = items.filter((i) => i.status === 'fallback').length;
    const timed = items.filter((i) => typeof i.ms === 'number');
    const avg = timed.length ? Math.round(timed.reduce((a, i) => a + i.ms, 0) / timed.length) : 0;
    setBadge('done', captionDevice === 'webgpu' ? 'model ready (GPU)' : 'model ready (CPU)');
    const on = captionDevice === 'webgpu' ? 'the GPU' : 'the CPU';
    const tail = avg ? ` Around ${avg} ms per image on ${on}.` : '';
    log(fbs
      ? `Done — ${dones} captioned, ${fbs} left as filename drafts.${tail}`
      : `Done — all ${dones} captioned.${tail} Read them before you ship.`);
  } finally {
    processing = false;
    genBtn.disabled = items.length === 0;
    render();
  }
}

async function generateAll() {
  loadFailed = false;
  for (const i of items) if (i.status === 'done' || i.status === 'fallback') i.status = 'draft';
  render();
  await processQueue();
}

async function setBackend(pref) {
  if (pref === backendPref && captioner) return;
  backendPref = pref;
  try { localStorage.setItem('altgen-backend', pref); } catch {}
  captioner = null;
  loadingPromise = null;
  loadFailed = false;
  for (const it of items) {
    if (it.status === 'done' || it.status === 'fallback') { it.status = 'draft'; it.ms = undefined; }
  }
  render();
  log(`Switching compute to ${pref === 'gpu' ? 'the GPU' : pref === 'cpu' ? 'the CPU' : 'Auto'}…`);
  await processQueue();
}

/* ── rendering ────────────────────────────────────────────────────────── */
function sizeLine(it) {
  const dims = it.w && it.h ? `, ${it.w}×${it.h}` : '';
  if (it.size == null) return 'size unknown' + dims;
  const saved = it.origSize && it.origSize > it.size ? ` (was ${formatBytes(it.origSize)})` : '';
  return `${formatBytes(it.size)}${saved}${dims}`;
}

function makeTile(item) {
  const b = document.createElement('div');
  b.className = 'tile';
  b.dataset.id = String(item.id);
  b.tabIndex = 0;
  b.setAttribute('role', 'button');
  b.title = item.name;

  const frame = document.createElement('div');
  frame.className = 'tile-frame';
  const img = document.createElement('img');
  img.alt = '';
  frame.appendChild(img);

  const remove = document.createElement('button');
  remove.className = 'tile-remove';
  remove.type = 'button';
  remove.textContent = '×';
  remove.title = 'Remove';
  remove.setAttribute('aria-label', 'Remove ' + item.name);
  remove.addEventListener('click', (e) => { e.stopPropagation(); removeItem(item.id); });

  const cap = document.createElement('div');
  cap.className = 'tile-cap';
  const alt = document.createElement('p');
  alt.className = 'tile-alt';
  const foot = document.createElement('div');
  foot.className = 'tile-foot';
  const st = document.createElement('span');
  st.className = 'tile-status';
  const ms = document.createElement('span');
  ms.className = 'tile-ms';
  foot.append(st, ms);
  cap.append(alt, foot);

  b.append(frame, remove, cap);
  b.addEventListener('click', () => selectItem(item.id));
  b.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectItem(item.id); } });
  updateTile(b, item);
  return b;
}

function updateTile(el, item) {
  el.classList.toggle('selected', item.id === selectedId);
  el.classList.toggle('working', item.status === 'working');
  el.classList.toggle('compressing', item.status === 'compressing');
  const img = el.querySelector('img');
  if (img && item.src && img.getAttribute('src') !== item.src) img.src = item.src;
  const alt = el.querySelector('.tile-alt');
  alt.textContent = item.alt || 'No alt text yet';
  alt.classList.toggle('empty', !item.alt);
  const st = el.querySelector('.tile-status');
  st.className = 'tile-status ' + item.status;
  st.textContent = STATUS_TEXT[item.status] || item.status;
  const ms = el.querySelector('.tile-ms');
  ms.textContent = typeof item.ms === 'number' ? item.ms + ' ms' : '';
}

function renderSheet() {
  const desired = items.map((i) => String(i.id));
  const current = Array.from(queueEl.children).map((el) => el.dataset.id);
  if (desired.join(',') !== current.join(',')) {
    queueEl.textContent = '';
    for (const item of items) queueEl.appendChild(makeTile(item));
  } else {
    for (const item of items) {
      const el = queueEl.querySelector(`[data-id="${item.id}"]`);
      if (el) updateTile(el, item);
    }
  }
}

function render() {
  const names = downloadNames();
  renderSheet();

  // selected preview + fields
  const sel = selectedItem();
  if (!sel) {
    previewEl.hidden = true;
    previewEmpty.hidden = false;
    altEl.value = '';
    titleEl.value = '';
    snippetEl.textContent = '<img src="" alt="" title="" />';
    countEl.textContent = '0 characters';
    selLabel.textContent = '';
    sizeInfo.textContent = '';
  } else {
    previewEmpty.hidden = true;
    previewEl.hidden = false;
    if (sel.src) {
      if (previewImg.getAttribute('src') !== sel.src) previewImg.src = sel.src;
    } else {
      previewImg.removeAttribute('src');
    }
    previewImg.alt = sel.alt || '';
    if (document.activeElement !== altEl || altEl.dataset.forId !== String(sel.id)) {
      altEl.value = sel.alt || '';
      altEl.dataset.forId = String(sel.id);
    }
    if (document.activeElement !== titleEl || titleEl.dataset.forId !== String(sel.id)) {
      titleEl.value = sel.title || '';
      titleEl.dataset.forId = String(sel.id);
    }
    updateSnippet(names);
    selLabel.textContent = sel.name;
    sizeInfo.textContent = sizeLine(sel) + ', saves as ' + (names.get(sel.id) || sel.name);
  }

  // batch list
  batchListEl.textContent = '';
  let done = 0;
  for (const item of items) {
    if (item.status === 'done' || item.status === 'fallback') done++;
    const row = document.createElement('div');
    row.className = 'bitem' + (item.id === selectedId ? ' selected' : '');
    const head = document.createElement('div');
    head.className = 'bitem-head';
    const nm = document.createElement('div');
    nm.className = 'bitem-name';
    nm.textContent = item.name;
    const stt = document.createElement('div');
    stt.className = 'bitem-state ' + item.status;
    stt.textContent = STATUS_TEXT[item.status] || item.status;
    head.append(nm, stt);
    const a = document.createElement('p');
    a.className = 'bitem-alt' + (item.alt ? '' : ' empty');
    a.textContent = item.alt || 'Waiting for alt text…';
    const m = document.createElement('div');
    m.className = 'bitem-meta';
    m.textContent = `saves as ${names.get(item.id)}, ${sizeLine(item)}${typeof item.ms === 'number' ? `, ${item.ms} ms` : ''}`;
    row.append(head, a, m);
    row.addEventListener('click', () => selectItem(item.id));
    batchListEl.appendChild(row);
  }
  doneCountEl.textContent = items.length ? `${done} of ${items.length} done` : '0 done';

  metaEl.textContent = `${items.length} of ${MAX_IMAGES}`;
  const busy = processing || items.some((i) => i.status === 'compressing');
  genBtn.disabled = busy || items.length === 0;
  if (backendSel) backendSel.disabled = processing;
  updateProgress();
}

function snippetFor(item, names) {
  const isSel = item.id === selectedId;
  const alt = (isSel ? altEl.value : item.alt || '').trim();
  const rawTitle = (isSel ? titleEl.value : item.title || '').trim();
  const title = rawTitle || alt;
  const src = names.get(item.id) || item.name;
  const dims = item.w && item.h ? ` width="${item.w}" height="${item.h}"` : '';
  return `<img src="${src}" alt="${escapeAttr(alt)}" title="${escapeAttr(title)}"${dims} />`;
}

function updateSnippet(precomputed) {
  const sel = selectedItem();
  const alt = altEl.value.trim();
  countEl.textContent = altEl.value.length + ' characters';
  if (!sel) { snippetEl.textContent = '<img src="" alt="" title="" />'; return; }
  const names = precomputed || downloadNames();
  const title = titleEl.value.trim() || alt;
  const src = names.get(sel.id) || sel.name;
  const dims = sel.w && sel.h ? ` width="${sel.w}" height="${sel.h}"` : '';
  snippetEl.textContent = `<img src="${src}" alt="${escapeAttr(alt)}" title="${escapeAttr(title)}"${dims} />`;
}

function renderBatchTextOnly() {
  const names = downloadNames();
  const rows = batchListEl.children;
  items.forEach((item, ix) => {
    const row = rows[ix];
    if (!row) return;
    const altDiv = row.querySelector('.bitem-alt');
    if (altDiv) { altDiv.textContent = item.alt || 'Waiting for alt text…'; altDiv.classList.toggle('empty', !item.alt); }
    const metaDiv = row.querySelector('.bitem-meta');
    if (metaDiv) metaDiv.textContent = `saves as ${names.get(item.id)}, ${sizeLine(item)}${typeof item.ms === 'number' ? `, ${item.ms} ms` : ''}`;
    const tile = queueEl.querySelector(`[data-id="${item.id}"] .tile-alt`);
    if (tile) { tile.textContent = item.alt || 'No alt text yet'; tile.classList.toggle('empty', !item.alt); }
  });
  const sel = selectedItem();
  if (sel && sizeInfo) sizeInfo.textContent = sizeLine(sel) + ', saves as ' + (names.get(sel.id) || sel.name);
}

/* ── exports ──────────────────────────────────────────────────────────── */
function buildHTML(names) { return items.map((i) => snippetFor(i, names)).join('\n'); }
function buildCSV(names) {
  const rows = [['filename', 'title', 'alt', 'width', 'height', 'size_kb']];
  for (const i of items) rows.push([names.get(i.id) || i.name, i.title || '', i.alt || '', i.w || '', i.h || '', i.size ? Math.round(i.size / 1024) : '']);
  return rows.map((r) => r.map(csvCell).join(',')).join('\n');
}
function buildManifest(names) {
  const cards = items.map((i) => {
    const fn = escapeHtml(names.get(i.id) || i.name);
    const t = i.title || i.alt || '';
    return `  <figure>\n    <img src="${fn}" alt="${escapeAttr(i.alt)}" title="${escapeAttr(t)}"${i.w ? ` width="${i.w}" height="${i.h}"` : ''} />\n    <figcaption><strong>${fn}</strong> (${formatBytes(i.size)})<br>alt: ${escapeHtml(i.alt)}<br>title: ${escapeHtml(t)}</figcaption>\n  </figure>`;
  }).join('\n');
  return `<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8" />\n<title>Alt text export — ${items.length} images (300 KB each)</title>\n<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:32px auto;padding:0 16px}figure{border:1px solid #ddd;border-radius:8px;padding:12px}img{max-width:100%;height:auto}</style>\n</head>\n<body>\n<h1>Exported images (${items.length})</h1>\n${cards}\n<h2>HTML snippets</h2>\n<pre>${escapeHtml(buildHTML(names))}</pre>\n</body>\n</html>`;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

async function ensureBlob(item) {
  if (item.blob) return item.blob;
  const res = await fetch(item.src);
  const blob = await res.blob();
  const r = await compressImageBlob(blob, blob.type);
  item.blob = r.blob; item.mime = r.mime; item.w = r.w; item.h = r.h; item.size = r.blob.size;
  if (item.objectUrl) { try { URL.revokeObjectURL(item.objectUrl); } catch {} }
  const url = URL.createObjectURL(r.blob);
  item.src = url; item.objectUrl = url;
  render();
  return r.blob;
}

async function downloadOne(item, precomputed) {
  try {
    const blob = await ensureBlob(item);
    const names = precomputed || downloadNames();
    downloadBlob(blob, names.get(item.id) || item.name);
    log(`Saved “${names.get(item.id)}”.`);
  } catch (e) {
    log('Download failed: ' + (e && e.message ? e.message : e));
  }
}

let JSZipClass = null;
async function getJSZip() {
  if (JSZipClass) return JSZipClass;
  const mod = await import('https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm');
  JSZipClass = mod.default || mod.JSZip;
  return JSZipClass;
}

async function downloadAllZip() {
  if (!items.length) return log('Nothing to download yet.');
  const names = downloadNames();
  log('Building the ZIP…');
  try {
    const Zip = await getJSZip();
    const zip = new Zip();
    for (const item of items) zip.file(names.get(item.id) || item.name, await ensureBlob(item));
    zip.file('alt-text.html', buildManifest(names));
    zip.file('alt-text.csv', buildCSV(names));
    zip.file('alt-text.txt', items.map((i) => `${names.get(i.id)} | title: ${i.title || ''} | alt: ${i.alt || ''}`).join('\n'));
    const out = await zip.generateAsync({ type: 'blob' });
    downloadBlob(out, 'alt-text-images-300kb.zip');
    log(`Saved the ZIP — ${items.length} image${items.length > 1 ? 's' : ''} plus the manifest.`);
  } catch (e) {
    log('ZIP failed (' + (e && e.message ? e.message : e) + ') — downloading the files individually instead.');
    for (const item of items) {
      try {
        downloadBlob(await ensureBlob(item), names.get(item.id) || item.name);
        await new Promise((r) => setTimeout(r, 400));
      } catch {}
    }
    downloadBlob(new Blob([buildManifest(names)], { type: 'text/html' }), 'alt-text.html');
  }
}

async function copyText(t, okMsg) {
  try {
    await navigator.clipboard.writeText(t);
    log(okMsg);
  } catch (e) {
    const ta = document.createElement('textarea');
    ta.value = t;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); log(okMsg); }
    catch { log('Copy failed — select the text and copy it manually.'); }
    ta.remove();
  }
}
function flashCopied(btn) {
  if (!btn) return;
  const orig = btn.dataset.label || btn.textContent;
  btn.dataset.label = orig;
  btn.textContent = 'copied';
  btn.classList.add('copied');
  setTimeout(() => { btn.textContent = orig; btn.classList.remove('copied'); }, 1100);
}

/* ── events ───────────────────────────────────────────────────────────── */
fileInput.addEventListener('change', () => { void addFiles(fileInput.files); });

drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('drag'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('drop', (e) => {
  const files = e.dataTransfer && e.dataTransfer.files;
  if (files && files.length) void addFiles(files);
});

$('loadUrl').addEventListener('click', () => {
  const u = urlInput.value.trim();
  if (!u) return;
  urlInput.value = '';
  void addUrlItem(u);
});
urlInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') $('loadUrl').click(); });

clearBtn.addEventListener('click', () => {
  for (const i of items) if (i.objectUrl) { try { URL.revokeObjectURL(i.objectUrl); } catch {} }
  items = [];
  selectedId = null;
  fileInput.value = '';
  altEl.dataset.forId = '';
  titleEl.dataset.forId = '';
  render();
  setProgress(0);
  log('Cleared. Choose images to begin.');
});

genBtn.addEventListener('click', () => { void generateAll(); });

if (backendSel) {
  backendSel.value = backendPref;
  backendSel.addEventListener('change', () => { void setBackend(backendSel.value); });
}

altEl.addEventListener('input', () => {
  const sel = selectedItem();
  if (sel) { sel.alt = altEl.value; sel.editedAlt = true; }
  updateSnippet();
  renderBatchTextOnly();
});
titleEl.addEventListener('input', () => {
  const sel = selectedItem();
  if (sel) { sel.title = titleEl.value; sel.editedTitle = true; }
  updateSnippet();
  renderBatchTextOnly();
});

$('copyAlt').addEventListener('click', () => copyText(altEl.value, 'Alt text copied.'));
$('copyTitle').addEventListener('click', () => copyText(titleEl.value, 'Title copied.'));
$('copyAltInline').addEventListener('click', async () => { await copyText(altEl.value, 'Alt text copied.'); flashCopied($('copyAltInline')); });
$('copyTitleInline').addEventListener('click', async () => { await copyText(titleEl.value, 'Title copied.'); flashCopied($('copyTitleInline')); });
$('copyHtml').addEventListener('click', () => copyText(snippetEl.textContent, 'HTML snippet copied.'));
$('copyAllAlt').addEventListener('click', () => {
  if (!items.length) return log('Nothing to copy yet.');
  copyText(items.map((i) => i.alt).join('\n'), `Copied ${items.length} alt text${items.length > 1 ? 's' : ''}.`);
});
$('copyAllTitle').addEventListener('click', () => {
  if (!items.length) return log('Nothing to copy yet.');
  copyText(items.map((i) => i.title || i.alt).join('\n'), `Copied ${items.length} title${items.length > 1 ? 's' : ''}.`);
});
$('copyAllHtml').addEventListener('click', () => {
  if (!items.length) return log('Nothing to copy yet.');
  copyText(buildHTML(downloadNames()), `Copied ${items.length} HTML snippet${items.length > 1 ? 's' : ''}.`);
});
$('dlOne').addEventListener('click', () => {
  const sel = selectedItem();
  if (!sel) return log('Select an image first.');
  void downloadOne(sel);
});
$('dlAll').addEventListener('click', () => { void downloadAllZip(); });
$('dlManifest').addEventListener('click', () => {
  if (!items.length) return log('Nothing to export yet.');
  const names = downloadNames();
  downloadBlob(new Blob([buildManifest(names)], { type: 'text/html' }), 'alt-text.html');
  log('Saved the manifest.');
});

/* theme */
(function initTheme() {
  const btn = $('themeToggle');
  if (!btn) return;
  const sync = () => {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    btn.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
  };
  btn.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('altgen-theme', next); } catch {}
    sync();
  });
  sync();
})();

/* single page-load reveal */
(function reveal() {
  const els = document.querySelectorAll('.hero, .workspace');
  els.forEach((el) => el.classList.add('reveal'));
  requestAnimationFrame(() => requestAnimationFrame(() => els.forEach((el) => el.classList.add('visible'))));
})();

render();
