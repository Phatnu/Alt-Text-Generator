import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';
env.allowLocalModels = false;
env.useBrowserCache = true;

const MAX_IMAGES = 10;
const TARGET_BYTES = 300 * 1024; // 300KB
const MAX_DIM = 1920;

const fileInput = document.getElementById('file');
const drop = document.getElementById('drop');
const queueEl = document.getElementById('queue');
const genBtn = document.getElementById('genBtn');
const clearBtn = document.getElementById('clearBtn');
const statusEl = document.getElementById('status');
const altEl = document.getElementById('alt');
const titleEl = document.getElementById('title');
const snippetEl = document.getElementById('snippet');
const countEl = document.getElementById('count');
const metaEl = document.getElementById('fileMeta');
const badge = document.getElementById('modelBadge');
const urlInput = document.getElementById('url');
const batchListEl = document.getElementById('batchList');
const doneCountEl = document.getElementById('doneCount');
const selLabel = document.getElementById('selLabel');
const sizeInfo = document.getElementById('sizeInfo');

let captioner = null;
let loadingPromise = null;
let items = []; // { id, name, origSize, size, src, objectUrl, blob, mime, w, h, status, alt, title, didCompress }
let selectedId = null;
let processing = false;
let nextId = 1;

function log(m) { statusEl.textContent = m; }
function setBadge(state, text) { badge.className = 'badge ' + state; badge.textContent = text; }

async function getCaptioner() {
  if (captioner) return captioner;
  if (loadingPromise) return loadingPromise;
  setBadge('busy', 'model: loading…');
  log('Loading caption model (first run downloads ~500MB, cached after)…');
  loadingPromise = pipeline('image-to-text', 'Xenova/vit-gpt2-image-captioning', {
    progress_callback: (p) => {
      if (p.status === 'downloading' || p.status === 'progress') {
        const pct = p.progress ? Math.round(p.progress) + '%' : '';
        log(`${p.status}: ${p.file || ''} ${pct}`.trim());
      }
    }
  }).then((c) => {
    captioner = c;
    setBadge('done', 'model: ready');
    log(items.length ? `Model ready. ${items.length} image(s) queued.` : 'Model ready. Add images to begin.');
    return c;
  }).catch((e) => {
    loadingPromise = null;
    setBadge('idle', 'model: failed');
    log('Model load failed: ' + (e && e.message ? e.message : e) + '\nFallback: filename-based alt text will be used.');
    throw e;
  });
  return loadingPromise;
}
// Preload in background
getCaptioner().catch(() => {});

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

function extForMime(mime, fallbackName = 'image.jpg') {
  const m = (mime || '').toLowerCase();
  if (m.includes('png')) return 'png';
  // Downloads are JPG or PNG only — webp/gif/etc. all become JPG.
  return 'jpg';
}
function normalizedOutputMime(mime) {
  return (mime || '').toLowerCase().includes('png') ? 'image/png' : 'image/jpeg';
}

function baseOf(name) { return (name || 'image').replace(/\.[a-z0-9]+$/i, ''); }

// Unique download filenames across the batch, derived from alt > title > original name
function downloadNames() {
  const used = new Set();
  const map = new Map();
  for (const it of items) {
    const ext = extForMime(it.mime, it.name);
    let stem = slugify(it.alt || it.title || baseOf(it.name));
    let name = `${stem}.${ext}`;
    let k = 2;
    while (used.has(name)) name = `${stem}-${k++}.${ext}`;
    used.add(name);
    map.set(it.id, name);
  }
  return map;
}
function downloadNameFor(item, precomputed) {
  if (precomputed) return precomputed.get(item.id) || item.name;
  return downloadNames().get(item.id) || item.name;
}

function escapeAttr(s) { return String(s ?? '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function escapeHtml(s) { return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function csvCell(s) { return '"' + String(s ?? '').replace(/"/g, '""') + '"'; }

// ---------- compression to <= 300KB ----------
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

async function compressImageBlob(origBlob, origType) {
  const rawType = origType || origBlob.type || 'image/jpeg';
  const type = normalizedOutputMime(rawType); // JPG or PNG only — never WebP
  // Small files: keep as-is if already JPG/PNG, otherwise transcode small inputs (webp/gif/…) to JPG/PNG
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

  // Transparent images → PNG only (resize until ≤300KB, PNG has no quality setting)
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
    // Extreme case: PNG still too big even tiny → fall through to JPG flatten below
    if (best && best.blob.size <= TARGET_BYTES) { try { src.close(); } catch {} return { ...best, didCompress: true }; }
  }

  // Opaque images (and PNG/JPG fallbacks) → JPEG only
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

// Detect real transparency (not just .png extension) on a small probe canvas
function hasAlpha(src) {
  try {
    const pw = Math.max(1, Math.min(160, src.w));
    const ph = Math.max(1, Math.round((src.h / src.w) * pw)) || 1;
    const c = document.createElement('canvas');
    c.width = pw; c.height = Math.min(ph, 160);
    const x = c.getContext('2d', { willReadFrequently: true });
    x.clearRect(0, 0, c.width, c.height);
    // draw full image scaled into probe
    src.draw(x, c.width, c.height);
    const d = x.getImageData(0, 0, c.width, c.height).data;
    for (let i = 3; i < d.length; i += 16) {
      if (d[i] < 250) return true;
    }
    return false;
  } catch { return false; }
}

// ---------- queue management ----------
function remainingSlots() { return MAX_IMAGES - items.length; }

async function addFiles(files) {
  const imgs = Array.from(files || []).filter((f) => f.type.startsWith('image/'));
  if (!imgs.length) return;
  const room = remainingSlots();
  if (room <= 0) { log(`Limit reached: max ${MAX_IMAGES} images.`); return; }
  const accepted = imgs.slice(0, room);
  const skipped = imgs.length - accepted.length;
  fileInput.value = '';
  log(`Compressing ${accepted.length} image(s) to ≤300KB…`);
  for (const f of accepted) {
    const item = { id: nextId++, name: f.name, origSize: f.size, size: f.size, src: '', objectUrl: null, blob: null, mime: f.type, w: null, h: null, status: 'compressing', alt: '', title: '' };
    items.push(item);
    if (selectedId == null) selectedId = item.id;
    render();
    try {
      const r = await compressImageBlob(f, f.type);
      const url = URL.createObjectURL(r.blob);
      Object.assign(item, { blob: r.blob, mime: r.mime, w: r.w, h: r.h, didCompress: r.didCompress, size: r.blob.size, src: url, objectUrl: url, status: 'queued' });
    } catch (e) {
      const url = URL.createObjectURL(f);
      Object.assign(item, { blob: f, size: f.size, src: url, objectUrl: url, status: 'queued' });
    }
    render();
  }
  if (skipped > 0) log(`Added ${accepted.length}. Skipped ${skipped}: limit is ${MAX_IMAGES}. Compressing done — generating alt text…`);
  else log(`Added ${accepted.length} image(s), each ≤300KB. ${items.length}/${MAX_IMAGES} used. Generating alt text…`);
  processQueue().catch(() => {});
}

async function addUrlItem(url) {
  if (remainingSlots() <= 0) { log(`Limit reached: max ${MAX_IMAGES} images.`); return; }
  let name = 'remote-image.jpg';
  try { name = decodeURIComponent(url.split('/').pop().split('?')[0]) || name; } catch {}
  const item = { id: nextId++, name, origSize: null, size: null, src: url, objectUrl: null, blob: null, mime: 'image/jpeg', w: null, h: null, status: 'compressing', alt: '', title: '' };
  items.push(item);
  if (selectedId == null) selectedId = item.id;
  render();
  log(`Fetching + compressing ${name}…`);
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const blob = await res.blob();
    if (!blob.type.startsWith('image/')) throw new Error('URL is not an image');
    const r = await compressImageBlob(blob, blob.type);
    const ourl = URL.createObjectURL(r.blob);
    Object.assign(item, { origSize: blob.size, blob: r.blob, mime: r.mime, w: r.w, h: r.h, didCompress: r.didCompress, size: r.blob.size, src: ourl, objectUrl: ourl, status: 'queued' });
    log(`Added ${name} at ${formatBytes(item.size)}. Generating…`);
  } catch (e) {
    // keep remote URL (caption may still work if CORS allows; download may fail)
    item.status = 'queued';
    log(`Added URL without compression (${e && e.message ? e.message : e}). Caption will try the remote URL.`);
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
  log(gone ? `Removed "${gone.name}". ${items.length}/${MAX_IMAGES} used.` : 'Removed.');
}

function selectItem(id) { selectedId = id; render(); }
function selectedItem() { return items.find((i) => i.id === selectedId) || null; }

async function captionOne(item) {
  item.status = 'working';
  render();
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
    item.alt = text;
    // Auto-fill title alongside alt (preserve manual edit on retry)
    if (!item.title || item.title === prevAlt) item.title = text;
    item.status = 'done';
  } catch (e) {
    const fb = filenameToAlt(item.name);
    item.alt = fb;
    if (!item.title) item.title = fb;
    item.status = 'fallback';
  }
  render();
}

async function processQueue() {
  if (processing) return;
  if (!items.length) { render(); return; }
  if (!items.some((i) => i.status === 'queued')) { render(); return; }
  processing = true;
  genBtn.disabled = true;
  setBadge('busy', 'model: working…');
  try {
    for (const item of items) {
      if (item.status === 'queued') {
        log(`Generating ${items.indexOf(item) + 1}/${items.length}: ${item.name}…`);
        await captionOne(item);
      }
    }
    const dones = items.filter((i) => i.status === 'done').length;
    const fbs = items.filter((i) => i.status === 'fallback').length;
    setBadge('done', 'model: ready');
    log(fbs ? `Done: ${dones} captioned, ${fbs} fallback (edit before use).` : `Done: all ${dones} captioned. Review before use.`);
  } finally {
    processing = false;
    genBtn.disabled = items.length === 0;
    render();
  }
}

async function generateAll() {
  for (const i of items) if (i.status !== 'done' || !i.alt) i.status = 'queued';
  render();
  await processQueue();
}

// ---------- rendering ----------
function statusLabel(s) {
  return s === 'queued' ? 'queued' : s === 'working' ? 'working…' : s === 'compressing' ? 'compressing…' : s === 'done' ? 'done' : 'filename fallback';
}

function sizeLine(it) {
  const dims = it.w && it.h ? ` · ${it.w}×${it.h}` : '';
  if (it.size == null) return 'size unknown' + dims;
  const saved = it.origSize && it.origSize > it.size ? ` (was ${formatBytes(it.origSize)})` : '';
  return `${formatBytes(it.size)}${saved}${dims}`;
}

function render() {
  const names = downloadNames();
  queueEl.innerHTML = '';
  for (const item of items) {
    const b = document.createElement('div');
    b.className = 'qitem' + (item.id === selectedId ? ' selected' : '');
    b.tabIndex = 0;
    b.title = item.name;
    const img = document.createElement('img');
    img.src = item.src;
    img.alt = '';
    const nm = document.createElement('span');
    nm.className = 'qname';
    nm.textContent = item.name;
    const st = document.createElement('span');
    st.className = 'qst ' + item.status;
    st.textContent = '● ' + statusLabel(item.status);
    const sz = document.createElement('span');
    sz.className = 'qsize';
    sz.textContent = item.status === 'compressing' ? 'compressing to ≤300KB…' : sizeLine(item);
    const x = document.createElement('button');
    x.className = 'qremove';
    x.textContent = '×';
    x.title = 'Remove';
    x.addEventListener('click', (e) => { e.stopPropagation(); removeItem(item.id); });
    b.append(img, nm, st, sz, x);
    b.addEventListener('click', () => selectItem(item.id));
    b.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectItem(item.id); } });
    queueEl.appendChild(b);
  }

  batchListEl.innerHTML = '';
  let done = 0;
  for (const item of items) {
    if (item.status === 'done' || item.status === 'fallback') done++;
    const row = document.createElement('div');
    row.className = 'bitem' + (item.id === selectedId ? ' selected' : '');
    const n = document.createElement('div');
    n.className = 'bname';
    n.textContent = `${item.name} — ${statusLabel(item.status)}`;
    const a = document.createElement('div');
    a.className = 'balt' + (item.alt ? '' : ' empty');
    a.textContent = (item.alt ? `alt: ${item.alt}` : '(pending alt…)');
    const t = document.createElement('div');
    t.className = 'btitle' + (item.title ? '' : ' empty');
    t.textContent = (item.title ? `title: ${item.title}` : '(pending title…)');
    const m = document.createElement('div');
    m.className = 'bmeta';
    m.textContent = `↓ ${names.get(item.id)} · ${sizeLine(item)}`;
    const r = document.createElement('div');
    r.className = 'brow';
    const dl = document.createElement('button');
    dl.className = 'mini';
    dl.textContent = 'Download';
    dl.addEventListener('click', (e) => { e.stopPropagation(); downloadOne(item, names); });
    const sel = document.createElement('button');
    sel.className = 'mini';
    sel.textContent = 'Select';
    sel.addEventListener('click', (e) => { e.stopPropagation(); selectItem(item.id); });
    r.append(dl, sel);
    row.append(n, a, t, m, r);
    row.addEventListener('click', () => selectItem(item.id));
    batchListEl.appendChild(row);
  }
  doneCountEl.textContent = String(done);

  const totalBytes = items.reduce((a, i) => a + (i.size || 0), 0);
  metaEl.textContent = `${items.length} / ${MAX_IMAGES} images` + (items.length ? ` · ${formatBytes(totalBytes)} total` : '');

  const sel = selectedItem();
  if (!sel) {
    altEl.value = '';
    titleEl.value = '';
    snippetEl.textContent = '<img src="" alt="" title="" />';
    countEl.textContent = '0 characters';
    if (selLabel) selLabel.textContent = '';
    if (sizeInfo) sizeInfo.textContent = '';
  } else {
    if (document.activeElement !== altEl || altEl.dataset.forId !== String(sel.id)) {
      altEl.value = sel.alt || '';
      altEl.dataset.forId = String(sel.id);
    }
    if (document.activeElement !== titleEl || titleEl.dataset.forId !== String(sel.id)) {
      titleEl.value = sel.title || '';
      titleEl.dataset.forId = String(sel.id);
    }
    updateSnippet(names);
    if (selLabel) selLabel.textContent = '· ' + sel.name;
    if (sizeInfo) sizeInfo.textContent = sizeLine(sel) + ' · downloads as ' + (names.get(sel.id) || sel.name);
  }
  genBtn.disabled = processing || items.length === 0;
}

function snippetFor(item, names) {
  const isSel = item.id === selectedId;
  const alt = (isSel ? altEl.value : item.alt).trim();
  const rawTitle = (isSel ? titleEl.value : (item.title || '')).trim();
  const title = rawTitle || alt; // always emit title — fall back to alt
  const src = names ? names.get(item.id) || item.name : item.name;
  const dims = item.w && item.h ? ` width="${item.w}" height="${item.h}"` : '';
  return `<img src="${src}" alt="${escapeAttr(alt)}" title="${escapeAttr(title)}"${dims} />`;
}

function updateSnippet(precomputed) {
  const sel = selectedItem();
  const alt = altEl.value.trim();
  countEl.textContent = altEl.value.length + ' characters';
  if (!sel) { snippetEl.textContent = '<img src="" alt="" title="" />'; return; }
  const names = precomputed || downloadNames();
  const rawTitle = titleEl.value.trim();
  const title = rawTitle || alt; // always emit title — fall back to alt
  const src = names.get(sel.id) || sel.name;
  const dims = sel.w && sel.h ? ` width="${sel.w}" height="${sel.h}"` : '';
  snippetEl.textContent = `<img src="${src}" alt="${escapeAttr(alt)}" title="${escapeAttr(title)}"${dims} />`;
}

// ---------- exports / downloads ----------
function buildHTML(names) {
  return items.map((i) => snippetFor(i, names)).join('\n');
}
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
  return `<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8" />\n<title>Alt text export — ${items.length} images (≤300KB each)</title>\n<style>body{font-family:sans-serif;max-width:900px;margin:32px auto;padding:0 16px}figure{border:1px solid #ddd;border-radius:8px;padding:12px}img{max-width:100%;height:auto}</style>\n</head>\n<body>\n<h1>Exported images (${items.length})</h1>\n${cards}\n<h2>HTML snippets</h2>\n<pre>${escapeHtml(buildHTML(names))}</pre>\n</body>\n</html>`;
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
    log(`Downloaded "${names.get(item.id)}" (${formatBytes(blob.size)}) with alt: "${item.alt || ''}"${item.title ? ` / title: "${item.title}"` : ''}.`);
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
  log('Building ZIP (images ≤300KB + HTML/CSV manifest)…');
  try {
    const Zip = await getJSZip();
    const zip = new Zip();
    for (const item of items) {
      const blob = await ensureBlob(item);
      zip.file(names.get(item.id) || item.name, blob);
    }
    zip.file('alt-text.html', buildManifest(names));
    zip.file('alt-text.csv', buildCSV(names));
    zip.file('alt-text.txt', items.map((i) => `${names.get(i.id)} | title: ${i.title || ''} | alt: ${i.alt || ''}`).join('\n'));
    const out = await zip.generateAsync({ type: 'blob' });
    downloadBlob(out, 'alt-text-images-300kb.zip');
    log(`Downloaded ZIP: ${items.length} images (each ≤300KB, named from alt text) + manifest.`);
  } catch (e) {
    log('ZIP failed (' + (e && e.message ? e.message : e) + ') — downloading files individually instead.');
    for (const item of items) {
      try {
        const blob = await ensureBlob(item);
        downloadBlob(blob, names.get(item.id) || item.name);
        await new Promise((r) => setTimeout(r, 400));
      } catch {}
    }
    downloadBlob(new Blob([buildManifest(names)], { type: 'text/html' }), 'alt-text.html');
  }
}

// ---------- events ----------
fileInput.addEventListener('change', () => { void addFiles(fileInput.files); });

drop.addEventListener('click', (e) => { if (e.target.closest('.qremove')) return; fileInput.click(); });
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('drag'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('drop', (e) => {
  const files = e.dataTransfer && e.dataTransfer.files;
  if (!files || !files.length) return;
  void addFiles(files);
});

document.getElementById('loadUrl').addEventListener('click', () => {
  const u = urlInput.value.trim();
  if (!u) return;
  urlInput.value = '';
  void addUrlItem(u);
});

clearBtn.addEventListener('click', () => {
  for (const i of items) if (i.objectUrl) { try { URL.revokeObjectURL(i.objectUrl); } catch {} }
  items = [];
  selectedId = null;
  fileInput.value = '';
  altEl.dataset.forId = '';
  titleEl.dataset.forId = '';
  render();
  log('Cleared. Choose images to begin.');
});

genBtn.addEventListener('click', generateAll);

altEl.addEventListener('input', () => {
  const sel = selectedItem();
  if (sel) sel.alt = altEl.value;
  updateSnippet();
  renderBatchTextOnly();
});
titleEl.addEventListener('input', () => {
  const sel = selectedItem();
  if (sel) sel.title = titleEl.value;
  updateSnippet();
  renderBatchTextOnly();
});

function renderBatchTextOnly() {
  const names = downloadNames();
  const rows = batchListEl.children;
  items.forEach((item, ix) => {
    const row = rows[ix];
    if (!row) return;
    const altDiv = row.querySelector('.balt');
    if (altDiv) {
      altDiv.textContent = item.alt ? `alt: ${item.alt}` : '(pending alt…)';
      altDiv.className = 'balt' + (item.alt ? '' : ' empty');
    }
    const titleDiv = row.querySelector('.btitle');
    if (titleDiv) {
      titleDiv.textContent = item.title ? `title: ${item.title}` : '(pending title…)';
      titleDiv.className = 'btitle' + (item.title ? '' : ' empty');
    }
    const metaDiv = row.querySelector('.bmeta');
    if (metaDiv) metaDiv.textContent = `↓ ${names.get(item.id)} · ${sizeLine(item)}`;
  });
  const sel = selectedItem();
  if (sel && sizeInfo) sizeInfo.textContent = sizeLine(sel) + ' · downloads as ' + (names.get(sel.id) || sel.name);
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
    catch { log('Copy failed — select and copy manually.'); }
    ta.remove();
  }
}

document.getElementById('copyAlt').addEventListener('click', () => copyText(altEl.value, 'Alt text copied to clipboard.'));
document.getElementById('copyTitle').addEventListener('click', () => copyText(titleEl.value, 'Title copied to clipboard.'));
// one-click inline copy buttons on the alt / title fields
function flashCopied(btn) {
  if (!btn) return;
  const orig = btn.textContent;
  btn.textContent = '✓';
  btn.classList.add('copied');
  setTimeout(() => { btn.textContent = orig; btn.classList.remove('copied'); }, 1200);
}
document.getElementById('copyAltInline').addEventListener('click', async () => {
  await copyText(altEl.value, 'Alt text copied to clipboard.');
  flashCopied(document.getElementById('copyAltInline'));
});
document.getElementById('copyTitleInline').addEventListener('click', async () => {
  await copyText(titleEl.value, 'Title copied to clipboard.');
  flashCopied(document.getElementById('copyTitleInline'));
});
document.getElementById('copyHtml').addEventListener('click', () => copyText(snippetEl.textContent, 'HTML snippet (alt + title) copied to clipboard.'));
document.getElementById('copyAllAlt').addEventListener('click', () => {
  if (!items.length) return log('Nothing to copy yet.');
  copyText(items.map((i) => i.alt).join('\n'), `Copied ${items.length} alt texts.`);
});
document.getElementById('copyAllTitle').addEventListener('click', () => {
  if (!items.length) return log('Nothing to copy yet.');
  copyText(items.map((i) => i.title || i.alt).join('\n'), `Copied ${items.length} titles.`);
});
document.getElementById('copyAllHtml').addEventListener('click', () => {
  if (!items.length) return log('Nothing to copy yet.');
  copyText(buildHTML(downloadNames()), `Copied ${items.length} HTML snippets.`);
});
document.getElementById('dlOne').addEventListener('click', () => {
  const sel = selectedItem();
  if (!sel) return log('Select an image first.');
  void downloadOne(sel);
});
document.getElementById('dlAll').addEventListener('click', () => { void downloadAllZip(); });
document.getElementById('dlManifest').addEventListener('click', () => {
  if (!items.length) return log('Nothing to export yet.');
  const names = downloadNames();
  downloadBlob(new Blob([buildManifest(names)], { type: 'text/html' }), 'alt-text.html');
  log('Downloaded manifest alt-text.html (filenames + alt + title).');
});

// theme toggle (dark mode)
(function initThemeToggle() {
  const btn = document.getElementById('themeToggle');
  if (!btn) return;
  const sync = () => {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    btn.textContent = dark ? '◑ Light' : '◐ Dark';
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

// scroll reveal
const io = new IntersectionObserver((entries) => {
  entries.forEach((en) => { if (en.isIntersecting) en.target.classList.add('visible'); });
}, { threshold: 0.1 });
document.querySelectorAll('.reveal').forEach((el) => io.observe(el));

render();
