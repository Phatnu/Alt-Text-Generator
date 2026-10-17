/**
 * Generates the site favicon set from a single geometry definition.
 *
 *   favicon.svg            <- modern browsers (crisp at any size)
 *   favicon.ico            <- 16 / 32 / 48 px, PNG-encoded entries
 *   favicon-32.png         <- raster fallback / manifest use
 *   apple-touch-icon.png   <- 180 px, full-bleed tile (iOS masks the corners)
 *
 * The mark is the "ALT" mono tag: an ink tile with monospace letterforms.
 *
 *   node tools/make-favicon.mjs
 *
 * No dependencies: the rasteriser, PNG encoder and ICO container are all
 * implemented below so the icons can be regenerated anywhere Node runs.
 * Change COLORS or the geometry constants to re-roll the mark.
 */
import { deflateSync, crc32 as zlibCrc32 } from "node:zlib";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/* ── Design tokens ────────────────────────────────────────────────── */
const COLORS = {
  tile: "#12171A", // --ink (light theme)
  letter: "#6FD3AD", // --accent (dark theme), reads on the dark tile
};

const CANVAS = 32; // design grid, everything below is in these units
const TILE_RADIUS = 7;
const STROKE = 2.4;
const CAP_TOP = 10.5;
const CAP_BOTTOM = 21.5;
const LETTER_WIDTHS = [7.4, 5.2, 7.0]; // A, L, T
const LETTER_GAP = 2.2;
const CROSSBAR_AT = 0.5; // vertical position of A's bar, 0 = top, 1 = bottom

/* ── Geometry (shared by the SVG and the rasteriser) ──────────────── */
/** Returns the strokes of "ALT" as [x1, y1, x2, y2] segments in design units. */
export function letterSegments() {
  const total =
    LETTER_WIDTHS.reduce((sum, w) => sum + w, 0) + LETTER_GAP * (LETTER_WIDTHS.length - 1);
  let x = (CANVAS - total) / 2;
  const [aW, lW, tW] = LETTER_WIDTHS;
  const segs = [];

  // A
  const apex = x + aW / 2;
  segs.push([x, CAP_BOTTOM, apex, CAP_TOP]);
  segs.push([apex, CAP_TOP, x + aW, CAP_BOTTOM]);
  const barY = CAP_TOP + (CAP_BOTTOM - CAP_TOP) * CROSSBAR_AT;
  segs.push([x + aW * 0.12, barY, x + aW * 0.88, barY]);
  x += aW + LETTER_GAP;

  // L
  segs.push([x, CAP_TOP, x, CAP_BOTTOM]);
  segs.push([x, CAP_BOTTOM, x + lW, CAP_BOTTOM]);
  x += lW + LETTER_GAP;

  // T
  segs.push([x, CAP_TOP, x + tW, CAP_TOP]);
  segs.push([x + tW / 2, CAP_TOP, x + tW / 2, CAP_BOTTOM]);

  return segs;
}

/** Scales letter strokes about the canvas centre (used for the iOS icon). */
function scaleSegments(segs, scale) {
  const c = CANVAS / 2;
  return segs.map(([x1, y1, x2, y2]) =>
    [c + (x1 - c) * scale, c + (y1 - c) * scale, c + (x2 - c) * scale, c + (y2 - c) * scale],
  );
}

/* ── SVG ──────────────────────────────────────────────────────────── */
export function buildSvg() {
  const segs = letterSegments();
  const d = (seg) => {
    const [x1, y1, x2, y2] = seg;
    return `M${n(x1)} ${n(y1)}L${n(x2)} ${n(y2)}`;
  };
  const path = (list) => list.map(d).join("");

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${CANVAS} ${CANVAS}" role="img" aria-label="Alt Text Studio">
  <rect width="${CANVAS}" height="${CANVAS}" rx="${TILE_RADIUS}" fill="${COLORS.tile}"/>
  <g fill="none" stroke="${COLORS.letter}" stroke-width="${STROKE}" stroke-linecap="round" stroke-linejoin="round">
    <path d="${path(segs.slice(0, 3))}"/>
    <path d="${path(segs.slice(3, 5))}"/>
    <path d="${path(segs.slice(5))}"/>
  </g>
</svg>
`;

  function n(v) {
    return String(Number(v.toFixed(3)));
  }
}

/* ── Rasteriser ───────────────────────────────────────────────────── */
function insideRoundedTile(x, y, radius) {
  if (radius <= 0) return true;
  const half = CANVAS / 2;
  const qx = Math.abs(x - half) - (half - radius);
  const qy = Math.abs(y - half) - (half - radius);
  const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0));
  const inside = Math.min(Math.max(qx, qy), 0);
  return outside + inside - radius <= 0;
}

function distanceToSegment(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lenSq = dx * dx + dy * dy;
  let t = lenSq === 0 ? 0 : ((px - x1) * dx + (py - y1) * dy) / lenSq;
  t = Math.min(Math.max(t, 0), 1);
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

function distanceToLetters(x, y, segs) {
  let best = Infinity;
  for (const [x1, y1, x2, y2] of segs) {
    const d = distanceToSegment(x, y, x1, y1, x2, y2);
    if (d < best) best = d;
  }
  return best;
}

/** Renders the mark to raw RGBA at `size` px. Supersampled for edge AA. */
export function renderRgba({ size, radius = TILE_RADIUS, letterScale = 1 }) {
  const segs = letterScale === 1 ? letterSegments() : scaleSegments(letterSegments(), letterScale);
  const halfStroke = (STROKE * letterScale) / 2;
  const [tr, tg, tb] = hexToRgb(COLORS.tile);
  const [lr, lg, lb] = hexToRgb(COLORS.letter);
  const step = CANVAS / size;
  const ss = size <= 64 ? 8 : 4; // samples per axis, per pixel
  const samples = ss * ss;
  const out = Buffer.alloc(size * size * 4);

  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      let tileHits = 0;
      let letterHits = 0;

      for (let sy = 0; sy < ss; sy++) {
        for (let sx = 0; sx < ss; sx++) {
          const x = (px + (sx + 0.5) / ss) * step;
          const y = (py + (sy + 0.5) / ss) * step;
          if (insideRoundedTile(x, y, radius)) tileHits++;
          if (letterHits < samples && distanceToLetters(x, y, segs) <= halfStroke) letterHits++;
        }
      }

      const tileA = tileHits / samples;
      const letterA = letterHits / samples;
      // Letters paint over the tile.
      const alpha = letterA + tileA * (1 - letterA);
      const i = (py * size + px) * 4;
      if (alpha > 0) {
        const pR = lr * letterA + tr * tileA * (1 - letterA);
        const pG = lg * letterA + tg * tileA * (1 - letterA);
        const pB = lb * letterA + tb * tileA * (1 - letterA);
        out[i] = Math.round(pR / alpha);
        out[i + 1] = Math.round(pG / alpha);
        out[i + 2] = Math.round(pB / alpha);
      }
      out[i + 3] = Math.round(alpha * 255);
    }
  }

  return out;
}

function hexToRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/* ── PNG encoder ──────────────────────────────────────────────────── */
const CRC_TABLE = (() => {
  const table = new Int32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[i] = c;
  }
  return table;
})();

function crc32(buf) {
  if (typeof zlibCrc32 === "function") return zlibCrc32(buf) >>> 0;
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const head = Buffer.alloc(8);
  head.writeUInt32BE(data.length, 0);
  head.write(type, 4, "ascii");
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), data])), 0);
  return Buffer.concat([head, data, crc]);
}

export function encodePng(size, rgba) {
  const stride = size * 4;
  const raw = Buffer.alloc((stride + 1) * size);
  for (let y = 0; y < size; y++) {
    raw[y * (stride + 1)] = 0; // filter: none
    rgba.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // colour type: RGBA
  // 10..12: compression, filter, interlace = 0

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", deflateSync(raw, { level: 9 })),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

/* ── ICO container (PNG-compressed entries) ───────────────────────── */
export function encodeIco(images) {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0); // reserved
  header.writeUInt16LE(1, 2); // type: icon
  header.writeUInt16LE(images.length, 4);

  const dir = Buffer.alloc(images.length * 16);
  let offset = header.length + dir.length;

  images.forEach(({ size, png }, i) => {
    const e = i * 16;
    dir[e] = size >= 256 ? 0 : size; // 0 means 256
    dir[e + 1] = size >= 256 ? 0 : size;
    dir[e + 2] = 0; // palette size
    dir[e + 3] = 0; // reserved
    dir.writeUInt16LE(1, e + 4); // colour planes
    dir.writeUInt16LE(32, e + 6); // bits per pixel
    dir.writeUInt32LE(png.length, e + 8); // size of the image payload
    dir.writeUInt32LE(offset, e + 12);
    offset += png.length;
  });

  return Buffer.concat([header, dir, ...images.map((i) => i.png)]);
}

/* ── Entry point ──────────────────────────────────────────────────── */
export function buildAll() {
  const icoSizes = [16, 32, 48];
  return {
    "favicon.svg": Buffer.from(buildSvg(), "utf8"),
    "favicon.ico": encodeIco(
      icoSizes.map((size) => ({ size, png: encodePng(size, renderRgba({ size })) })),
    ),
    "favicon-32.png": encodePng(32, renderRgba({ size: 32 })),
    // iOS masks the icon itself, so this one is full-bleed with roomier margins.
    "apple-touch-icon.png": encodePng(
      180,
      renderRgba({ size: 180, radius: 0, letterScale: 0.72 }),
    ),
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  for (const [name, data] of Object.entries(buildAll())) {
    writeFileSync(join(ROOT, name), data);
    console.log(`${name.padEnd(20)} ${String(data.length).padStart(7)} bytes`);
  }
}
