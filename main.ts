// pumpfun-top5-forecast-01 entry: a horizon-averaged Monte Carlo forecaster.
//
// One static forecast is scored at seven answer reads (task.json reads), so the
// model simulates every read, then averages each read's place and size
// distributions into the single forecast the scorer reads. Existing coins are
// the last mark's largest 140; coins born after the cut cannot be named, so
// their place probability is left unnamed (the scorer shares it among the
// answer chart's unnamed coins), and their size contribution is simulated as a
// Pareto tail of the competitive newcomers.
//
// Node 22, standard library only, no network. Deterministic: fixed seed, no
// clock and no Math.random.

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DIAG = process.env.PUMP_DIAG === '1';

interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string;
  venue: string;
  created_at: string;
  complete: boolean;
  mayhem: boolean;
}
interface Mark {
  at: string;
  ok: boolean;
  rows: Row[];
}
interface Task {
  places: number;
  chart_size: number;
  reads: string[];
  history_until: string;
  cadence_minutes?: number;
  output: string;
  size: { edges: number[] };
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const marks = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '')
  .map((l) => JSON.parse(l) as Mark)
  .filter((m) => m.ok === true && Array.isArray(m.rows));
if (marks.length === 0) throw new Error('no ok mark');
const cut = marks[marks.length - 1]!;
const prev = marks.length > 1 ? marks[marks.length - 2]! : null;
const baseMs = Date.parse(cut.at);
const cadence = task.cadence_minutes ?? 10;
const places = task.places;
const edges = task.size.edges;
const binWidth = edges.length > 1 ? edges[1]! - edges[0]! : 0.05;

// The last mark's largest coins, in rank order.
const MAXC = 140;
const chart = [...cut.rows].sort((a, b) => a.rank - b.rank).slice(0, MAXC);
const prevByMint = new Map<string, Row>();
if (prev) for (const r of prev.rows) prevByMint.set(r.mint, r);

const log10Sol = (lamports: string): number => Math.log10(Number(BigInt(lamports))) - 9;

interface Coin {
  mint: string;
  logcap0: number;
  ageMin: number;
  growth: number;
  pool: number;
  complete: number;
  mayhem: number;
}
const coins: Coin[] = chart.map((r) => {
  const p = prevByMint.get(r.mint);
  const growth = p ? log10Sol(r.market_cap_lamports) - log10Sol(p.market_cap_lamports) : 0;
  const ageMin = (baseMs - Date.parse(r.created_at)) / 60000;
  return {
    mint: r.mint,
    logcap0: log10Sol(r.market_cap_lamports),
    ageMin: Number.isFinite(ageMin) ? ageMin : 60,
    growth: Number.isFinite(growth) ? Math.max(-0.8, Math.min(0.8, growth)) : 0,
    pool: r.venue === 'pool' ? 1 : 0,
    complete: r.complete ? 1 : 0,
    mayhem: r.mayhem ? 1 : 0,
  };
});
if (coins.length < places) throw new Error(`only ${coins.length} coins at the cut`);

const horizons = task.reads.map((r) => Math.max(cadence, (Date.parse(r) - baseMs) / 60000));
const reads = horizons.length;

// Seeded PRNG (mulberry32) and a cached Box-Muller normal.
let seed = 0x9e3779b9;
function rnd(): number {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
let spare: number | null = null;
function gauss(): number {
  if (spare !== null) {
    const v = spare;
    spare = null;
    return v;
  }
  let u = 0;
  let v = 0;
  while (u <= 1e-12) u = rnd();
  v = rnd();
  const mag = Math.sqrt(-2 * Math.log(u));
  spare = mag * Math.sin(2 * Math.PI * v);
  return mag * Math.cos(2 * Math.PI * v);
}

const M = 12000;
const K = 8; // simulated competitive newcomers per read
const BIRTH = 5.5; // competitive priced newcomers per minute
const XMIN = 100; // SOL
const ALPHA = 1.85; // Pareto tail of newcomer caps at a read
const RHO = 0.25;

// counts[i * places + k] = simulations in which existing coin i is at place k,
// summed over the seven reads.
const counts = new Float64Array(coins.length * places);
const sizeCounts: Float64Array[] = Array.from({ length: places }, () => new Float64Array(edges.length + 1));
const top5Count = new Float64Array(coins.length);
const unnamed = new Float64Array(places);

const cap = new Float64Array(coins.length);
const drift = new Float64Array(coins.length);
const topVal = new Float64Array(places);
const topIdx = new Int32Array(places); // -1 = newcomer
const newVal = new Float64Array(K);

const floorLog = 1.4;
const ceilLog = 6.5;

for (let sim = 0; sim < M; sim++) {
  // One market regime for the whole path, drawn per simulation.
  const u = rnd();
  let mu: number;
  let sig: number;
  if (u < 0.18) {
    mu = 0.004;
    sig = 0.1;
  } else if (u < 0.72) {
    mu = -0.012;
    sig = 0.14;
  } else {
    mu = -0.045;
    sig = 0.2;
  }
  for (let i = 0; i < coins.length; i++) {
    const c = coins[i]!;
    let d = mu + c.growth * 0.012;
    if (c.ageMin < 15) d += 0.006;
    else if (c.ageMin > 40) d -= 0.004;
    if (c.pool) d += 0.002;
    drift[i] = d;
    cap[i] = c.logcap0;
  }
  let prevH = 0;
  for (let r = 0; r < reads; r++) {
    const h = horizons[r]!;
    const steps = (h - prevH) / cadence;
    prevH = h;
    const zc = gauss();
    const common = RHO * zc;
    const idio = Math.sqrt(1 - RHO * RHO);
    const sd = sig * Math.sqrt(steps);
    for (let i = 0; i < coins.length; i++) {
      let v = cap[i]! + drift[i]! * steps + sd * (common + idio * gauss());
      if (v < floorLog) v = floorLog;
      else if (v > ceilLog) v = ceilLog;
      cap[i] = v;
    }
    // Pareto order statistics of the competitive newcomers at this read.
    const nEff = BIRTH * h;
    let s = 0;
    for (let k = 0; k < K; k++) {
      s += -Math.log(1 - rnd() * 0.999999);
      newVal[k] = Math.min(ceilLog, Math.log10(XMIN) + Math.log10(nEff / s) / ALPHA);
    }
    // Top places at this read: insertion into tiny sorted arrays.
    for (let k = 0; k < places; k++) {
      topVal[k] = -Infinity;
      topIdx[k] = -1;
    }
    for (let i = 0; i < coins.length; i++) {
      const v = cap[i]!;
      if (v <= topVal[places - 1]!) continue;
      let j = places - 1;
      while (j > 0 && topVal[j - 1]! < v) {
        topVal[j] = topVal[j - 1]!;
        topIdx[j] = topIdx[j - 1]!;
        j--;
      }
      topVal[j] = v;
      topIdx[j] = i;
    }
    for (let k = 0; k < K; k++) {
      const v = newVal[k]!;
      if (v <= topVal[places - 1]!) continue;
      let j = places - 1;
      while (j > 0 && topVal[j - 1]! < v) {
        topVal[j] = topVal[j - 1]!;
        topIdx[j] = topIdx[j - 1]!;
        j--;
      }
      topVal[j] = v;
      topIdx[j] = -1;
    }
    for (let k = 0; k < places; k++) {
      const idx = topIdx[k]!;
      if (idx >= 0) counts[idx * places + k]! += 1;
      else unnamed[k]! += 1;
      const v = topVal[k]!;
      let b = Math.floor((v - edges[0]!) / binWidth) + 1;
      if (b < 0) b = 0;
      else if (b > edges.length) b = edges.length;
      sizeCounts[k]![b]! += 1;
    }
    // Tie-break tally: the top 5 coins of this read (a newcomer counts for none).
    for (let k = 0; k < places; k++) {
      const idx = topIdx[k]!;
      if (idx >= 0) top5Count[idx]! += 1 / (places * reads * M);
    }
  }
}

const total = M * reads;
const micro = 1_000_000;
const floor6 = (p: number): number => Math.floor(p * micro) / micro;

const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let k = 0; k < places; k++) {
  const rows: Array<{ i: number; p: number }> = [];
  for (let i = 0; i < coins.length; i++) {
    const p = floor6(counts[i * places + k]! / total);
    if (p >= 1e-4) rows.push({ i, p });
  }
  rows.sort((a, b) => b.p - a.p);
  for (const row of rows.slice(0, 100)) {
    probabilities.push({ mint: coins[row.i]!.mint, place: k + 1, p: row.p });
  }
}

const sizes: Array<{ place: number; bin: number; p: number }> = [];
for (let k = 0; k < places; k++) {
  for (let b = 0; b <= edges.length; b++) {
    const p = floor6(sizeCounts[k]![b]! / total);
    if (p > 0) sizes.push({ place: k + 1, bin: b, p });
  }
}

// Tie-break top5: the five coins with the highest average top-5 membership,
// ordered by their most likely place.
const tie: Array<{ i: number; score: number; place: number }> = [];
for (let i = 0; i < coins.length; i++) {
  let place = 1;
  let best = -1;
  for (let k = 0; k < places; k++) {
    const p = counts[i * places + k]!;
    if (p > best) {
      best = p;
      place = k + 1;
    }
  }
  tie.push({ i, score: top5Count[i]! + best / total / 1000, place });
}
tie.sort((a, b) => b.score - a.score || a.place - b.place);
const top5: string[] = [];
for (const t of tie) {
  if (top5.length >= places) break;
  top5.push(coins[t.i]!.mint);
}

writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);

if (DIAG) {
  const out: string[] = [];
  out.push(`coins=${coins.length} reads=${reads} horizons=${horizons.map((h) => h.toFixed(0)).join(',')} M=${M}`);
  for (let k = 0; k < places; k++) {
    out.push(`place ${k + 1}: unnamed=${(unnamed[k]! / total).toFixed(3)} named=${(1 - unnamed[k]! / total).toFixed(3)}`);
  }
  for (let i = 0; i < Math.min(12, coins.length); i++) {
    const ps = [];
    for (let k = 0; k < places; k++) ps.push((counts[i * places + k]! / total).toFixed(3));
    out.push(`coin ${i} ${coins[i]!.mint.slice(0, 6)} logcap=${coins[i]!.logcap0.toFixed(2)} age=${coins[i]!.ageMin.toFixed(0)} p=[${ps.join(',')}]`);
  }
  for (const mint of [
    '4Dgjitjn8br3LcdcBnkc9tYJ8d1cnCCJd2nsVgzZpump',
    'BuaseKYxFH5Bcz6uoTU2JiX32hRGsGfGLTgamoWpump',
  ]) {
    const i = coins.findIndex((c) => c.mint === mint);
    out.push(`known ${mint.slice(0, 6)} index=${i} p=[${i >= 0 ? Array.from({ length: places }, (_, k) => (counts[i * places + k]! / total).toFixed(3)).join(',') : 'absent'}]`);
  }
  process.stderr.write(out.join('\n') + '\n');
}
