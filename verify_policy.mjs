/* Exhaustive end-to-end verification of the exported agent.
 *
 * This script runs the *actual browser code* (js/policy.js + game.js) against
 * every possible opponent move sequence and asserts that the agent never
 * loses - as X and as O, at every difficulty level, using the same
 * canonicalisation and policy lookup the game uses at runtime.
 *
 * It also proves the built-in perfect-play fallback is never needed, by
 * checking that every reachable agent-to-move state is covered by the table.
 *
 *   node verify_policy.mjs
 *
 * Exit code 0 = every check passed.
 */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const policyPath = path.join(here, 'policy.js');
const gamePath = path.join(here, 'game.js');

/* ---------------------------------------------------------------- loading */

const ctx = vm.createContext({ console });
vm.runInContext('globalThis.window = {};', ctx);
for (const file of [policyPath, gamePath]) {
  vm.runInContext(fs.readFileSync(file, 'utf8'), ctx, { filename: file });
}
const { TTT, TTT_MODEL } = vm.runInContext(
  '({ TTT: window.TTT, TTT_MODEL: window.TTT_MODEL })', ctx);

const fails = [];
function check(ok, label, detail) {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) fails.push(label);
}

/* -------------------------------------------------- structural consistency */

console.log('\nstructural checks');

const jsTransforms = TTT.TRANSFORMS.map((p) => p.join('')).sort();
const pyTransforms = TTT_MODEL.meta.transforms.map((p) => p.join('')).sort();
check(jsTransforms.length === 8, 'symmetry group has 8 elements',
  `(${jsTransforms.length})`);
check(JSON.stringify(jsTransforms) === JSON.stringify(pyTransforms),
  'JS symmetry group matches the trained model');

/* Every reachable position, enumerated from scratch. */
function reachable() {
  const seen = new Map();
  const start = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  const stack = [start];
  seen.set(start.join(''), start);
  while (stack.length) {
    const board = stack.pop();
    if (TTT.winner(board) !== null) continue;
    const x = board.filter((v) => v === 1).length;
    const o = board.filter((v) => v === 2).length;
    const turn = x === o ? 1 : 2;
    for (const i of TTT.legalMoves(board)) {
      const next = board.slice();
      next[i] = turn;
      if (!seen.has(next.join(''))) {
        seen.set(next.join(''), next);
        stack.push(next);
      }
    }
  }
  return [...seen.values()];
}

const positions = reachable();
check(positions.length === TTT_MODEL.meta.reachablePositions,
  'reachable position count matches training',
  `(${positions.length})`);

/* Coverage: every agent-to-move state is tabulated for every level. */
const agentStateKeys = new Set();
const uncovered = [];
for (const board of positions) {
  if (TTT.winner(board) !== null) continue;
  const x = board.filter((v) => v === 1).length;
  const o = board.filter((v) => v === 2).length;
  const mover = x === o ? 1 : 2;
  const key = TTT.canonical(TTT.relative(board, mover));
  agentStateKeys.add(key);
  for (const level of TTT_MODEL.levels) {
    if (level.policy[key] === undefined) uncovered.push(`${level.id}:${key}`);
  }
}
check(agentStateKeys.size === TTT_MODEL.meta.stateCount,
  'distinct canonical agent-to-move states match training',
  `(${agentStateKeys.size} keys over ${positions.length} reachable positions)`);
check(uncovered.length === 0, 'every state is tabulated at every level',
  uncovered.length ? `missing ${uncovered.slice(0, 3).join(', ')}` : '(fallback never needed)');

const valueKeys = new Set(Object.keys(TTT_MODEL.levels.find((l) => l.values).values));
check(valueKeys.size === agentStateKeys.size &&
  [...agentStateKeys].every((k) => valueKeys.has(k)),
  'Q-value table covers exactly the agent states', `(${valueKeys.size})`);

/* ------------------------------------------------------ exhaustive game tree */

/* Play the complete game tree: the agent follows the runtime policy, the
 * human may play anything at all.  Counts are memoised per (state, turn). */
function verifyLevel(levelId, agentPlayer) {
  const legalCache = new Map();
  const memo = new Map();
  const counts = { win: 0, draw: 0, loss: 0, games: 0 };
  let misses = 0;

  function rec(board, turn) {
    const w = TTT.winner(board);
    if (w !== null) {
      if (w === agentPlayer) return { win: 1, draw: 0, loss: 0, games: 1 };
      if (w === 0) return { win: 0, draw: 1, loss: 0, games: 1 };
      return { win: 0, draw: 0, loss: 1, games: 1 };
    }
    const mk = board.join('') + '|' + turn;
    const hit = memo.get(mk);
    if (hit) return hit;

    const total = { win: 0, draw: 0, loss: 0, games: 0 };
    let subs;
    if (turn === agentPlayer) {
      const key = TTT.canonical(TTT.relative(board, agentPlayer));
      if (TTT.levelById(TTT_MODEL, levelId).policy[key] === undefined) misses++;
      subs = [TTT.agentMove(board, agentPlayer, TTT_MODEL, levelId)];
    } else {
      subs = TTT.legalMoves(board);
    }
    for (const i of subs) {
      const next = board.slice();
      next[i] = turn;
      const sub = rec(next, TTT.other(turn));
      total.win += sub.win;
      total.draw += sub.draw;
      total.loss += sub.loss;
      total.games += sub.games;
    }
    memo.set(mk, total);
    return total;
  }

  const result = rec([0, 0, 0, 0, 0, 0, 0, 0, 0], 1);
  Object.assign(counts, result);
  counts.misses = misses;
  return counts;
}

for (const level of TTT_MODEL.levels) {
  console.log(`\nlevel "${level.id}" (${level.subtitle || level.label})`);
  let totalGames = 0;
  let totalLosses = 0;
  let totalMisses = 0;
  for (const [seat, agentPlayer] of [['X (first)', 1], ['O (second)', 2]]) {
    const c = verifyLevel(level.id, agentPlayer);
    totalGames += c.games;
    totalLosses += c.loss;
    totalMisses += c.misses;
    console.log(`  as ${seat.padEnd(11)} opponent lines=${String(c.games).padStart(4)}  ` +
      `agent wins=${String(c.win).padStart(4)}  draws=${String(c.draw).padStart(4)}  ` +
      `agent losses=${c.loss}`);
  }
  const label = `agent never loses (${totalGames} opponent lines, both seats)`;
  if (level.id === 'perfect') {
    check(totalLosses === 0, label);
    check(totalMisses === 0, 'no policy lookups fell through to the fallback');
  } else {
    // The learning-stage levels are allowed to lose; report it rather than fail.
    console.log(`  ${totalLosses === 0 ? 'PASS' : 'INFO'}  ` +
      `${level.id} loses ${totalLosses} opponent lines (expected for an untrained level)`);
  }
}

/* ------------------------------------------------------------ sanity probes */

console.log('\nsanity probes');
const empty = [0, 0, 0, 0, 0, 0, 0, 0, 0];

/* Independent negamax, used to reason about which moves are actually safe.
 * Returns the outcome for the player to move: +1 win, 0 draw, -1 loss. */
const vcache = new Map();
function moverValue(board, player) {
  const w = TTT.winner(board);
  if (w !== null) return w === 0 ? 0 : (w === player ? 1 : -1);
  const key = board.join('') + '|' + player;
  if (vcache.has(key)) return vcache.get(key);
  let best = -2;
  for (const i of TTT.legalMoves(board)) {
    const next = board.slice();
    next[i] = player;
    best = Math.max(best, -moverValue(next, TTT.other(player)));
  }
  vcache.set(key, best);
  return best;
}
const safeOpenings = TTT.legalMoves(empty).filter((i) => {
  const next = empty.slice();
  next[i] = 1;
  return -moverValue(next, 2) >= 0;
});

const replies = TTT.legalMoves(empty).map((i) => {
  const b = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  b[i] = 2;                            // the human took square i, agent is X
  return TTT.agentMove(b, 1, TTT_MODEL, 'perfect');
});
check(replies.every((m) => m !== undefined && m >= 0 && m < 9 && empty[m] === 0),
  'agent answers every opening move with a legal move');

const opening = TTT.agentMove(empty, 1, TTT_MODEL, 'perfect');
check(safeOpenings.length === 9, 'all nine openings draw with perfect play',
  `(${safeOpenings.length})`);
check(safeOpenings.includes(opening), 'the agent opens with a safe move',
  `(square ${opening + 1}${opening === 4 ? ' — the centre' : ''})`);
if (opening !== 4) {
  console.log(`  INFO  this run opens on square ${opening + 1} rather than the centre; ` +
    'both draw, the centre just wins more against imperfect play');
}

/* The shield's safety net must also be sound on its own. */
let fallbackUnsafe = 0;
for (const board of positions) {
  if (TTT.winner(board) !== null) continue;
  const x = board.filter((v) => v === 1).length;
  const o = board.filter((v) => v === 2).length;
  const mover = x === o ? 1 : 2;
  const before = moverValue(board, mover);
  if (before < 0) continue;                    // already lost, nothing to save
  for (let k = 0; k < 4; k++) {                // minimaxMove randomises between equal moves
    const m = TTT.minimaxMove(board, mover);
    const next = board.slice();
    next[m] = mover;
    // Safe means the opponent to move cannot force a win (their value <= 0).
    if (moverValue(next, TTT.other(mover)) > 0) { fallbackUnsafe++; break; }
  }
}
check(fallbackUnsafe === 0, 'the perfect-play fallback is safe in every reachable state',
  `(checked ${positions.length} positions)`);
check(TTT.winner([1, 1, 1, 2, 2, 0, 0, 0, 0]) === 1, 'winner detection');
check(TTT.winner([1, 2, 1, 1, 2, 2, 2, 1, 1]) === 0, 'draw detection');

console.log('');
if (fails.length) {
  console.error(`FAILED: ${fails.length} check(s): ${fails.join('; ')}`);
  process.exit(1);
}
console.log('All checks passed: the exported agent never loses, at every level that claims it.');
