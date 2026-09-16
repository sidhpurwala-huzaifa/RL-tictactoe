/* Headless integration test for index.html.
 *
 * Loads the real page in jsdom, clicks the real squares and reads the real
 * scoreboard - so it exercises policy.js, game.js and the UI wiring exactly as
 * a browser would.  It plays games as X and as O at every level and asserts
 * that the Perfect agent is never beaten.
 *
 * Needs jsdom (not a runtime dependency of the game):
 *
 *   npm install jsdom
 *   node ui_test.mjs            # 18 games per seat at Perfect (~4 minutes)
 *   GAMES=4 node ui_test.mjs    # quick smoke run
 *
 * If jsdom lives somewhere else (it is only a test dependency), point at it:
 *
 *   JSDOM_PATH=/path/to/node_modules/jsdom/lib/api.js node ui_test.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const PERFECT_GAMES = Number(process.env.GAMES || 18);
const OTHER_GAMES = Math.max(2, Math.round(PERFECT_GAMES / 3));

let JSDOM;
try {
  ({ JSDOM } = await import(process.env.JSDOM_PATH || 'jsdom'));
} catch {
  console.error('jsdom is required for this test:  npm install jsdom');
  console.error('(or set JSDOM_PATH to a jsdom install)');
  process.exit(2);
}

const dom = new JSDOM(fs.readFileSync(path.join(here, 'index.html'), 'utf8'), {
  url: 'file://' + path.join(here, 'index.html'),
  runScripts: 'dangerously',
  resources: 'usable',
  pretendToBeVisual: true,
});
const { window } = dom;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const errors = [];
window.addEventListener('error', (e) => errors.push(String(e.message)));
await new Promise((res) => window.addEventListener('load', res));
await sleep(300);

const doc = window.document;
const $ = (s) => doc.querySelector(s);
const cells = [...doc.querySelectorAll('.cell')];
const fails = [];
const check = (ok, label, extra = '') => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${extra ? '  ' + extra : ''}`);
  if (!ok) fails.push(label);
};
const click = (el) => el.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
const visibleText = () => [...doc.body.children]
  .filter((e) => !['SCRIPT', 'STYLE'].includes(e.tagName))
  .map((e) => e.textContent).join(' ');
const score = () => [+$('#scoreYou').textContent, +$('#scoreDraw').textContent,
  +$('#scoreAgent').textContent];
const setLevel = (id) => {
  const b = doc.querySelector(`#levels button[data-level="${id}"]`);
  if (b.getAttribute('aria-pressed') !== 'true') click(b);
};
const setSide = (s) => {
  const b = doc.querySelector(`#sides button[data-side="${s}"]`);
  if (b.getAttribute('aria-pressed') !== 'true') click(b);
};
const isOver = () => /win|wins|draw/i.test($('#statusText').textContent);

console.log('\npage boot');
check(!!window.TTT_MODEL && !!window.TTT, 'policy.js + game.js loaded');
check(cells.length === 9, 'nine squares rendered');
check(doc.querySelectorAll('#levels button').length === 3, 'three opponent levels');
check(!/undefined|NaN/.test(visibleText()), 'no undefined/NaN in visible page text');
check($('#badge').textContent.includes('verified never loses'), 'verification badge present');

console.log('\nopening behaviour');
setLevel('perfect');
setSide(2);                              // human is O, so the agent (X) opens
click($('#newGame'));
await sleep(700);
check(cells[4].querySelector('.glyph') !== null, 'agent opens in the centre as X');
check(cells.filter((c) => c.querySelector('.glyph')).length === 1, 'agent played exactly one mark');
check($('#statusText').textContent.includes('Your move'), 'control passes to the human');
check(!$('#thinking').classList.contains('on'), 'thinking indicator clears after the move');

setSide(1);                              // human is X, so the human opens
click($('#newGame'));
await sleep(200);
check(cells.every((c) => c.querySelector('.glyph') === null), 'board empty when the human opens');
check(cells.filter((c) => !c.disabled).length === 9, 'all squares clickable on the human turn');
click(cells[0]);
await sleep(700);
check(cells[0].querySelector('.glyph') !== null, 'human click places a mark');
check(cells.filter((c) => c.querySelector('.glyph')).length === 2, 'agent replied');

console.log('\nmove-value overlay');
const empty = [0, 0, 0, 0, 0, 0, 0, 0, 0];
const vals = window.TTT.agentValues(empty, 1, window.TTT_MODEL, 'perfect');
check(vals && Math.max(...vals) === vals[4],
  'learned values prefer the centre on an empty board',
  `(centre ${vals[4].toFixed(3)} vs best corner ${Math.max(vals[0], vals[2], vals[6], vals[8]).toFixed(3)})`);
setSide(2);
click($('#toggleHeat'));
click($('#newGame'));
await sleep(120);                        // agent is thinking about its opening move
check(cells.filter((c) => c.classList.contains('heat-on')).length === 9,
  'values shown for every square while the agent is to move');
await sleep(700);
setLevel('novice');
await sleep(60);
click($('#newGame'));
await sleep(60);
check(cells.every((c) => !c.classList.contains('heat-on')),
  'overlay hidden for levels with no values');
click($('#toggleHeat'));

console.log('\nplaying through the real UI (clicking squares, reading the scoreboard)');
const rng = { s: 987654321, r() { this.s = (this.s * 1103515245 + 12345) % 2147483648; return this.s / 2147483648; } };
async function playGame() {
  click($('#newGame'));                  // deal a fresh board for every game
  await sleep(500);
  const before = score();
  for (let step = 0; step < 14 && !isOver(); step++) {
    const open = cells.filter((c) => !c.disabled);
    if (open.length) click(open[Math.floor(rng.r() * open.length)]);
    await sleep(500);
  }
  await sleep(650);
  const after = score();
  return { you: after[0] - before[0], draw: after[1] - before[1], agent: after[2] - before[2], ended: isOver() };
}

for (const level of ['perfect', 'medium', 'novice']) {
  for (const side of [1, 2]) {
    setLevel(level);
    setSide(side);
    click($('#resetScore'));
    click($('#newGame'));
    await sleep(650);
    const N = level === 'perfect' ? PERFECT_GAMES : OTHER_GAMES;
    let you = 0, agent = 0, draw = 0, unfinished = 0;
    for (let g = 0; g < N; g++) {
      const r = await playGame();
      if (!r.ended) unfinished++;
      else if (r.you) you++;
      else if (r.agent) agent++;
      else draw++;
    }
    const tag = `${level} as ${side === 1 ? 'X' : 'O'}`;
    console.log(`  ${tag.padEnd(22)} ${N} games -> you ${you}, agent ${agent}, draws ${draw}` +
      (unfinished ? `, UNFINISHED ${unfinished}` : ''));
    check(unfinished === 0, `every game reaches a result (${tag})`);
    if (level === 'perfect') {
      check(you === 0, `human never beats the verified agent (${tag})`);
      check(agent + draw === N, `verified agent wins or draws every game (${tag})`);
    }
  }
}

console.log('\nagent vs itself demo');
setLevel('perfect');
setSide(1);
click($('#toggleAuto'));
check($('#toggleAuto').getAttribute('aria-pressed') === 'true', 'autoplay toggle engages');
const seen = new Set();
for (let i = 0; i < 60; i++) {
  const t = $('#statusText').textContent;
  seen.add(t.trim());
  if (t.includes('Draw — both sides')) seen.add('DRAW');
  if (t.includes('Game over')) seen.add('NONDRAW');
  await sleep(250);
}
console.log(`  status texts seen: ${[...seen].join(' | ')}`);
check(seen.has('DRAW'), 'perfect self-play produces drawn games');
check(!seen.has('NONDRAW'), 'perfect self-play never produces a decisive result');
click($('#toggleAuto'));
await sleep(600);

console.log('\nfinal state');
const sc = score();
check(sc.every((n) => Number.isInteger(n)), 'scoreboard intact', `(${sc.join('/')})`);
check(!/undefined|NaN/.test(visibleText()), 'still no undefined/NaN in visible text');
check(errors.length === 0, 'no runtime errors during the session', errors.join('; '));

console.log('');
if (fails.length) {
  console.error(`FAILED: ${fails.join('; ')}`);
  process.exit(1);
}
console.log('All UI integration checks passed.');
