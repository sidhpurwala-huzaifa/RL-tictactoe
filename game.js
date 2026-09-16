/* Tic-tac-toe core logic for the browser game.
 *
 * The agent's policy is stored in *canonical* coordinates: the state is encoded
 * from the point of view of the player to move, reduced under the 8 symmetries
 * of the square.  This file rebuilds that symmetry group independently and
 * maps a canonical move back onto the real board, so the browser plays exactly
 * the tabulated policy that train.py verified.
 *
 * Loaded as a classic script (works from file://) and also usable from Node
 * via the verify_policy.mjs script.
 */
(function (global) {
  'use strict';

  var LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
  ];

  var YOU = -1;                        // pseudo player id used only by the UI

  /* ---------------------------------------------------------------- rules */

  function winner(board) {
    for (var i = 0; i < LINES.length; i++) {
      var a = LINES[i][0], b = LINES[i][1], c = LINES[i][2];
      var v = board[a];
      if (v && v === board[b] && v === board[c]) return v;
    }
    for (var j = 0; j < 9; j++) if (!board[j]) return null;
    return 0;                          // draw
  }

  function winningLine(board) {
    for (var i = 0; i < LINES.length; i++) {
      var a = LINES[i][0], b = LINES[i][1], c = LINES[i][2];
      if (board[a] && board[a] === board[b] && board[a] === board[c]) return LINES[i];
    }
    return null;
  }

  function legalMoves(board) {
    var out = [];
    for (var i = 0; i < 9; i++) if (!board[i]) out.push(i);
    return out;
  }

  function other(player) { return player === 1 ? 2 : 1; }

  /* --------------------------------------------------------- symmetries */

  function buildTransforms() {
    function rot(j) { return (2 - j % 3) * 3 + Math.floor(j / 3); }
    function flip(j) { return Math.floor(j / 3) * 3 + (2 - j % 3); }
    var r = [], f = [], identity = [], i;
    for (i = 0; i < 9; i++) { r.push(rot(i)); f.push(flip(i)); identity.push(i); }
    var seen = {}, order = [identity];
    seen[identity.join('')] = true;
    while (order.length) {
      var p = order.pop();
      [r, f].forEach(function (g) {
        var next = [];
        for (var k = 0; k < 9; k++) next.push(g[p[k]]);
        var k2 = next.join('');
        if (!seen[k2]) { seen[k2] = true; order.push(next); }
      });
    }
    var out = Object.keys(seen).map(function (s) {
      return s.split('').map(Number);
    });
    out.sort(function (a, b) { return a.join('') < b.join('') ? -1 : 1; });
    return out;
  }

  var TRANSFORMS = buildTransforms();

  function applyTransform(board, perm) {
    var out = new Array(9);
    for (var i = 0; i < 9; i++) out[i] = board[perm[i]];
    return out;
  }

  function keyOf(board) { return board.join(''); }

  function canonical(board) {
    var best = null;
    for (var i = 0; i < TRANSFORMS.length; i++) {
      var s = keyOf(applyTransform(board, TRANSFORMS[i]));
      if (best === null || s < best) best = s;
    }
    return best;
  }

  function transformToCanonical(board) {
    var target = canonical(board);
    for (var i = 0; i < TRANSFORMS.length; i++) {
      if (keyOf(applyTransform(board, TRANSFORMS[i])) === target) return TRANSFORMS[i];
    }
    return null;
  }

  /* Encode an absolute board from `agentPlayer`'s point of view. */
  function relative(board, agentPlayer) {
    var out = new Array(9);
    for (var i = 0; i < 9; i++) {
      out[i] = board[i] === 0 ? 0 : (board[i] === agentPlayer ? 1 : 2);
    }
    return out;
  }

  /* ------------------------------------------------- perfect play (fallback) */

  /* Exact minimax with memoisation.  Alpha-beta pruning is deliberately NOT
   * used here: a value obtained under a narrow window is only a bound, so
   * caching it (as the memo table does) would let the search pick a losing
   * move.  Plain negamax over <=5478 states is fast enough. */
  var negamaxCache = {};

  function negamax(board, player) {
    var w = winner(board);
    if (w !== null) {
      if (w === 0) return 0;
      return w === player ? 1 : -1;
    }
    var key = board.join('') + '|' + player;
    var cached = negamaxCache[key];
    if (cached !== undefined) return cached;
    var best = -2;
    for (var i = 0; i < 9; i++) {
      if (board[i]) continue;
      var next = board.slice();
      next[i] = player;
      var score = -negamax(next, other(player));
      if (score > best) best = score;
    }
    negamaxCache[key] = best;
    return best;
  }

  /* Best move by exhaustive search; ties broken at random. */
  function minimaxMove(board, player) {
    var bestScore = -2, candidates = [];
    for (var i = 0; i < 9; i++) {
      if (board[i]) continue;
      var next = board.slice();
      next[i] = player;
      var score = -negamax(next, other(player));
      if (score > bestScore) { bestScore = score; candidates = [i]; }
      else if (score === bestScore) candidates.push(i);
    }
    if (!candidates.length) return -1;
    return candidates[Math.floor(Math.random() * candidates.length)];
  }

  /* ------------------------------------------------------------ agent */

  function levelById(model, id) {
    for (var i = 0; i < model.levels.length; i++) {
      if (model.levels[i].id === id) return model.levels[i];
    }
    return model.levels[model.levels.length - 1];
  }

  /* The move the agent plays on `board` (absolute coordinates). */
  function agentMove(board, agentPlayer, model, levelId) {
    var level = levelById(model, levelId);
    var rel = relative(board, agentPlayer);
    var key = canonical(rel);
    var perm = transformToCanonical(rel);
    var action = level.policy[key];
    if (action !== undefined && perm && board[perm[action]] === 0) {
      return perm[action];
    }
    // Not tabulated: the shielded level falls back to perfect play, the
    // learning levels fall back to a random legal move (as they were trained).
    if (level.shielded) return minimaxMove(board, agentPlayer);
    var legal = legalMoves(board);
    return legal[Math.floor(Math.random() * legal.length)];
  }

  /* Q-values for the agent's moves on this board, in board coordinates.
   * Positive is good for the agent.  Returns null when unavailable. */
  function agentValues(board, agentPlayer, model, levelId) {
    var level = levelById(model, levelId);
    if (!level.values) return null;
    var rel = relative(board, agentPlayer);
    var vals = level.values[canonical(rel)];
    if (!vals) return null;
    var perm = transformToCanonical(rel);
    var out = new Array(9);
    for (var j = 0; j < 9; j++) out[perm[j]] = vals[j];
    return out;
  }

  var api = {
    YOU: YOU,
    LINES: LINES,
    TRANSFORMS: TRANSFORMS,
    winner: winner,
    winningLine: winningLine,
    legalMoves: legalMoves,
    other: other,
    keyOf: keyOf,
    canonical: canonical,
    applyTransform: applyTransform,
    transformToCanonical: transformToCanonical,
    relative: relative,
    minimaxMove: minimaxMove,
    levelById: levelById,
    agentMove: agentMove,
    agentValues: agentValues
  };

  global.TTT = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
