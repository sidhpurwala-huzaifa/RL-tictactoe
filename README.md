# Tic-Tac-Toe vs. a self-play RL agent

A graphical tic-tac-toe game where you play against an agent that learned the game
purely by playing itself — and that is **provably unable to lose**.

Open `index.html` in a browser. That's it: no server, no build step, no dependencies.

```
open index.html          # macOS
```

---

## The game

* Click a square (or press <kbd>1</kbd>–<kbd>9</kbd>, or <kbd>N</kbd> for a new game).
* Choose whether you play **X (first)** or **O (second)**.
* Three opponents, taken from different points in training:

  | Level | Games trained | Behaviour |
  |---|---|---|
  | **Novice** | 1,000 | genuinely weak — it loses games, including to random play |
  | **Intermediate** | 20,000 | strong, but a sharp player can still punish it |
  | **Perfect** | 400,000 | the finished agent: verified to never lose |

* **Show move values** displays the value the agent has learned for each of its
  possible moves, live, while it is choosing (Perfect level only).
* **Agent vs itself** plays the perfect agent against itself — a game that always
  ends in a draw.
* The scoreboard, winning-line highlight, and turn indicator are all wired to the
  real game state.

## How it learned

Nothing about strategy is hard-coded. `train.py` runs self-play value learning:

1. **State encoding.** A position is stored from the point of view of the player to
   move (own stones `1`, opponent `2`), so a single table serves both X and O.
2. **Symmetry reduction.** States are canonicalised under the 8 symmetries of the
   square (4 rotations × mirror). The 5,478 reachable positions collapse to **627
   canonical states**, so the whole game fits in a table that can be solved exactly.
3. **Self-play.** Both seats share the table and play hundreds of thousands of games,
   ε-greedy, starting from random play.
4. **Update rule.** Each move's value is moved toward a target that mixes the
   bootstrapped value of the resulting position with the **actual result of the
   game** (+1 win / 0 draw / −1 loss):

   ```
   target = (1 − λ) · γ · ( − max_a' Q[next][a'] )  +  λ · outcome        λ = 0.5
   ```

   The `outcome` term matters. With a purely bootstrapped target, a self-play agent
   that cannot yet punish a bad move sees that move as a draw and locks into an
   "everything is a draw" fixed point — that failure mode is reproducible in this
   repo (it plateaus at 19–102 wrong states). Mixing in the game result lets a rare
   losing line correct the move that caused it immediately.

### Training result

400,000 self-play games, 8.3 seconds, single CPU core, no numerical libraries.

| Self-play games | Losing lines as X | Losing lines as O | Record vs. random (as X) |
|---|---|---|---|
| 1,000 (Novice) | 10 | 166 | 1905 W / 0 D / 95 L |
| 20,000 (Intermediate) | 1 | 81 | 1915 W / 75 D / 10 L |
| **400,000 (Perfect)** | **0** | **0** | 1980 W / 20 D / **0 L** |

"Losing lines" = number of distinct opponent play sequences that beat the agent,
found by exhaustive search of the entire game tree. The finished agent: 0 losing
lines as X, 0 as O, and 400/400 draws against perfect minimax play from both seats.

## Why it cannot lose

Tic-tac-toe is a solved game: with perfect play it is a draw, so "never loses" is the
strongest guarantee that exists — an opponent playing perfectly should draw, and
anything less should lose. Two independent mechanisms enforce that:

1. **The learned policy is already safe.** At 400,000 games the learned policy has
   zero losing lines on its own — no patches, no special cases.
2. **A verified safety shield, just in case.** `train.py` also computes, by retrograde
   analysis, the set of moves from which the agent can still guarantee not losing, and
   overrides the learned move anywhere it was unsafe. At 400,000 games it overrode
   **0 of 627** states, so it is a proof-carrying safety net rather than a crutch.

   The safety net is not decoration. Training with a purely bootstrapped target
   (`--lam 0.0`) produces an agent that loses 24 opponent lines as O, while the shield
   overrides 19 of the 627 states and the agent that actually plays still never loses:

   ```bash
   python3 train.py --episodes 40000 --lam 0.0
   #   as O: losses=24            -> raw policy is beatable
   #   shield overrode 19/627     -> repaired
   #   never loses: True          -> the exported agent is still safe
   ```

### Verify it yourself

Three checks, in increasing order of end-to-end-ness:

```bash
python3 train.py --episodes 400000     # retrain (~8s), re-verify, rewrite policy.js
node verify_policy.mjs                 # exhaustive game-tree search of the exported agent
npm install jsdom && node ui_test.mjs  # drives the actual page in a headless browser
```

* `verify_policy.mjs` loads `policy.js` and `game.js` in a VM and searches the complete
  game tree against every legal opponent reply, using the exact canonicalisation and
  lookup the browser uses. It checks parity between the Python and JavaScript symmetry
  groups, proves every reachable state is tabulated (so the built-in perfect-play
  fallback is never needed), and asserts 0 losses as X and as O. Exit code 0 on success.
* `ui_test.mjs` loads `index.html` in jsdom, clicks real squares, reads the real
  scoreboard, and asserts the human never beats the Perfect agent over many games
  (`GAMES=4 node ui_test.mjs` for a quick run).
* `training_report.json` records the numbers above, including both verification
  results, the learning curve, and the shield override count.
* Training is deterministic for a fixed seed: re-running `train.py` with the default
  arguments reproduces the shipped `policy.js` tables exactly.

## Files

| File | Role |
|---|---|
| `index.html` | The game: board, controls, scoreboard, move-value overlay, self-play demo |
| `game.js` | Game rules, symmetry canonicalisation, policy lookup, minimax fallback |
| `policy.js` | **Generated** by `train.py`: the learned tables for all three levels |
| `train.py` | Self-play training, safety shield, exhaustive verification, export |
| `verify_policy.mjs` | Exhaustive end-to-end verification of the exported agent (Node) |
| `ui_test.mjs` | Headless browser integration test (needs `jsdom`) |
| `training_report.json` | Machine-readable training and verification report |

## Retraining

```bash
python3 train.py                              # defaults: 400k games, seed 7, λ 0.5
python3 train.py --episodes 100000 --seed 3   # a different run
python3 train.py --lam 0.0                    # pure bootstrapped target (watch it stall)
python3 train.py --help
```

Only the Python standard library is used, so it runs anywhere. `train.py` refuses to
export if the shielded policy can lose, and prints the learning curve, the shield
override count, and the exhaustive verification results for both seats.

## Notes

* The agent plays as both X and O from one table, because states are encoded relative
  to the player to move.
* No neural network is used: the state space is small enough that a table is exact, so
  there is no approximation error to hide behind.
* `policy.js` is the only file the browser needs from training; deleting it breaks the
  page (there is no fallback model), so re-run `train.py` if it goes missing.
* Learned values for a drawn game hover near zero (e.g. `+0.03` for the centre on an
  empty board versus `−0.01` for a corner). That is correct rather than broken: with
  perfect play every move leads to a draw. The ordering is what the agent plays by.
