#!/usr/bin/env python3
"""
Self-play reinforcement learning for tic-tac-toe, with an exhaustively
verified "never loses" safety shield.

Pipeline
--------
1.  SELF-PLAY VALUE LEARNING.  Both seats share one Q-table.  A board is
    encoded from the perspective of the player to move (own stones = 1,
    opponent = 2) and canonicalised under the 8 symmetries of the square,
    which shrinks the state space from 5478 reachable positions to 627
    canonical states and makes learning very fast.  Rewards: +1 win, -1 loss,
    0 draw.  Because the encoding is relative, one table learns to play both
    X and O.  Each update mixes the bootstrapped negamax target with the
    actual game outcome (lambda = 0.5); see QLearning for why that matters.

2.  CHECKPOINTS.  Raw policies are snapshotted at several episode counts so the
    learning progress (and its remaining mistakes) is visible.

3.  SAFETY SHIELD.  Retrograde analysis computes, for every reachable state,
    the set of moves from which the agent can still guarantee at least a draw.
    The final policy plays the highest-Q *safe* move.  By induction over the
    game it therefore never loses, while staying as close to the RL policy as
    possible.

4.  EXHAUSTIVE VERIFICATION.  Each policy is checked by searching the *entire*
    game tree against *every* legal opponent reply.  A policy passes only if no
    opponent strategy - however weird - can ever win.

5.  EXPORT.  Everything is written to js/policy.js for the browser game, and a
    summary is written to training_report.json.

Usage
-----
    python3 train.py                    # full run (default 400k self-play games)
    python3 train.py --episodes 50000   # quicker run
"""

from __future__ import annotations

import argparse
import json
import random
import time
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------
# Game rules
# --------------------------------------------------------------------------

LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
)

EMPTY = (0,) * 9


def winner(board):
    """Return 1/2 if that player has three in a row, 0 for a draw, None if the
    game is still running.  Works in both absolute and relative coordinates."""
    for a, b, c in LINES:
        v = board[a]
        if v and v == board[b] == board[c]:
            return v
    return 0 if all(board) else None


def with_move(board, index, player):
    nxt = list(board)
    nxt[index] = player
    return tuple(nxt)


# --------------------------------------------------------------------------
# Symmetry canonicalisation (the 8 symmetries of the square)
# --------------------------------------------------------------------------

def _build_transforms():
    """All 8 permutations `p` such that apply(board, p)[i] == board[p[i]]."""
    def rot(j):        # rotate 90 degrees
        return (2 - j % 3) * 3 + j // 3

    def flip(j):       # mirror left/right
        return (j // 3) * 3 + (2 - j % 3)

    r = tuple(rot(j) for j in range(9))
    f = tuple(flip(j) for j in range(9))
    identity = tuple(range(9))
    seen = {identity}
    stack = [identity]
    while stack:
        p = stack.pop()
        for g in (r, f):
            nxt = tuple(g[p[i]] for i in range(9))   # g applied after p
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return tuple(sorted(seen))


TRANSFORMS = _build_transforms()
assert len(TRANSFORMS) == 8, len(TRANSFORMS)


def inverse_perm(p):
    inv = [0] * 9
    for i, j in enumerate(p):
        inv[j] = i
    return tuple(inv)


INVERSES = {p: inverse_perm(p) for p in TRANSFORMS}


def apply_transform(board, perm):
    return tuple(board[perm[i]] for i in range(9))


@lru_cache(maxsize=None)
def canonical(board):
    """Smallest equivalent board under the 8 symmetries (a canonical key)."""
    return min(apply_transform(board, p) for p in TRANSFORMS)


@lru_cache(maxsize=None)
def transform_to_canonical(board):
    """The permutation p with apply_transform(board, p) == canonical(board).

    Since apply_transform(board, p)[i] == board[p[i]], the canonical cell `a`
    corresponds to the original cell `p[a]` - this is how a move chosen in
    canonical coordinates is translated back to the real board.
    """
    target = canonical(board)
    for p in TRANSFORMS:
        if apply_transform(board, p) == target:
            return p
    raise AssertionError("no symmetry maps board to its canonical form")


def policy_move(policy, rel_board):
    """Look up the policy in canonical space, return the move in `rel_board`
    coordinates (or None if the state is unknown)."""
    key = key_of(canonical(rel_board))
    action = policy.get(key)
    if action is None:
        return None
    return transform_to_canonical(rel_board)[action]


def key_of(board) -> str:
    return "".join(map(str, board))


def board_of(key: str):
    return tuple(int(c) for c in key)


def relative(board, agent):
    """Encode an absolute board from `agent`'s point of view."""
    return tuple(0 if v == 0 else (1 if v == agent else 2) for v in board)


# --------------------------------------------------------------------------
# Reachable state enumeration
# --------------------------------------------------------------------------

def reachable_positions():
    """Every position reachable from the empty board with X moving first."""
    seen = {EMPTY}
    stack = [EMPTY]
    while stack:
        board = stack.pop()
        if winner(board) is not None:
            continue
        player = 1 if board.count(1) == board.count(2) else 2
        for i in range(9):
            if board[i] == 0:
                nxt = with_move(board, i, player)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return seen


REACHABLE = reachable_positions()


def agent_to_move_states():
    """Map canonical key -> canonical board, for every reachable state in which
    the agent is to move (agent may be either X or O)."""
    states: dict[str, tuple] = {}
    for board in REACHABLE:
        if winner(board) is not None:
            continue
        player = 1 if board.count(1) == board.count(2) else 2
        key = key_of(canonical(relative(board, player)))
        states[key] = board_of(key)          # canonical representative
    return states


STATES = agent_to_move_states()              # all keys are "agent to move"


# --------------------------------------------------------------------------
# Retrograde analysis: can the agent still avoid losing?
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def best_outcome(board, agent_moves: bool) -> int:
    """Optimal-play outcome for the agent (player 1) from `board`.

    +1 agent wins, 0 draw, -1 agent loses.  `agent_moves` says who moves next;
    the board is always in relative coordinates (agent = 1, opponent = 2).
    Tracking the mover explicitly matters because the two families of relative
    states (agent is X with equal stone counts, agent is O with one stone
    fewer) cannot be told apart by the stone counts alone.
    """
    result = winner(board)
    if result is not None:
        return 1 if result == 1 else (0 if result == 0 else -1)
    children = [best_outcome(with_move(board, i, 1 if agent_moves else 2),
                             not agent_moves)
                for i in range(9) if board[i] == 0]
    return max(children) if agent_moves else min(children)


def forced_win(board) -> bool:
    """Agent to move: can he force a win?"""
    return best_outcome(board, True) == 1


def safe_moves(board):
    """Agent-to-move moves that keep a guarantee of never losing."""
    return [i for i in range(9)
            if board[i] == 0 and best_outcome(with_move(board, i, 1), False) >= 0]


# --------------------------------------------------------------------------
# Q-learning by self-play
# --------------------------------------------------------------------------

class QLearning:
    """Self-play value learning with a mixed TD / Monte-Carlo target.

    target = (1 - lam) * gamma * (-max_a' Q[next_state][a'])  +  lam * outcome

    The first term is the usual bootstrapped negamax target; the second is the
    actual result of the game.  Mixing them matters: with a purely bootstrapped
    target, a self-play agent that has not yet learned to punish a bad move
    sees that move as a draw, and the zero-initialised table locks into an
    "everything is a draw" fixed point.  The outcome term carries the terminal
    reward to every state of the episode immediately, which is what lets the
    rare punished lines correct the policy.
    """

    def __init__(self, lr=0.5, gamma=1.0, lam=0.5, seed=7):
        self.Q: dict[str, list[float]] = {}
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.rng = random.Random(seed)

    def values(self, key: str) -> list[float]:
        vals = self.Q.get(key)
        if vals is None:
            vals = [0.0] * 9
            self.Q[key] = vals
        return vals

    def play_episode(self, epsilon: float):
        """One self-play game; both seats use this same table.

        Everything the table sees is in *canonical* coordinates: the state key
        and the action index.  A move made on the real board is translated into
        canonical coordinates before it is written to Q, otherwise the 8
        symmetric copies of a position would all write into different slots.
        """
        board = list(EMPTY)
        player = 1
        trajectory = []                        # (key, canonical action) per ply

        while True:
            rel = tuple(1 if v == player else (2 if v else 0) for v in board)
            key = key_of(canonical(rel))
            perm = transform_to_canonical(rel)
            inv = INVERSES[perm]
            legal = [i for i in range(9) if board[i] == 0]
            canon_legal = [inv[i] for i in legal]
            vals = self.values(key)

            if self.rng.random() < epsilon:
                canon_action = self.rng.choice(canon_legal)
            else:
                best = max(vals[c] for c in canon_legal)
                canon_action = self.rng.choice(
                    [c for c in canon_legal if vals[c] == best])

            board[perm[canon_action]] = player
            trajectory.append((key, canon_action))

            result = winner(tuple(board))
            if result is not None:
                break
            player = 3 - player

        # Backward update with the zero-sum sign flip between the seats.
        n = len(trajectory)
        for idx in range(n - 1, -1, -1):
            key, action = trajectory[idx]
            mover = 1 if idx % 2 == 0 else 2
            outcome = 1.0 if result == mover else (0.0 if result == 0 else -1.0)
            if idx == n - 1:
                target = outcome
            else:
                next_key = trajectory[idx + 1][0]
                td_target = self.gamma * (-max(self.values(next_key)))
                target = (1.0 - self.lam) * td_target + self.lam * outcome
            vals = self.values(key)
            vals[action] += self.lr * (target - vals[action])

    def train(self, episodes: int, eps_start=1.0, eps_end=0.05,
              lr_start=0.6, lr_end=0.10, checkpoints=(), progress=50_000):
        """Train, returning raw policy snapshots at the requested checkpoints."""
        snapshots: dict[int, dict[str, int]] = {}
        wanted = sorted(checkpoints)
        t0 = time.time()
        for ep in range(1, episodes + 1):
            frac = ep / episodes
            epsilon = eps_start + (eps_end - eps_start) * frac
            self.lr = lr_start + (lr_end - lr_start) * frac
            self.play_episode(epsilon)
            if wanted and ep == wanted[0]:
                snapshots[ep] = raw_policy(self)
                wanted.pop(0)
                print(f"  checkpoint {ep:>7,} games  "
                      f"({time.time() - t0:6.1f}s, {len(self.Q)} states)")
            elif progress and ep % progress == 0:
                print(f"  trained   {ep:>7,}/{episodes:,} games  "
                      f"({time.time() - t0:6.1f}s, {len(self.Q)} states)")
        self.train_seconds = time.time() - t0
        return snapshots


def raw_policy(q: QLearning) -> dict[str, int]:
    """Greedy policy over every reachable agent-to-move state (no shield)."""
    policy = {}
    for key, board in STATES.items():
        vals = q.Q.get(key)
        legal = [i for i in range(9) if board[i] == 0]
        if not vals:
            policy[key] = legal[0]
            continue
        best = max(vals[i] for i in legal)
        policy[key] = min(i for i in legal if vals[i] == best)
    return policy


def shielded_policy(q: QLearning) -> tuple[dict[str, int], int]:
    """Highest-Q move among the provably safe moves, per state.

    The RL policy is kept wherever it is already safe, so the shield only
    overrides genuinely losing choices."""
    policy = {}
    overrides = 0
    for key, board in STATES.items():
        legal = [i for i in range(9) if board[i] == 0]
        safe = safe_moves(board) or legal
        vals = q.Q.get(key)
        rl_choice = raw_policy_for(board, vals, legal)
        if rl_choice in safe:
            choice = rl_choice
        else:
            best = max((vals[i] if vals else 0.0) for i in safe)
            choice = min(i for i in safe
                         if (vals[i] if vals else 0.0) == best)
            overrides += 1
        policy[key] = choice
    return policy, overrides


def raw_policy_for(board, vals, legal):
    if not vals:
        return legal[0]
    best = max(vals[i] for i in legal)
    return min(i for i in legal if vals[i] == best)


# --------------------------------------------------------------------------
# Exhaustive verification against every possible opponent
# --------------------------------------------------------------------------

def verify(policy: dict[str, int], agent: int) -> dict:
    """Search the complete game tree.  The agent follows `policy`; the opponent
    may play anything.  `agent` is the absolute player id the agent plays.

    Returns outcome counts over all possible opponent strategies.
    """
    counts = {"agent_wins": 0, "draws": 0, "agent_losses": 0,
              "games": 0, "illegal_or_missing": 0}
    memo: dict[tuple, dict] = {}

    def rec(board, agent_moves: bool) -> dict:
        """Outcome counts from `board` in relative coordinates (agent = 1)."""
        result = winner(board)
        if result is not None:
            if result == 1:
                return {"agent_wins": 1, "draws": 0, "agent_losses": 0, "games": 1}
            if result == 0:
                return {"agent_wins": 0, "draws": 1, "agent_losses": 0, "games": 1}
            return {"agent_wins": 0, "draws": 0, "agent_losses": 1, "games": 1}

        cache_key = (board, agent_moves)
        if cache_key in memo:
            return memo[cache_key]

        total = {"agent_wins": 0, "draws": 0, "agent_losses": 0, "games": 0}
        if agent_moves:
            action = policy_move(policy, board)
            if action is None or board[action] != 0:
                counts["illegal_or_missing"] += 1
                sub = {"agent_wins": 0, "draws": 0, "agent_losses": 1, "games": 1}
            else:
                sub = rec(with_move(board, action, 1), False)
            for k in total:
                total[k] += sub[k]
        else:
            for i in range(9):
                if board[i] == 0:
                    sub = rec(with_move(board, i, 2), True)
                    for k in total:
                        total[k] += sub[k]
        memo[cache_key] = total
        return total

    # The agent may be X (moves first) or O (moves second).
    rel_start = EMPTY
    agent_moves_first = (agent == 1)
    result = rec(rel_start, agent_moves_first)
    counts["agent_wins"] = result["agent_wins"]
    counts["draws"] = result["draws"]
    counts["agent_losses"] = result["agent_losses"]
    counts["games"] = result["games"]
    counts["never_loses"] = counts["agent_losses"] == 0
    counts["agent"] = "X (first)" if agent == 1 else "O (second)"
    return counts


def verify_both_seats(policy) -> dict:
    both = {"as_X": verify(policy, 1), "as_O": verify(policy, 2)}
    both["never_loses"] = all(v["never_loses"] for v in both.values())
    return both


def policy_as_seen(policy) -> dict:
    """How many reachable agent-to-move states are outright won, and how many
    must be drawn, under optimal play."""
    wins = draws = 0
    for board in STATES.values():
        if forced_win(board):
            wins += 1
        else:
            draws += 1
    return {"winning_states": wins, "drawn_states": draws,
            "total_states": len(STATES)}


def eval_vs_random(policy, games, seed=11, agent=1):
    """Play `games` games against a uniformly random opponent."""
    rng = random.Random(seed)
    stats = {"wins": 0, "draws": 0, "losses": 0}
    for _ in range(games):
        board = EMPTY
        turn = 1
        while True:
            if turn == agent:
                action = policy_move(policy, relative(board, agent))
                if action is None:
                    action = rng.choice([i for i in range(9) if board[i] == 0])
            else:
                legal = [i for i in range(9) if board[i] == 0]
                action = rng.choice(legal)
            board = with_move(board, action, turn)
            result = winner(board)
            if result is not None:
                if result == agent:
                    stats["wins"] += 1
                elif result == 0:
                    stats["draws"] += 1
                else:
                    stats["losses"] += 1
                break
            turn = 3 - turn
    return stats


def minimax_move(board, player):
    """Perfect play, used only for reporting."""
    best_score, best_move = -2, None
    for i in range(9):
        if board[i] == 0:
            score = -_negamax(with_move(board, i, player), 3 - player)
            if score > best_score:
                best_score, best_move = score, i
    return best_move


@lru_cache(maxsize=None)
def _negamax(board, player):
    result = winner(board)
    if result is not None:
        if result == 0:
            return 0
        return 1 if result == player else -1
    return max(-_negamax(with_move(board, i, player), 3 - player)
               for i in range(9) if board[i] == 0)


def eval_vs_perfect(policy, agent, games=400, seed=5):
    """Play against perfect minimax (which breaks ties randomly)."""
    rng = random.Random(seed)
    stats = {"wins": 0, "draws": 0, "losses": 0}
    for _ in range(games):
        board = EMPTY
        turn = 1
        while True:
            if turn == agent:
                action = policy_move(policy, relative(board, agent))
                if action is None:
                    action = minimax_move(board, agent)
            else:
                scored = []
                for i in range(9):
                    if board[i] == 0:
                        scored.append((-_negamax(with_move(board, i, turn), 3 - turn), i))
                top = max(s for s, _ in scored)
                action = rng.choice([i for s, i in scored if s == top])
            board = with_move(board, action, turn)
            result = winner(board)
            if result is not None:
                if result == agent:
                    stats["wins"] += 1
                elif result == 0:
                    stats["draws"] += 1
                else:
                    stats["losses"] += 1
                break
            turn = 3 - turn
    return stats


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def build_export(q, snapshots, perfect, overrides, meta_extra):
    def table(policy, values=None):
        out = {"policy": {k: policy[k] for k in sorted(policy)}}
        if values is not None:
            out["values"] = {
                k: [round(v, 3) for v in values.get(k, [0.0] * 9)]
                for k in sorted(policy)
            }
        return out

    levels = []
    novice = snapshots.get(1000)
    if novice:
        levels.append({
            "id": "novice",
            "label": "Novice",
            "subtitle": "raw RL policy, 1k self-play games",
            "episodes": 1000,
            "shielded": False,
            "blurb": "What the agent plays before it has learned much.",
            **table(novice),
        })
    inter = snapshots.get(20_000)
    if inter:
        levels.append({
            "id": "medium",
            "label": "Intermediate",
            "subtitle": "raw RL policy, 20k self-play games",
            "episodes": 20_000,
            "shielded": False,
            "blurb": "Strong, but a sharp human can still punish it.",
            **table(inter),
        })
    levels.append({
        "id": "perfect",
        "label": "Perfect",
        "subtitle": "RL policy + verified safety shield",
        "episodes": q.episodes if hasattr(q, "episodes") else None,
        "shielded": True,
        "verified_never_loses": True,
        "overrides": overrides,
        "blurb": "Never loses. Exhaustively proven against every opponent line.",
        **table(perfect, q.Q),
    })

    return {
        "meta": {
            "algorithm": "tabular self-play value learning, symmetry-canonical states",
            "encoding": "board from the mover's perspective; 8-fold symmetry reduction",
            "target": "mixed TD negamax target and Monte-Carlo outcome (lambda)",
            "rewards": "+1 win / 0 draw / -1 loss, gamma 1.0, epsilon and lr decayed",
            "transforms": [list(p) for p in TRANSFORMS],
            "stateCount": len(STATES),
            "reachablePositions": len(REACHABLE),
            **meta_extra,
        },
        "levels": levels,
    }


def write_policy_js(export, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(export, separators=(",", ":"))
    path.write_text(
        "/* Generated by train.py - do not edit by hand.\n"
        "   Self-play Q-learning policy for tic-tac-toe, with a safety shield\n"
        "   that is verified to never lose. */\n"
        f"window.TTT_MODEL = {body};\n",
        encoding="utf-8",
    )
    return path.stat().st_size


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lam", type=float, default=0.5,
                    help="weight of the Monte-Carlo outcome term in the target")
    ap.add_argument("--checkpoints", type=int, nargs="*", default=[1_000, 20_000])
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "policy.js")
    ap.add_argument("--report", type=Path,
                    default=Path(__file__).parent / "training_report.json")
    args = ap.parse_args()

    print("Tic-tac-toe self-play Q-learning")
    print(f"  reachable positions      : {len(REACHABLE):,}")
    print(f"  canonical agent states   : {len(STATES):,}")

    q = QLearning(seed=args.seed, lam=args.lam)
    checkpoints = [c for c in args.checkpoints if c < args.episodes]
    print(f"  self-play games          : {args.episodes:,}")
    print("training ...")
    snapshots = q.train(args.episodes, checkpoints=tuple(checkpoints))
    q.episodes = args.episodes
    print(f"  done in {q.train_seconds:.1f}s, {len(q.Q)} states in Q-table")

    curve = []
    for ep in sorted(snapshots):
        pol = snapshots[ep]
        vr = verify_both_seats(pol)
        curve.append({
            "episodes": ep,
            "as_X_never_loses": vr["as_X"]["never_loses"],
            "as_O_never_loses": vr["as_O"]["never_loses"],
            "loss_lines_as_X": vr["as_X"]["agent_losses"],
            "loss_lines_as_O": vr["as_O"]["agent_losses"],
            "vs_random": {str(k): eval_vs_random(pol, 2000, agent=k) for k in (1, 2)},
        })

    print("\nverifying the final RL policy (no shield) ...")
    final_raw = raw_policy(q)
    raw_verdict = verify_both_seats(final_raw)
    print(f"  as X: losses={raw_verdict['as_X']['agent_losses']} "
          f"draws={raw_verdict['as_X']['draws']} wins={raw_verdict['as_X']['agent_wins']}")
    print(f"  as O: losses={raw_verdict['as_O']['agent_losses']} "
          f"draws={raw_verdict['as_O']['draws']} wins={raw_verdict['as_O']['agent_wins']}")

    print("\nbuilding safety shield ...")
    perfect, overrides = shielded_policy(q)
    print(f"  states where the shield overrode the RL choice: {overrides}/{len(STATES)}")

    print("verifying the shielded policy exhaustively ...")
    verdict = verify_both_seats(perfect)
    for seat in ("as_X", "as_O"):
        v = verdict[seat]
        print(f"  {v['agent']:<12} games={v['games']:,}  "
              f"agent wins={v['agent_wins']:,}  draws={v['draws']:,}  "
              f"LOSSES={v['agent_losses']}  never loses={v['never_loses']}")
    if not verdict["never_loses"]:
        raise SystemExit("FATAL: shielded policy can still lose")
    assert raw_verdict is not None

    style = policy_as_seen(perfect)
    random_stats = {str(k): eval_vs_random(perfect, 2000, agent=k) for k in (1, 2)}
    perfect_stats = {str(k): eval_vs_perfect(perfect, k) for k in (1, 2)}

    meta_extra = {
        "episodes": args.episodes,
        "lambda": args.lam,
        "seed": args.seed,
        "trainSeconds": round(q.train_seconds, 1),
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rawPolicyVerifiedNeverLoses": raw_verdict["never_loses"],
        "shieldOverrides": overrides,
        "shieldVerification": verdict,
        "verifiedAsX": verdict["as_X"]["never_loses"],
        "verifiedAsO": verdict["as_O"]["never_loses"],
        "verificationGamesAsX": verdict["as_X"]["games"],
        "verificationGamesAsO": verdict["as_O"]["games"],
        "vsRandom": random_stats,
        "vsPerfect": perfect_stats,
        "learningCurve": curve,
    }
    export = build_export(q, snapshots, perfect, overrides, meta_extra)
    size = write_policy_js(export, args.out)
    print(f"\nwrote {args.out} ({size/1024:.1f} KiB), "
          f"{len(export['levels'])} difficulty levels")

    report = {
        "meta": export["meta"],
        "rawPolicyVerification": raw_verdict,
        "shieldVerification": verdict,
        "shieldOverrides": overrides,
        "stateStyle": style,
        "vsRandom": random_stats,
        "vsPerfect": perfect_stats,
        "levelSizes": {lv["id"]: len(lv["policy"]) for lv in export["levels"]},
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.report}")

    print("\nsummary")
    print(f"  raw RL policy never loses : {raw_verdict['never_loses']}")
    print(f"  shielded policy never loses: {verdict['never_loses']}  (proven, both seats)")
    print(f"  perfect play vs agent as X : {perfect_stats['1']}")
    print(f"  perfect play vs agent as O : {perfect_stats['2']}")


if __name__ == "__main__":
    main()
