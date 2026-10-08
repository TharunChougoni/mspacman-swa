"""Pixel-grid BFS sweep for maze 1. No RAM bit→cell map.

Walk every walkable 4px cell; RAM pellet count is the done-signal.
Ghosts: flee the nearest blob. Do not call skip-4 C3 on skip-1 frames.
"""
from __future__ import annotations

from collections import deque

import numpy as np

CELL = 4
# ALE MsPacman meanings
A_NOOP, A_UP, A_RIGHT, A_LEFT, A_DOWN = 0, 1, 2, 3, 4


def grid_shape(h: int, w: int) -> tuple[int, int]:
    return h // CELL, w // CELL


def extract_full(rgb: np.ndarray):
    """wall, dot, player, ghost masks. rgb is (H, W, 3) uint8."""
    R = rgb[..., 0].astype(np.int16)
    G = rgb[..., 1].astype(np.int16)
    B = rgb[..., 2].astype(np.int16)
    rows = np.arange(rgb.shape[0])[:, None]
    wall = (R > 210) & (G > 95) & (G < 135) & (B > 95) & (B < 135)
    dot = (R > 180) & (G > 60) & (G < 95) & (B > 60) & (B < 95)
    player = (R > 150) & (G > 95) & (B < 120) & (R - G < 120)
    ghost = (
        (np.maximum.reduce([R, G, B]) > 150)
        & (~wall)
        & (~dot)
        & (~player)
        & (rows > 30)
        & (rows < rgb.shape[0] - 14)
        & ~((R > 200) & (G > 200) & (B > 200))
    )
    return wall, dot, player, ghost


def extract(rgb: np.ndarray):
    """wall_or_ghost, player, ghost masks. rgb is (H, W, 3) uint8."""
    wall, _dot, player, ghost = extract_full(rgb)
    return wall | ghost, player, ghost


def blobs(mask: np.ndarray, topk: int = 4, min_px: int = 12):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    cells: dict[tuple[int, int], list] = {}
    for x, y in zip(xs, ys):
        cells.setdefault((int(x) // 4, int(y) // 4), []).append((int(x), int(y)))
    comps = []
    seen: set[tuple[int, int]] = set()
    for key in cells:
        if key in seen:
            continue
        q = [key]
        comp = []
        while q:
            k = q.pop()
            if k in seen:
                continue
            seen.add(k)
            comp.extend(cells.get(k, []))
            cx, cy = k
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nk = (cx + dx, cy + dy)
                if nk in cells and nk not in seen:
                    q.append(nk)
        arr = np.asarray(comp)
        comps.append((float(arr[:, 0].mean()), float(arr[:, 1].mean()), int(len(comp))))
    comps.sort(key=lambda c: -c[2])
    return [(x, y, n) for x, y, n in comps if n >= min_px][:topk]


def player_pos(player_mask: np.ndarray, prev):
    comps = blobs(player_mask)
    if not comps:
        return None, comps
    if prev is None:
        return (comps[0][0], comps[0][1]), comps
    best = min(comps, key=lambda c: (c[0] - prev[0]) ** 2 + (c[1] - prev[1]) ** 2)
    return (best[0], best[1]), comps


def build_grid(wall: np.ndarray, gh: int, gw: int):
    hh, ww = gh * CELL, gw * CELL
    h, w = wall.shape
    block = np.zeros((hh, ww), dtype=np.float32)
    block[: min(h, hh), : min(w, ww)] = wall[: min(h, hh), : min(w, ww)]
    return block.reshape(gh, CELL, gw, CELL).mean(axis=(1, 3)) > 0.45


def in_grid(x: int, y: int, gh: int, gw: int) -> bool:
    return 0 <= x < gw and 0 <= y < gh


def bfs_next(wall_c: np.ndarray, start: tuple[int, int], visited: np.ndarray):
    """Nearest unvisited walkable cell. Marks the path visited. Returns (next_cell, dist) or None."""
    if start is None:
        return None
    gx, gy = start
    gh, gw = wall_c.shape
    if not in_grid(gx, gy, gh, gw) or wall_c[gy, gx]:
        return None
    dist = {start: 0}
    prev = {start: None}
    q = deque([start])
    best = None
    while q:
        cur = q.popleft()
        cx, cy = cur
        if not visited[cy, cx] and (best is None or dist[cur] < dist[best]):
            best = cur
            if dist[cur] > 8:
                break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if in_grid(nx, ny, gh, gw) and not wall_c[ny, nx] and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[cur] + 1
                prev[(nx, ny)] = cur
                q.append((nx, ny))
    if best is None:
        return None
    cur = best
    while cur is not None:
        visited[cur[1], cur[0]] = True
        cur = prev[cur]
    cur = best
    while prev.get(cur) != start and prev.get(cur) is not None:
        cur = prev[cur]
    return cur, dist[best]


def bfs_toward(wall_c: np.ndarray, start: tuple[int, int], goals: set[tuple[int, int]]):
    """First step cell from start toward nearest goal. None if no path."""
    if start is None or not goals:
        return None
    gh, gw = wall_c.shape
    gx, gy = start
    if not in_grid(gx, gy, gh, gw) or wall_c[gy, gx]:
        return None
    if start in goals:
        return start
    dist = {start: 0}
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q = deque([start])
    hit = None
    while q:
        cur = q.popleft()
        if cur in goals and cur != start:
            hit = cur
            break
        cx, cy = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if in_grid(nx, ny, gh, gw) and not wall_c[ny, nx] and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[cur] + 1
                prev[(nx, ny)] = cur
                q.append((nx, ny))
    if hit is None:
        return None
    cur = hit
    while prev[cur] is not None and prev[cur] != start:
        cur = prev[cur]
    return cur


def pick_goal(pc: tuple[int, int], goals: set[tuple[int, int]], rng: np.random.Generator, mode: str):
    """Choose a pellet cell. nearest | farthest | k3 (random of 3 nearest)."""
    if not goals:
        return None
    scored = sorted(goals, key=lambda g: (g[0] - pc[0]) ** 2 + (g[1] - pc[1]) ** 2)
    if mode == "farthest":
        return scored[-1]
    if mode == "k3":
        k = scored[: min(3, len(scored))]
        return k[int(rng.integers(0, len(k)))]
    return scored[0]


_WALL_C = None


def seek_action(
    rgb: np.ndarray,
    prev_player,
    rng: np.random.Generator,
    flee_px: float = 36.0,
    epsilon: float = 0.12,
    mode: str = "nearest",
    goals: set[tuple[int, int]] | None = None,
):
    """Flee close ghosts, else BFS to a remaining pellet (mode=nearest|farthest|k3).

    `goals` if given are (gx, gy) cells; pixel dots are not re-extracted.
    """
    global _WALL_C
    wall, dot, player, ghost = extract_full(rgb)
    h, w = rgb.shape[:2]
    gh, gw = grid_shape(h, w)
    if _WALL_C is None or _WALL_C.shape != (gh, gw):
        _WALL_C = build_grid(wall, gh, gw)
    wall_c = _WALL_C
    pp, _ = player_pos(player, prev_player)
    pc = None
    if pp is not None:
        pc = (int(pp[0] // CELL), int(pp[1] // CELL))
        if not in_grid(pc[0], pc[1], gh, gw):
            pc = None
    gb = blobs(ghost)
    gcells: set[tuple[int, int]] = set()
    dmin = 1e9
    if gb and pc:
        for x, y, _n in gb:
            gx, gy = int(x // CELL), int(y // CELL)
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    if in_grid(gx + ddx, gy + ddy, gh, gw):
                        gcells.add((gx + ddx, gy + ddy))
        dmin = min(
            ((x - (pc[0] * CELL + 2)) ** 2 + (y - (pc[1] * CELL + 2)) ** 2) ** 0.5
            for x, y, _n in gb
        )
    # Treat current-frame ghost cells as walls so the seek path routes AROUND
    # ghosts instead of walking through their corridor and dying before it can
    # reach/eat a pellet. Does NOT mutate the static _WALL_C cache.
    if gcells:
        wall_c = _WALL_C.copy()
        for _gx, _gy in gcells:
            wall_c[_gy, _gx] = True
        if pc is not None:
            wall_c[pc[1], pc[0]] = False
    if pc is not None and dmin < flee_px:
        return flee(wall_c, pc, gcells), pp
    if pc is None:
        return int(rng.integers(1, 5)), pp
    if rng.random() < epsilon:
        opts = []
        for dx, dy, aa in ((0, -1, A_UP), (1, 0, A_RIGHT), (-1, 0, A_LEFT), (0, 1, A_DOWN)):
            nx, ny = pc[0] + dx, pc[1] + dy
            if in_grid(nx, ny, gh, gw) and not wall_c[ny, nx]:
                opts.append(aa)
        if opts:
            return int(opts[int(rng.integers(0, len(opts)))]), pp
    if goals is None:
        goals = set()
        ys, xs = np.nonzero(dot)
        for x, y in zip(xs.tolist(), ys.tolist()):
            gx, gy = int(x) // CELL, int(y) // CELL
            if in_grid(gx, gy, gh, gw) and not wall_c[gy, gx]:
                goals.add((gx, gy))
    else:
        goals = {g for g in goals if in_grid(g[0], g[1], gh, gw) and not wall_c[g[1], g[0]]}
    goal = pick_goal(pc, goals, rng, mode)
    nxt = bfs_toward(wall_c, pc, {goal} if goal is not None else set())
    if nxt is None or nxt == pc:
        return int(rng.integers(1, 5)), pp
    return to_action(pc, nxt), pp


def flee(wall_c: np.ndarray, pc: tuple[int, int], ghost_cells: set[tuple[int, int]]) -> int:
    gx, gy = pc
    if not ghost_cells:
        return A_NOOP
    gh, gw = wall_c.shape
    ggx, ggy = min(ghost_cells, key=lambda c: (c[0] - gx) ** 2 + (c[1] - gy) ** 2)
    best_a, best_d = A_NOOP, -1
    for dx, dy, a in ((0, -1, A_UP), (1, 0, A_RIGHT), (-1, 0, A_LEFT), (0, 1, A_DOWN)):
        nx, ny = gx + dx, gy + dy
        if in_grid(nx, ny, gh, gw) and not wall_c[ny, nx]:
            d = (nx - ggx) ** 2 + (ny - ggy) ** 2
            if d > best_d:
                best_d, best_a = d, a
    return best_a


def to_action(start: tuple[int, int], nxt: tuple[int, int]) -> int:
    dx, dy = nxt[0] - start[0], nxt[1] - start[1]
    if dx == 0 and dy == 0:
        return A_NOOP
    if dx > 0 and dy == 0:
        return A_RIGHT
    if dx < 0 and dy == 0:
        return A_LEFT
    if dx == 0 and dy > 0:
        return A_DOWN
    if dx == 0 and dy < 0:
        return A_UP
    if dx > 0 and dy < 0:
        return 5
    if dx < 0 and dy < 0:
        return 6
    if dx > 0 and dy > 0:
        return 7
    return 8


def choose_action(
    rgb: np.ndarray,
    visited: np.ndarray,
    prev_player,
    hold: int,
    hold_a: int,
    rng: np.random.Generator,
    flee_px: float = 40.0,
    hold_n: int = 26,
    epsilon: float = 0.05,
):
    """One skip-1 action. Returns (action, hold, hold_a, prev_player, dmin)."""
    h, w = rgb.shape[:2]
    gh, gw = visited.shape
    wall, player, ghost = extract(rgb)
    wall_c = build_grid(wall, gh, gw)
    pp, _ = player_pos(player, prev_player)
    pc = None
    if pp is not None:
        pc = (int(pp[0] // CELL), int(pp[1] // CELL))
        if in_grid(pc[0], pc[1], gh, gw):
            visited[pc[1], pc[0]] = True
        else:
            pc = None
    gb = blobs(ghost)
    gcells: set[tuple[int, int]] = set()
    dmin = 1e9
    if gb and pc:
        for x, y, _n in gb:
            gx, gy = int(x // CELL), int(y // CELL)
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    if in_grid(gx + ddx, gy + ddy, gh, gw):
                        gcells.add((gx + ddx, gy + ddy))
        dmin = min(
            ((x - (pc[0] * CELL + 2)) ** 2 + (y - (pc[1] * CELL + 2)) ** 2) ** 0.5
            for x, y, _n in gb
        )
    if pc is not None and dmin < flee_px:
        return flee(wall_c, pc, gcells), 0, hold_a, pp, dmin
    if hold > 0 and dmin > flee_px * 0.6:
        return hold_a, hold - 1, hold_a, pp, dmin
    if pc is None:
        return A_NOOP, 0, hold_a, pp, dmin
    nxt = bfs_next(wall_c, pc, visited)
    if nxt is None:
        return A_NOOP, 0, hold_a, pp, dmin
    nextc, _dist = nxt
    if rng.random() < epsilon:
        opts = []
        for dx, dy, aa in ((0, -1, A_UP), (1, 0, A_RIGHT), (-1, 0, A_LEFT), (0, 1, A_DOWN)):
            nx, ny = pc[0] + dx, pc[1] + dy
            if in_grid(nx, ny, gh, gw) and not wall_c[ny, nx]:
                opts.append(aa)
        a = int(opts[int(rng.integers(0, len(opts)))]) if opts else A_NOOP
        return a, 8, a, pp, dmin
    a = to_action(pc, nextc)
    return a, hold_n, a, pp, dmin
