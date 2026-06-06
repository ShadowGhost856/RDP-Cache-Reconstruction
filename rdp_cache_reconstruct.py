#!/usr/bin/env python3
"""
rdp_cache_reconstruct.py  –  RDP Bitmap Cache reconstruction tool
=================================================================
Reconstructs screen fragments from an RDP bitmap cache (.bin) file.

Approach
--------
Phase 1 – Edge-matching (v3 base: 3px MAD + mutual-best)
Phase 2 – Correlation validation per junction:
  • Correct join: text continues -> per-row brightness correlates (~0.98)
  • Wrong join (gradient/chrome tile): MAD low but correlation negative
  • Bad junctions are split; a better candidate is searched

Requirements: pip install Pillow numpy
bmc-tools.py next to this script (github.com/ANSSI-FR/bmc-tools)

Usage:
  python rdp_cache_reconstruct.py Cache0000.bin
  python rdp_cache_reconstruct.py Cache0000.bin --out ./output --top 60
  python rdp_cache_reconstruct.py Cache0000.bin --corr-threshold 0.4
"""

import argparse, base64, io, json, os, subprocess, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


# ── Startup banner ────────────────────────────────────────────────
_ART = [
    r" ____  ____  ____     ____           _          ",
    r"|  _ \|  _ \|  _ \   / ___|__ _  ___| |__   ___ ",
    r"| |_) | | | | |_) | | |   / _` |/ __| '_ \ / _ \ ",
    r"|  _ <| |_| |  __/  | |__| (_| | (__| | | |  __/",
    r"|_| \_\____/|_|      \____\__,_|\___|_| |_|\___|",
]
_TITLE = "RDP Bitmap Cache  \u2014  Screen Reconstruction Toolkit"
_SUB   = "edge-matching \u00b7 2D jigsaw \u00b7 scroll-dedup \u00b7 OCR \u00b7 assembler"
_FOOT  = "by S. Bosman  \u00b7  forensic screen recovery from Cache000X.bin"

def print_banner(color=None, stream=None):
    stream = stream or sys.stdout
    if color is None:
        color = bool(getattr(stream, "isatty", lambda: False)()) and not os.environ.get("NO_COLOR")
    if color:
        CB="\033[38;5;240m"; CA="\033[38;5;45m"; CT="\033[1;38;5;221m"
        CS="\033[38;5;109m"; CF="\033[38;5;245m"; R="\033[0m"
    else:
        CB=CA=CT=CS=CF=R=""
    V="\u2551"; H="\u2550"; TL="\u2554"; TR="\u2557"; BL="\u255a"; BR="\u255d"
    inner = max(max(len(a) for a in _ART), len(_TITLE), len(_SUB), len(_FOOT)) + 4
    def row(text, col):
        left = 2; right = inner - len(text) - left
        return CB + V + R + " "*left + col + text + R + " "*right + CB + V + R
    top = CB + TL + H*inner + TR + R
    bot = CB + BL + H*inner + BR + R
    mid = CB + V + " "*inner + V + R
    lines = [top, mid] + [row(a, CA) for a in _ART] + [mid,
             row(_TITLE, CT), row(_SUB, CS), row(_FOOT, CF), mid, bot]
    print("\n".join(lines), file=stream)



# ── Tile extraction ────────────────────────────────────────────────────────────

def extract_tiles_bmc(bin_path, out_dir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "bmc-tools.py"),
        os.path.join(script_dir, "bmc-tools", "bmc-tools.py"),
        os.path.expanduser("~/bmc-tools/bmc-tools.py"),
    ]
    bmc = next((c for c in candidates if os.path.exists(c)), None)
    if bmc is None:
        r = subprocess.run(["which", "bmc-tools"], capture_output=True, text=True)
        bmc = r.stdout.strip() if r.returncode == 0 else None
    if bmc is None:
        raise FileNotFoundError(
            "bmc-tools not found.\n"
            "Put bmc-tools.py next to this script, or:\n"
            "  git clone https://github.com/ANSSI-FR/bmc-tools.git")
    print(f"      bmc-tools: {bmc}")
    subprocess.run([sys.executable, bmc, "-s", bin_path, "-d", out_dir], check=True)
    return sorted(str(p) for p in Path(out_dir).glob("*.bmp")
                  if "collage" not in p.name)


# ── Loading & deduplication ──────────────────────────────────────────────────────

def load_tiles(bmp_paths):
    tiles, seen, dups = {}, {}, 0
    for path in bmp_paths:
        arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        key = arr.tobytes()
        if key in seen:
            dups += 1
        else:
            seen[key] = path
            tiles[os.path.basename(path)] = arr
    sizes = {n: (tiles[n].shape[1], tiles[n].shape[0]) for n in tiles}
    print(f"  {len(tiles)} unique tiles ({dups} duplicates removed)")
    return tiles, sizes


# ── Helper functions ──────────────────────────────────────────────────────────────

def is_dark(arr):
    """Dark background (terminal, console, dark mode)."""
    return float(arr.mean()) < 60.0


def bg_sample(arr):
    """Achtergrondkleur via hoekpixels (mediaan)."""
    h, w = arr.shape[:2]
    pts = np.array([arr[0,0], arr[0,w//2], arr[0,-1],
                    arr[h//2,0], arr[h//2,-1],
                    arr[-1,0], arr[-1,w//2], arr[-1,-1]], dtype=float)
    return np.median(pts, axis=0)


def junction_correlation(arr_a, arr_b, n_cols=4):
    """
    Pearson correlation of per-row brightness at the junction.

    Correct join: text continues -> high correlation (≈ +0.98).
    Gradient/chrome tile inserted wrongly: MAD low but correlation negative.
    Onverenigbare tiles: correlatie ≈ 0.

    Returns -2.0 for incompatible height (do not judge).
    """
    if arr_a.shape[0] != arr_b.shape[0]:
        return -2.0
    bg_a = bg_sample(arr_a)
    bg_b = bg_sample(arr_b)
    ra = np.abs(arr_a[:, -n_cols:, :].astype(float) - bg_a).sum(axis=(1, 2))
    rb = np.abs(arr_b[:,  :n_cols, :].astype(float) - bg_b).sum(axis=(1, 2))

    def norm(x):
        s = x.std()
        return (x - x.mean()) / s if s > 1e-6 else np.zeros_like(x)

    return float(np.dot(norm(ra), norm(rb)) / len(ra))


# ── Phase 1: Edge-match graph (v3: 3px MAD + mutual-best) ─────────────────────

def build_neighbor_graph(tiles, tile_sizes,
                         thr_normal=18.0, thr_terminal=28.0,
                         top_k=8, vstrip=8):
    """
    3-pixel MAD horizontal + 8px MAD/seam vertical.
    Donkere tiles: thr_terminal, overige: thr_normal.
    """
    names  = list(tiles.keys())
    N      = len(names)
    t_mask = np.array([is_dark(tiles[n]) for n in names], dtype=bool)

    by_height = defaultdict(list)
    for n in names:
        by_height[tile_sizes[n][1]].append(n)

    h_nbr = {n: [] for n in names}

    # Horizontaal: 3px MAD per hoogte-groep
    for h, members in by_height.items():
        M = len(members)
        if M < 2:
            continue
        arrs  = np.stack([tiles[n] for n in members]).astype(np.float32)
        is_t  = np.array([is_dark(tiles[n]) for n in members], dtype=bool)
        right = arrs[:, :, -3:, :].mean(axis=2)
        left  = arrs[:, :,  :3, :].mean(axis=2)
        CHUNK = 128
        for start in range(0, M, CHUNK):
            end  = min(start + CHUNK, M)
            diff = np.abs(right[start:end, np.newaxis, :, :] -
                          left [np.newaxis,        :,  :, :]).mean(axis=(2, 3))
            for ci, ai in enumerate(range(start, end)):
                sc  = diff[ci].copy(); sc[ai] = 9999.
                thr = float(thr_terminal) if is_t[ai] else float(thr_normal)
                good = np.where(sc < thr)[0]
                h_nbr[members[ai]] = sorted(
                    (float(sc[bi]), members[bi]) for bi in good)[:top_k]

    # Vertical: 8px MAD + line-profile seam for dark tiles
    def _lp(arr):
        fg = np.abs(arr.astype(np.float32) - np.array([12.,12.,12.])).sum(axis=2)
        p  = fg.sum(axis=1); mx = p.max()
        return (p / mx) if mx > 0 else p

    def _ev(arr, bot):
        rows = min(vstrip, arr.shape[0])
        p = arr[-rows:, :64, :] if bot else arr[:rows, :64, :]
        if p.shape[0] < vstrip:
            pad = ((vstrip-p.shape[0],0),(0,0),(0,0)) if bot \
                  else ((0,vstrip-p.shape[0]),(0,0),(0,0))
            p = np.pad(p, pad, mode="edge")
        return p.mean(axis=0).astype(np.float32)

    def _lpv(arr, bot, K=8):
        lp = _lp(arr); k = min(K, len(lp))
        p  = lp[-k:] if bot else lp[:k]
        return np.pad(p, (K-len(p),0) if bot else (0,K-len(p)),
                      mode="edge").astype(np.float32)

    bot_e  = np.stack([_ev(tiles[n],  True)  for n in names])
    top_e  = np.stack([_ev(tiles[n],  False) for n in names])
    bot_lp = np.stack([_lpv(tiles[n], True)  for n in names])
    top_lp = np.stack([_lpv(tiles[n], False) for n in names])

    v_nbr = {n: [] for n in names}
    CHUNK = 200
    for start in range(0, N, CHUNK):
        end  = min(start + CHUNK, N)
        diff = np.abs(bot_e[start:end,np.newaxis,:,:] -
                      top_e[np.newaxis,       :,:,:]).mean(axis=(2,3))
        seam = np.abs(bot_lp[start:end,np.newaxis,:] -
                      top_lp[np.newaxis,      :,  :]).mean(axis=2)
        for ci, ai in enumerate(range(start, end)):
            a   = names[ai]
            thr = float(thr_terminal) if t_mask[ai] else float(thr_normal)
            sc  = diff[ci].copy(); sc[ai] = 9999.
            if t_mask[ai]:
                sc = sc + 6.0 * seam[ci]
            good = np.where(sc < thr)[0]
            v_nbr[a] = sorted((float(sc[bi]), names[bi]) for bi in good)[:top_k]

    he  = sum(len(v) for v in h_nbr.values())
    ve  = sum(len(v) for v in v_nbr.values())
    iso = sum(1 for n in names if not h_nbr[n])
    print(f"  H-edges: {he}, V-edges: {ve}, isolated: {iso}")
    return h_nbr, v_nbr


# ── Phase 1: Chains (mutual-best) ──────────────────────────────────────────────

def build_chains_mutual(h_nbr, names):
    lo, ro = {}, {}
    for a, nbrs in h_nbr.items():
        if not nbrs: continue
        s, b = nbrs[0]; lo[a] = b
        if b not in ro or s < ro[b][0]: ro[b] = (s, a)
    confirmed = {a: b for a, b in lo.items()
                 if b in ro and ro[b][1] == a}
    has_left = set(confirmed.values())
    visited, chains = set(), []
    for start in names:
        if start in visited or start in has_left or start not in confirmed:
            continue
        ch = [start]; visited.add(start); cur = start
        while cur in confirmed:
            nxt = confirmed[cur]
            if nxt in visited: break
            ch.append(nxt); visited.add(nxt); cur = nxt
        if len(ch) >= 2: chains.append(ch)
    for n in names:
        if n not in visited: visited.add(n); chains.append([n])
    chains.sort(key=len, reverse=True)
    return chains


# ── Phase 2: Correlation validation and repair ─────────────────────────────────

def validate_and_repair_chains(chains, tiles, h_nbr,
                                corr_threshold=0.3, n_cols=4):
    """
    Validate each junction via per-row brightness correlation.

    Kernbevinding uit analyse:
    - Correct junctions: corr ≈ 0.98 (text continues)
    - Gradient/chrome wrong tiles: MAD low (score ~0.1) BUT corr negative (~-0.97)
    - Onverenigbare tiles: corr ≈ 0.09

    Werkwijze per junction:
    1. Bereken correlatie
    2. Als corr < threshold: splits keten op dit punt
    3. For the loose end, find the best candidate with higher correlation
    4. Re-join if an improved candidate is found

    Only dark (terminal) tiles are repaired.
    """
    new_chains = []

    for chain in chains:
        dark_frac = sum(1 for t in chain if is_dark(tiles[t])) / len(chain)
        if len(chain) < 3 or dark_frac < 0.6:
            new_chains.append(chain)
            continue

        # Evalueer alle junctions
        corrs = []
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            if tiles[a].shape[0] == tiles[b].shape[0] == 64:
                c = junction_correlation(tiles[a], tiles[b], n_cols)
            else:
                c = 1.0  # non-standard height: do not judge
            corrs.append(c)

        if not corrs or min(corrs) >= corr_threshold:
            new_chains.append(chain)
            continue

        # Splits op slechtste junction
        worst_idx = int(np.argmin(corrs))
        left_part  = chain[:worst_idx + 1]
        right_part = chain[worst_idx + 1:]

        # Zoek betere rechter-buur voor left_part[-1]
        a_last = left_part[-1]
        placed  = set(left_part) | set(right_part)
        best_alt, best_alt_corr = None, corr_threshold

        for _, cand in (h_nbr.get(a_last) or [])[1:6]:
            if cand in placed: continue
            if tiles[a_last].shape[0] != tiles[cand].shape[0]: continue
            c = junction_correlation(tiles[a_last], tiles[cand], n_cols)
            if c > best_alt_corr:
                best_alt_corr = c
                best_alt = cand

        if best_alt is not None:
            new_chains.append(left_part + [best_alt])
        else:
            new_chains.append(left_part)
        new_chains.append(right_part)

    new_chains.sort(key=len, reverse=True)
    return new_chains


# ── Vertical groups ─────────────────────────────────────────────────────────

def build_groups(chains, v_nbr, tile_sizes):
    chain_of = {t: ci for ci, ch in enumerate(chains) for t in ch}
    below_a, above_b = {}, {}
    for a, nbrs in v_nbr.items():
        if nbrs:
            s, b = nbrs[0]; below_a[a] = b
            if b not in above_b or s < above_b[b][0]: above_b[b] = (s, a)
    confirmed_v = {a: b for a, b in below_a.items()
                   if b in above_b and above_b[b][1] == a}
    chain_below, chain_above = {}, {}
    for ci, ch in enumerate(chains):
        vote = defaultdict(int)
        for t in ch:
            if t in confirmed_v:
                cj = chain_of.get(confirmed_v[t])
                if cj is not None and cj != ci: vote[cj] += 1
        if vote:
            best = max(vote, key=vote.get)
            chain_below[ci] = best; chain_above[best] = ci
    has_above = set(chain_above.keys())
    visited_c, groups = set(), []
    for sci in range(len(chains)):
        if sci in has_above or sci not in chain_below or sci in visited_c:
            continue
        g = [sci]; visited_c.add(sci); cur = sci
        while cur in chain_below:
            nxt = chain_below[cur]
            if nxt in visited_c: break
            g.append(nxt); visited_c.add(nxt); cur = nxt
        if len(g) >= 2: groups.append(g)
    for ci in range(len(chains)):
        if ci not in visited_c: visited_c.add(ci); groups.append([ci])
    groups.sort(key=lambda g: sum(len(chains[ci]) for ci in g), reverse=True)
    return groups


# ── Rendering ──────────────────────────────────────────────────────────────────

def _row_array(ci, chains, tiles, tile_sizes):
    """Render a chain (row) as a numpy array, tiles joined horizontally."""
    ts = chains[ci]
    w  = sum(tile_sizes[t][0] for t in ts)
    h  = max((tile_sizes[t][1] for t in ts), default=64)
    im = np.zeros((h, w, 3), np.uint8)
    x  = 0
    for t in ts:
        a = tiles[t]
        im[:a.shape[0], x:x + a.shape[1]] = a
        x += tile_sizes[t][0]
    return im


def _best_xshift(top, bot, max_shift=900, band=6, min_overlap=64, info_min=8.0):
    """
    Determine the horizontal offset of the bottom row relative to the top
    by the seam band (last `band` rows of top vs first `band` of bottom)
    over elkaar te schuiven en de MAD op de overlap te minimaliseren.

    Only offsets where the overlap has enough detail (std >= info_min)
    count, so flat/black edges do not align falsely.

    Returns (shift, score, reliable).
    """
    tb = top[-band:, :, :].astype(np.float32).mean(0)   # (Wt,3)
    bb = bot[:band, :, :].astype(np.float32).mean(0)     # (Wb,3)
    Wt, Wb = tb.shape[0], bb.shape[0]
    best = (0, 1e9, False)
    for s in range(-max_shift, max_shift + 1, 2):
        lo = max(0, s); hi = min(Wt, s + Wb)
        if hi - lo < min_overlap:
            continue
        seg_t = tb[lo:hi]
        seg_b = bb[lo - s:hi - s]
        if max(seg_t.std(), seg_b.std()) < info_min:
            continue
        mad = float(np.abs(seg_t - seg_b).mean())
        if mad < best[1]:
            best = (s, mad, True)
    return best


def render_group(group, chains, tiles, tile_sizes,
                 align=True, accept=22.0, max_shift=900):
    """
    Compose a group (stacked chains) into a single image.

    align=True  -> each row gets its true horizontal offset relative to the
                   row above via seam alignment (_best_xshift); this places
                   content at the correct screen position instead of left-aligned.
    align=False -> old behavior: all rows left-aligned (x=0).

    Rows for which no reliable alignment is found (too flat, or MAD
    above `accept`) fall back to the offset of the row above.
    """
    rows = [_row_array(ci, chains, tiles, tile_sizes) for ci in group]
    if not rows:
        return Image.new("RGB", (64, 64), (20, 20, 30))

    xs = [0]
    ys = [0]
    for i in range(1, len(rows)):
        s = 0
        if align:
            shift, score, ok = _best_xshift(rows[i - 1], rows[i], max_shift)
            if ok and score <= accept:
                s = shift
        xs.append(xs[i - 1] + s)
        ys.append(ys[i - 1] + rows[i - 1].shape[0])

    minx = min(xs)
    xs = [x - minx for x in xs]
    W = max(xs[i] + rows[i].shape[1] for i in range(len(rows)))
    H = sum(r.shape[0] for r in rows)

    canvas = np.full((H, W, 3), (20, 20, 30), np.uint8)
    for i, r in enumerate(rows):
        canvas[ys[i]:ys[i] + r.shape[0], xs[i]:xs[i] + r.shape[1]] = r
    return Image.fromarray(canvas, "RGB")


# ── Terminal-text pass ───────────────────────────────────────────────────────
#
# Console/PowerShell tiles are dark with thin text lines. The plain 3px MAD
# matches poorly there (black matches black). This pass:
#   1. selects dark text tiles (enough ink, sharp horizontal edges),
#   2. gates candidate neighbors on LINE REGISTRATION: the per-row ink profiles on
#      both sides of the seam must correlate (text lines at the same height),
#   3. only then ranks by seam-color MAD (glyph continuity),
#   4. builds mutual-best chains and suppresses consecutive near-duplicates
#      (same command from different frames -> "stutter").
# Result: readable text-line strips (commands/output) instead of black noise.

def _term_bg(a):
    g = a.reshape(-1, 3).astype(np.float32)
    lum = g.mean(1)
    idx = np.argsort(lum)[:int(len(lum) * 0.4)]
    return np.median(g[idx], axis=0)


def _term_feats(a, K=4):
    bg = _term_bg(a)
    ink = np.abs(a.astype(np.float32) - bg).sum(2)          # afstand-tot-bg
    g = a.astype(np.float32).mean(2)
    sharp = float((np.abs(np.diff(g, axis=1)) > 40).mean())  # sharp glyph edges
    return {
        "bg": bg,
        "rp": ink[:, -K:].mean(1),                  # right edge row profile (H,)
        "lp": ink[:,  :K].mean(1),                  # left  edge row profile (H,)
        "rcol": a[:, -2:, :].astype(np.float32).mean(1),   # rechter 2 kol (H,3)
        "lcol": a[:,  :2, :].astype(np.float32).mean(1),   # linker  2 kol (H,3)
        "blue": float(bg[2] - bg[0]),
        "ink": float(ink.mean()),
        "sharp": sharp,
    }


def reconstruct_terminal_strips(tiles, tile_sizes,
                                corr_gate=0.83, dedup_mad=3.5,
                                stutter_mad=6.0, min_len=3, std_h=64):
    """
    Returns a list of text-line strips (each a list of tile names, left->right),
    sorted by length. Works at the standard console height (64px).
    """
    names = [n for n in tiles
             if tile_sizes[n][1] == std_h and tile_sizes[n][0] == 64]
    F = {}
    term = []
    for n in names:
        f = _term_feats(tiles[n])
        if 4 < f["ink"] < 140 and f["bg"].mean() < 75 and f["sharp"] > 0.018:
            F[n] = f
            term.append(n)
    if len(term) < 2:
        return []

    # near-duplicate suppression (same text from different frames)
    sig = np.stack([F[n]["rp"] for n in term]).round(0)
    buck = defaultdict(list)
    for i, n in enumerate(term):
        buck[tuple(sig[i, ::8].astype(int))].append(i)
    used = np.zeros(len(term), bool)
    keep = []
    for ids in buck.values():
        for ai in ids:
            if used[ai]:
                continue
            keep.append(term[ai]); used[ai] = True
            for bi in ids:
                if used[bi]:
                    continue
                if np.abs(tiles[term[ai]].astype(np.int16)
                          - tiles[term[bi]].astype(np.int16)).mean() < dedup_mad:
                    used[bi] = True
    term = keep

    RP = np.stack([F[n]["rp"] for n in term])
    LP = np.stack([F[n]["lp"] for n in term])
    RC = np.stack([F[n]["rcol"] for n in term])
    LC = np.stack([F[n]["lcol"] for n in term])
    BL = np.array([F[n]["blue"] for n in term])

    def zn(M):
        m = M - M.mean(1, keepdims=True)
        s = M.std(1, keepdims=True); s[s < 1e-3] = 1
        return m / s
    RPz, LPz = zn(RP), zn(LP)
    M = len(term)

    h_nbr = {n: [] for n in term}
    CH = 256
    for s in range(0, M, CH):
        e = min(s + CH, M)
        corr = (RPz[s:e] @ LPz.T) / RPz.shape[1]
        madc = np.abs(RC[s:e, None] - LC[None]).mean((2, 3))
        bld = np.abs(BL[s:e, None] - BL[None])
        score = madc - 4.0 * corr
        score[(corr < corr_gate) | (bld > 10)] = 1e9
        for ci, ai in enumerate(range(s, e)):
            score[ci, ai] = 1e9
            order = np.argsort(score[ci])[:6]
            h_nbr[term[ai]] = [(float(score[ci, j]), term[j])
                               for j in order if score[ci, j] < 1e8]

    lo, ro = {}, {}
    for a, nb in h_nbr.items():
        if nb:
            sc, b = nb[0]; lo[a] = b
            if b not in ro or sc < ro[b][0]:
                ro[b] = (sc, a)
    conf = {a: b for a, b in lo.items() if b in ro and ro[b][1] == a}

    def near(a, b):
        return np.abs(tiles[a].astype(np.int16)
                      - tiles[b].astype(np.int16)).mean() < stutter_mad

    has_left = set(conf.values())
    vis, strips = set(), []
    for st in term:
        if st in vis or st in has_left or st not in conf:
            continue
        ch = [st]; vis.add(st); cur = st
        while cur in conf:
            nx = conf[cur]
            if nx in vis:
                break
            vis.add(nx)
            if not near(cur, nx):       # skip stutter duplicates
                ch.append(nx)
            cur = nx
        if len(ch) >= min_len:
            strips.append(ch)
    strips.sort(key=len, reverse=True)
    return strips


def render_strip(strip, tiles, tile_sizes):
    w = sum(tile_sizes[t][0] for t in strip)
    h = max(tile_sizes[t][1] for t in strip)
    im = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    for t in strip:
        im.paste(Image.fromarray(tiles[t], "RGB"), (x, 0))
        x += tile_sizes[t][0]
    return im


def _strip_array(strip, tiles, tile_sizes):
    w = sum(tile_sizes[t][0] for t in strip)
    h = max(tile_sizes[t][1] for t in strip)
    im = np.zeros((h, w, 3), np.uint8)
    x = 0
    for t in strip:
        a = tiles[t]
        im[:a.shape[0], x:x + a.shape[1]] = a
        x += tile_sizes[t][0]
    return im


def _vseam(top, bot, band=6, max_shift=700, info_min=10.0, min_ov=80):
    """Best horizontal offset of bottom vs top strip via the
       horizontal seam (bottom band of top vs top band of bottom)."""
    tb = top[-band:, :, :].astype(np.float32).mean(0)
    bb = bot[:band, :, :].astype(np.float32).mean(0)
    Wt, Wb = tb.shape[0], bb.shape[0]
    best = (0, 1e9, False)
    for s in range(-max_shift, max_shift + 1, 2):
        lo = max(0, s); hi = min(Wt, s + Wb)
        if hi - lo < min_ov:
            continue
        st = tb[lo:hi]; sb = bb[lo - s:hi - s]
        if max(st.std(), sb.std()) < info_min:
            continue
        mad = float(np.abs(st - sb).mean())
        if mad < best[1]:
            best = (s, mad, True)
    return best


def stack_terminal_blocks(strips, tiles, tile_sizes, accept=26.0, min_rows=2):
    """
    Stack text-line strips vertically into multi-line console blocks.
    Mutual-best on the horizontal seam, with a horizontal offset per line
    (so lines land at their true column position). Returns PIL images.
    """
    S = [_strip_array(s, tiles, tile_sizes) for s in strips]
    N = len(S)
    below, above = {}, {}
    for i in range(N):
        best = (-1, 1e9, 0)
        for j in range(N):
            if i == j:
                continue
            sh, mad, ok = _vseam(S[i], S[j])
            if ok and mad < best[1]:
                best = (j, mad, sh)
        if best[0] >= 0 and best[1] < accept:
            below[i] = (best[0], best[2], best[1])
            j = best[0]
            if j not in above or best[1] < above[j][2]:
                above[j] = (i, best[2], best[1])
    conf = {i: below[i] for i in below if above.get(below[i][0], (None,))[0] == i}
    has_above = set(v[0] for v in conf.values())
    vis, blocks = set(), []
    for st in range(N):
        if st in vis or st in has_above or st not in conf:
            continue
        seq = [(st, 0)]; vis.add(st); cur = st; acc = 0
        while cur in conf:
            nx, sh, _ = conf[cur]
            if nx in vis:
                break
            acc += sh; seq.append((nx, acc)); vis.add(nx); cur = nx
        if len(seq) >= min_rows:
            blocks.append(seq)
    blocks.sort(key=len, reverse=True)

    imgs = []
    for seq in blocks:
        minx = min(x for _, x in seq)
        xs = [x - minx for _, x in seq]
        W = max(xs[k] + S[seq[k][0]].shape[1] for k in range(len(seq)))
        H = sum(S[i].shape[0] for i, _ in seq)
        cv = np.zeros((H, W, 3), np.uint8); y = 0
        for k, (i, _) in enumerate(seq):
            r = S[i]; cv[y:y + r.shape[0], xs[k]:xs[k] + r.shape[1]] = r
            y += r.shape[0]
        imgs.append((Image.fromarray(cv), len(seq)))
    return imgs


# ── 2D best-buddy jigsaw assembly ───────────────────────────────────────────
#
# Instead of building rows and stacking them (1D then 1D), this method grows a
# 2D region: each new tile is judged against ALL of its already-placed neighbors
# (left/right/above/below) at once. A near-duplicate usually fits on one edge but
# not on the perpendicular one, so the joint edge agreement resolves much of the
# ambiguity that 1D chains leave standing. This is the classic approach from
# automatic jigsaw-puzzle solvers (best-buddy + greedy placement) applied to the
# cache tiles.

_JDIRS = {"R": (1, 0), "L": (-1, 0), "D": (0, 1), "U": (0, -1)}
_JINV = {"R": "L", "L": "R", "D": "U", "U": "D"}


def _jigsaw_candidates(idx_list, tiles, tile_sizes, k=5,
                       h_gate=0.80, v_gate=0.55):
    """Top-k compatible neighbors per direction (R/L/D/U) for the given tiles."""
    names = [n for n in idx_list if tile_sizes[n] == (64, 64)]
    if len(names) < 2:
        return None, names
    A = np.stack([tiles[n].astype(np.float32) for n in names])
    bg = np.stack([_term_bg(tiles[n]) for n in names])
    blue = np.array([float(b[2] - b[0]) for b in bg])
    ink = np.abs(A - bg[:, None, None, :]).sum(3)
    K = 4
    RP = ink[:, :, -K:].mean(2); LP = ink[:, :, :K].mean(2)
    BP = ink[:, -K:, :].mean(1); TP = ink[:, :K, :].mean(1)
    Rc = A[:, :, -2:, :].mean(2); Lc = A[:, :, :2, :].mean(2)
    Bc = A[:, -2:, :, :].mean(1); Tc = A[:, :2, :, :].mean(1)

    def zn(M):
        m = M - M.mean(1, keepdims=True)
        s = M.std(1, keepdims=True); s[s < 1e-3] = 1
        return m / s
    RPz, LPz, BPz, TPz = zn(RP), zn(LP), zn(BP), zn(TP)
    Nn = len(names)

    def topk(Pa, Pb, Ca, Cb, gate):
        out = {}
        CH = 128
        for s in range(0, Nn, CH):
            e = min(s + CH, Nn)
            corr = (Pa[s:e] @ Pb.T) / Pa.shape[1]
            mad = np.abs(Ca[s:e, None] - Cb[None]).mean((2, 3))
            bd = np.abs(blue[s:e, None] - blue[None])
            sc = mad - 4.0 * corr
            sc[(corr < gate) | (bd > 10)] = 1e9
            for ci, ai in enumerate(range(s, e)):
                sc[ci, ai] = 1e9
                order = np.argsort(sc[ci])[:k]
                out[ai] = [(float(sc[ci, j]), int(j)) for j in order if sc[ci, j] < 1e8]
        return out

    candR = topk(RPz, LPz, Rc, Lc, h_gate)
    candD = topk(BPz, TPz, Bc, Tc, v_gate)
    candL, candU = {}, {}
    for a, lst in candR.items():
        for sc, b in lst:
            candL.setdefault(b, []).append((sc, a))
    for a, lst in candD.items():
        for sc, b in lst:
            candU.setdefault(b, []).append((sc, a))
    for d in (candL, candU):
        for b in d:
            d[b] = sorted(d[b])[:k]
    return {"R": candR, "L": candL, "D": candD, "U": candU}, names


def scroll_dedup(idx_list, tiles, tile_sizes, maxd=12, step=2,
                 mad_thr=6.0, hd_thr=0.02, bucket_cap=200):
    """
    Merge tiles that are 2D shifts of each other (same text, different scroll
    or horizontal position). Console text is cached at many positions across the
    session as separate tiles; these are each other's perfect neighbors and
    cause repeated columns ("stutter") in the assembly.

    STRICT merging: only when the tiles are nearly pixel-identical at their best
    2D shift (low MAD and almost no strongly-differing pixels), so lines that
    merely look alike in layout (e.g. -Identity "Natasha" vs "Maria") are NOT
    merged. Returns one representative (most ink) per cluster.
    """
    from collections import defaultdict
    names = [n for n in idx_list if tile_sizes[n] == (64, 64)]
    G = {n: tiles[n].astype(np.float32).mean(2) for n in names}
    ink = {n: float(np.abs(tiles[n].astype(np.float32) - _term_bg(tiles[n])).sum(2).mean())
           for n in names}

    def match2d(a, b):
        best = (1e9, 1.0)
        H, W = a.shape
        for dy in range(-maxd, maxd + 1, step):
            ay0 = max(0, dy); ay1 = min(H, H + dy)
            by0 = max(0, -dy); by1 = by0 + (ay1 - ay0)
            if ay1 - ay0 < 44:
                continue
            for dx in range(-maxd, maxd + 1, step):
                ax0 = max(0, dx); ax1 = min(W, W + dx)
                bx0 = max(0, -dx); bx1 = bx0 + (ax1 - ax0)
                if ax1 - ax0 < 44:
                    continue
                diff = np.abs(a[ay0:ay1, ax0:ax1] - b[by0:by1, bx0:bx1])
                m = diff.mean()
                if m < best[0]:
                    best = (float(m), float((diff > 40).mean()))
        return best

    def thumb(g):
        im = Image.fromarray(g.astype(np.uint8)).resize((4, 4))
        return tuple((np.array(im) // 44).flatten())
    buck = defaultdict(list)
    for n in names:
        buck[thumb(G[n])].append(n)

    parent = {n: n for n in names}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for grp in buck.values():
        g = grp[:bucket_cap]
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                m, hd = match2d(G[g[i]], G[g[j]])
                if m < mad_thr and hd < hd_thr:
                    union(g[i], g[j])

    clusters = defaultdict(list)
    for n in names:
        clusters[find(n)].append(n)
    return [max(c, key=lambda n: ink[n]) for c in clusters.values()]


def jigsaw_blocks(idx_list, tiles, tile_sizes, accept=22.0, min_block=6):
    """Assemble 2D blocks from the given tiles; returns a list of PIL images."""
    CAND, names = _jigsaw_candidates(idx_list, tiles, tile_sizes)
    if CAND is None:
        return []

    def best(lst):
        return lst[0] if lst else None

    # best-buddy seeds (mutual top-1 in R of D)
    seeds = []
    for a, lst in CAND["R"].items():
        b = best(lst)
        if b and best(CAND["L"].get(b[1], [])) and best(CAND["L"][b[1]])[1] == a:
            seeds.append((b[0], a, b[1], "R"))
    for a, lst in CAND["D"].items():
        b = best(lst)
        if b and best(CAND["U"].get(b[1], [])) and best(CAND["U"][b[1]])[1] == a:
            seeds.append((b[0], a, b[1], "D"))
    seeds.sort()

    def nb_score(idx, gx, gy, occ):
        tot, n = 0.0, 0
        for d, (dx, dy) in _JDIRS.items():
            p = (gx + dx, gy + dy)
            if p in occ:
                j = occ[p]
                m = dict((b, s) for s, b in CAND[d].get(idx, []))
                if j not in m:
                    return None
                tot += m[j]; n += 1
        return tot / n if n else None

    placed_global = set()
    blocks = []
    for _, a, b, d in seeds:
        if a in placed_global or b in placed_global:
            continue
        dx, dy = _JDIRS[d]
        occ = {(0, 0): a, (dx, dy): b}
        local = {a, b}
        while True:
            slots = set()
            for (gx, gy) in occ:
                for _, (ddx, ddy) in _JDIRS.items():
                    p = (gx + ddx, gy + ddy)
                    if p not in occ:
                        slots.add(p)
            cands = []
            for (sx, sy) in slots:
                pool = set()
                for d2, (ddx, ddy) in _JDIRS.items():
                    npos = (sx + ddx, sy + ddy)
                    if npos in occ:
                        j = occ[npos]
                        for _, t in CAND[_JINV[d2]].get(j, []):
                            pool.add(t)
                for t in pool:
                    if t in placed_global or t in local:
                        continue
                    sc = nb_score(t, sx, sy, occ)
                    if sc is not None:
                        cands.append((sc, t, sx, sy))
            if not cands:
                break
            cands.sort()
            sc, t, gx, gy = cands[0]
            if sc > accept:
                break
            occ[(gx, gy)] = t; local.add(t)
        if len(local) >= min_block:
            placed_global |= local
            blocks.append(dict(occ))
    blocks.sort(key=len, reverse=True)

    imgs = []
    for occ in blocks:
        xs = [p[0] for p in occ]; ys = [p[1] for p in occ]
        x0, y0 = min(xs), min(ys)
        W = (max(xs) - x0 + 1) * 64; H = (max(ys) - y0 + 1) * 64
        im = Image.new("RGB", (W, H), (15, 15, 20))
        for (gx, gy), t in occ.items():
            im.paste(Image.fromarray(tiles[names[t]], "RGB"),
                     ((gx - x0) * 64, (gy - y0) * 64))
        imgs.append((im, len(occ)))
    return imgs


# ── OCR transcript ────────────────────────────────────────────────────────────
#
# OCR (Tesseract) turns reconstructed console text into searchable text and
# flags forensically relevant lines. Note: OCR is only as good as its input.
# On the AUTOMATIC strips the text is partly garbled (the strips stutter);
# on a CLEAN reconstruction (e.g. assembled by hand) the same pass yields
# neat, report-ready text. So also use `--transcribe <image>` to transcribe
# a cleanly assembled image.

_OCR_PATTERNS = [
    ("email",     r"[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("forward",   r"Deliver(?:To)?\w*|Forwarding?\w*|SMTPAddress"),
    ("mailbox",   r"(?:Set|Get|New|Remove)-?Mailbox|-Identity"),
    ("imperson",  r"Impersonat\w*"),
    ("host",      r"exchange\d+|shieldbase|[A-Za-z0-9-]+\.[A-Za-z0-9-]+\.(?:com|local|net)"),
    ("path",      r"C:\\\\|system32|PS\s*C:|>\s*$"),
    ("bool",      r"\$true|\$false"),
]


def _ocr_prep(img, scale=5):
    import numpy as _np
    g = _np.array(img.convert("L"), _np.float32)
    if g.mean() < 128:                      # light-on-dark -> invert
        g = 255 - g
    rng = float(g.max() - g.min())
    g = (g - g.min()) / (rng + 1e-3) * 255
    out = Image.fromarray(g.astype("uint8"))
    return out.resize((out.width * scale, out.height * scale), Image.LANCZOS)


def _ocr_flags(line):
    import re as _re
    return [name for name, rx in _OCR_PATTERNS if _re.search(rx, line, _re.I)]


def ocr_lines(image, psm=7, scale=5, min_len=4):
    """OCR a single PIL image -> list of cleaned lines."""
    try:
        import pytesseract
    except ImportError:
        return None
    cfg = f"--psm {psm} -c preserve_interword_spaces=1"
    try:
        txt = pytesseract.image_to_string(_ocr_prep(image, scale), config=cfg)
    except (getattr(pytesseract, "TesseractNotFoundError", OSError),
            EnvironmentError, OSError):
        # pytesseract is installed but the 'tesseract' binary is missing/not on PATH
        return None
    out = []
    for ln in txt.splitlines():
        ln = ln.rstrip()
        if len(ln.strip()) >= min_len:
            out.append(ln)
    return out


def ocr_extract_indicators(full_text):
    """Trek e-mails, hosts, commando's en identiteiten uit de volledige OCR-tekst."""
    import re as _re
    def find(rx):
        return sorted(set(m.group(0).strip()
                          for m in _re.finditer(rx, full_text, _re.I)))
    # identities: -Identity "<name>"  and standalone "First Last" between
    # (eventueel krullende) aanhalingstekens
    Q = "\"\u201c\u201d\u2018\u2019'"
    idents = set()
    for m in _re.finditer(rf'-Identity\s*[{Q}]([^{Q}]{{2,40}})', full_text, _re.I):
        idents.add(m.group(1).strip())
    for m in _re.finditer(rf'[{Q}]([A-Z][a-z]+\s+[A-Z][a-z]+)[{Q}]?', full_text):
        idents.add(m.group(1).strip())
    # only plausible names: letters/space/-./' , no digits, 1-3 words
    def _name_ok(s):
        s = s.strip(" ,.\"'")
        return bool(_re.fullmatch(r"[A-Za-z][A-Za-z .'\-]{2,38}", s)) and \
               1 <= len(s.split()) <= 3 and not any(c.isdigit() for c in s)
    idents = {s.strip(" ,.\"'") for s in idents if _name_ok(s)}
    return {
        "emails":   find(r"[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "hosts":    find(r"\b(?:exchange\d+\.)?[A-Za-z0-9-]+\.(?:com|local|net)\b"),
        "commands": find(r"(?:Set|Get|New|Remove)-?Mailbox|DeliverToMailboxAndForward"
                         r"|Forwarding?SMTPAddress|ForwardingAddress|ApplicationImpersonation"),
        "identities": sorted(idents),
    }


def build_transcript(image_specs, out_path, light_dedup=True):
    """
    image_specs: list of (PIL image, psm, source-label).
    Writes a transcript with (1) forensically flagged lines,
    (2) recovered indicators, (3) all remaining lines. Returns a dict with counts.
    """
    import difflib
    all_lines = []          # (line, flags, source)
    raw_text = []
    for img, psm, label in image_specs:
        lines = ocr_lines(img, psm=psm)
        if lines is None:
            return None     # tesseract/pytesseract not available
        for ln in lines:
            raw_text.append(ln)
            all_lines.append((ln, _ocr_flags(ln), label))

    # light dedup: drop near-substrings of longer lines (stutter)
    kept = []
    for ln, fl, src in sorted(all_lines, key=lambda x: -len(x[0])):
        if light_dedup and any(
                difflib.SequenceMatcher(None, ln, k[0]).ratio() > 0.86 for k in kept):
            continue
        kept.append((ln, fl, src))

    indicators = ocr_extract_indicators("\n".join(raw_text))
    flagged = [(l, f, s) for (l, f, s) in kept if f]
    rest = [(l, f, s) for (l, f, s) in kept if not f]

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("RDP Cache - OCR transcript\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("NOTE: OCR is only as good as the reconstruction. On automatic\n")
        fh.write("strips text may be garbled; on clean (manual) montages\n")
        fh.write("the transcription is reliable.\n\n")

        fh.write("── RECOVERED INDICATORS ──────────────────────────────────\n")
        for k, label in [("emails", "Email addresses"), ("hosts", "Hostnames"),
                         ("commands", "Commands"), ("identities", "Identities")]:
            vals = indicators.get(k) or ["(none)"]
            fh.write(f"{label}:\n")
            for v in vals:
                fh.write(f"    {v}\n")
            fh.write("\n")

        fh.write("── FORENSICALLY FLAGGED LINES ──────────────────────────\n")
        for l, f, s in sorted(flagged, key=lambda x: -len(x[0])):
            fh.write(f"[{'/'.join(f)}]  {l}\n")
        fh.write("\n── OTHER RECOGNIZED LINES ────────────────────────────────\n")
        for l, f, s in rest:
            fh.write(f"{l}\n")

    return {"flagged": len(flagged), "total": len(kept),
            "indicators": indicators}


# ── HTML viewer ───────────────────────────────────────────────────────────────

def build_viewer(entries, out_path):
    js = "const GROUPS=[\n" + ",\n".join(
        f"{{id:{e['id']},tiles:{e['tiles']},origW:{e['w']},origH:{e['h']},"
        f"rows:{e['rows']},pw:{e['pw']},ph:{e['ph']},term:{e.get('term',0)},"
        f"b64:\"{e['b64']}\"}}"
        for e in entries) + "\n];"

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>RDP Cache Reconstruction</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--border:#30363d;--accent:#58a6ff;
      --green:#3fb950;--yellow:#d29922;--red:#f85149;--text:#c9d1d9;--muted:#8b949e}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Consolas','Courier New',monospace}}
header{{background:var(--panel);border-bottom:1px solid var(--border);
  padding:12px 24px;display:flex;align-items:center;gap:16px}}
.logo{{font-size:11px;color:var(--red);letter-spacing:2px;text-transform:uppercase;font-weight:bold}}
header h1{{font-size:14px;color:var(--accent);font-weight:normal}}
.hstats{{margin-left:auto;display:flex;gap:16px;font-size:12px;color:var(--muted)}}.hstats b{{color:var(--green)}}
.toolbar{{background:var(--panel);border-bottom:1px solid var(--border);
  padding:8px 24px;display:flex;align-items:center;gap:10px;font-size:12px;flex-wrap:wrap}}
.toolbar label{{color:var(--muted)}}.sep{{width:1px;height:20px;background:var(--border);margin:0 4px}}
.btn{{background:var(--border);border:1px solid #444;color:var(--text);padding:4px 12px;
  border-radius:4px;cursor:pointer;font-size:12px;font-family:inherit}}
.btn:hover{{background:#444}}.btn.active{{background:var(--accent);color:#000;border-color:var(--accent)}}
select,input[type=range]{{background:#1c2128;border:1px solid var(--border);color:var(--text);
  padding:3px 8px;border-radius:4px;font-size:12px;font-family:inherit;accent-color:var(--accent)}}
.main{{display:flex;height:calc(100vh - 92px)}}
.sidebar{{width:220px;min-width:220px;background:var(--panel);border-right:1px solid var(--border);
  overflow-y:auto;padding:10px;display:flex;flex-direction:column;gap:6px}}
.sidebar h3{{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:4px}}
.thumb-item{{border:1px solid var(--border);border-radius:4px;overflow:hidden;cursor:pointer;
  transition:border-color .15s;background:#090d12}}
.thumb-item:hover{{border-color:var(--accent)}}.thumb-item.active{{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}}
.thumb-item img{{width:100%;display:block}}.thumb-label{{font-size:10px;color:var(--muted);padding:2px 6px}}.thumb-label b{{color:var(--green)}}
.canvas-wrap{{flex:1;overflow:auto;background:#090d12;padding:20px}}
.recon-container{{position:relative;display:inline-block}}
#recon-img{{display:block;image-rendering:auto;border:1px solid var(--border);cursor:zoom-in}}
.grid-overlay{{position:absolute;top:0;left:0;pointer-events:none;display:none}}
.info-panel{{width:200px;min-width:200px;background:var(--panel);border-left:1px solid var(--border);
  padding:12px;overflow-y:auto;font-size:11px}}
.info-panel h3{{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:8px}}
.info-row{{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid var(--border)}}
.info-row:last-child{{border:none}}.info-row .k{{color:var(--muted)}}.info-row .v{{color:var(--green)}}
.note{{background:#1c2128;border-left:3px solid var(--yellow);border-radius:0 4px 4px 0;
  padding:8px 10px;line-height:1.6;margin-top:10px;color:var(--muted)}}
#lb{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:1000;
  align-items:center;justify-content:center;cursor:zoom-out;flex-direction:column;gap:10px}}
#lb.open{{display:flex}}#lb img{{max-width:95vw;max-height:90vh;border:1px solid var(--border);border-radius:4px}}
#lbinfo{{color:var(--muted);font-size:12px}}
</style></head><body>
<header>
  <div class="logo">&#x1F512; IR Forensics</div>
  <h1>RDP Bitmap Cache &mdash; Reconstruction Viewer</h1>
  <div class="hstats"><span>Groups: <b id="ng">-</b></span>
    <span>Method: <b>3px MAD + correlation validation</b></span></div>
</header>
<div class="toolbar">
  <label>Zoom:</label><input type="range" id="zs" min="10" max="300" value="100">
  <span id="zv" style="color:var(--accent);min-width:36px">100%</span>
  <div class="sep"></div>
  <button class="btn active" onclick="setZoom(100)">100%</button>
  <button class="btn" onclick="fitWin()">Fit</button>
  <button class="btn" onclick="setZoom(50)">50%</button>
  <button class="btn" onclick="setZoom(200)">200%</button>
  <div class="sep"></div>
  <label>Filter:</label>
  <select id="tf" onchange="renderThumbs(this.value)">
    <option value="all">All groups</option>
    <option value="term">Terminal (text)</option>
    <option value="xl">Large (&ge;40)</option>
    <option value="large">Medium (20&ndash;39)</option>
    <option value="medium">Small-medium (5&ndash;19)</option>
    <option value="small">Small (&lt;5)</option>
  </select>
  <div class="sep"></div>
  <button class="btn" id="bg" onclick="toggleGrid()">Grid on</button>
  <div class="sep"></div>
  <span style="color:var(--muted);font-size:11px">Click &rarr; fullscreen &nbsp;|&nbsp; Esc to close</span>
</div>
<div class="main">
  <div class="sidebar"><h3>Reconstructions (<span id="vc">-</span>)</h3><div id="tl"></div></div>
  <div class="canvas-wrap" id="cw">
    <div class="recon-container" id="rc">
      <img id="recon-img" src="" alt=""><canvas class="grid-overlay" id="gc"></canvas>
    </div>
  </div>
  <div class="info-panel">
    <h3>Details</h3>
    <div id="ir"><div class="info-row"><span class="k">Select a group</span></div></div>
    <div class="note">
      <b style="color:var(--yellow)">Approach</b><br><br>
      <b>Phase 1</b> 3px MAD matching<br>
      &bull; Mutual-best confirmation<br>
      &bull; V-seam for dark tiles<br><br>
      <b>Phase 2</b> Correlation validation<br>
      &bull; Correct: text continues<br>
      &nbsp;&nbsp;corr ≈ +0.98<br>
      &bull; Gradient error: MAD low<br>
      &nbsp;&nbsp;but corr ≈ -0.97<br>
      &bull; Bad junctions are<br>
      &nbsp;&nbsp;split &amp; re-joined
    </div>
  </div>
</div>
<div id="lb" onclick="closeLb()"><img id="lbi" src=""><div id="lbinfo"></div></div>
<script>
{js}
let cur=null,showGrid=false,zoom=1.0;
document.getElementById('ng').textContent=GROUPS.length;
function renderThumbs(f){{
  const list=document.getElementById('tl');list.innerHTML='';let v=0;
  GROUPS.forEach((g,i)=>{{
    const t=g.tiles;
    const show=f==='all'||(f==='term'&&g.term)||(f==='xl'&&!g.term&&t>=40)||(f==='large'&&!g.term&&t>=20&&t<40)||(f==='medium'&&!g.term&&t>=5&&t<20)||(f==='small'&&!g.term&&t<5);
    if(!show)return;v++;
    const d=document.createElement('div');d.className='thumb-item';
    const badge=g.term?' <span style=\"color:var(--yellow)\">[TERM]</span>':'';
    d.innerHTML=`<img src="data:image/jpeg;base64,${{g.b64}}" loading="lazy"><div class="thumb-label"><b>#${{i+1}}</b>${{badge}} &mdash; ${{g.tiles}} tiles<br><span>${{g.origW}}&times;${{g.origH}}px &middot; ${{g.rows}} rows</span></div>`;
    d.onclick=()=>show_g(i,d);list.appendChild(d);
  }});
  document.getElementById('vc').textContent=v;
}}
function show_g(i,el){{
  document.querySelectorAll('.thumb-item').forEach(x=>x.classList.remove('active'));
  el?.classList.add('active');cur=GROUPS[i];
  const img=document.getElementById('recon-img');
  img.src='data:image/jpeg;base64,'+cur.b64;
  img.style.width=Math.round(cur.pw*zoom)+'px';img.style.height=Math.round(cur.ph*zoom)+'px';
  document.getElementById('ir').innerHTML=`
    <div class="info-row"><span class="k">Group</span><span class="v">#${{i+1}}</span></div>
    <div class="info-row"><span class="k">Tiles</span><span class="v">${{cur.tiles}}</span></div>
    <div class="info-row"><span class="k">Size</span><span class="v">${{cur.origW}}&times;${{cur.origH}}px</span></div>
    <div class="info-row"><span class="k">Rows</span><span class="v">${{cur.rows}}</span></div>
    <div class="info-row"><span class="k">Columns</span><span class="v">~${{Math.round(cur.origW/64)}}</span></div>`;
  if(showGrid)drawGrid();
}}
function setZoom(p){{zoom=p/100;document.getElementById('zs').value=p;document.getElementById('zv').textContent=p+'%';
  if(cur){{document.getElementById('recon-img').style.width=Math.round(cur.pw*zoom)+'px';document.getElementById('recon-img').style.height=Math.round(cur.ph*zoom)+'px';}}if(showGrid)drawGrid();}}
function fitWin(){{if(!cur)return;setZoom(Math.round((document.getElementById('cw').clientWidth-40)/cur.pw*100));}}
document.getElementById('zs').oninput=e=>setZoom(+e.target.value);
function toggleGrid(){{showGrid=!showGrid;const b=document.getElementById('bg');b.textContent=showGrid?'Grid off':'Grid on';b.classList.toggle('active',showGrid);document.getElementById('gc').style.display=showGrid?'block':'none';if(showGrid)drawGrid();}}
function drawGrid(){{if(!cur)return;const img=document.getElementById('recon-img'),gc=document.getElementById('gc');gc.width=img.offsetWidth;gc.height=img.offsetHeight;gc.style.width=gc.width+'px';gc.style.height=gc.height+'px';const ctx=gc.getContext('2d'),tw=64*(cur.pw/cur.origW)*zoom;ctx.clearRect(0,0,gc.width,gc.height);ctx.strokeStyle='rgba(88,166,255,0.3)';ctx.lineWidth=0.5;for(let x=0;x<=gc.width;x+=tw){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,gc.height);ctx.stroke();}}for(let y=0;y<=gc.height;y+=tw){{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(gc.width,y);ctx.stroke();}}}}
document.getElementById('recon-img').onclick=()=>{{if(!cur)return;document.getElementById('lbi').src='data:image/jpeg;base64,'+cur.b64;document.getElementById('lbinfo').textContent=cur.origW+'×'+cur.origH+'px — '+cur.tiles+' tiles';document.getElementById('lb').classList.add('open');}};
function closeLb(){{document.getElementById('lb').classList.remove('open');}}
document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeLb();}});
renderThumbs('all');
setTimeout(()=>{{const f=document.querySelector('.thumb-item');if(f){{show_g(0,f);fitWin();}}}},100);
</script></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="RDP Bitmap Cache reconstruction tool")
    p.add_argument("bin_file", nargs="?")
    p.add_argument("--out",                 default="./rdp_output")
    p.add_argument("--top",                 type=int,   default=60)
    p.add_argument("--threshold",           type=float, default=18.0)
    p.add_argument("--threshold-terminal",  type=float, default=28.0)
    p.add_argument("--corr-threshold",      type=float, default=0.3,
                   help="correlation threshold for junction validation (default: 0.3)")
    p.add_argument("--no-viewer",           action="store_true")
    p.add_argument("--no-banner",           action="store_true",
                   help="do not print the startup banner")
    p.add_argument("--preview-width",       type=int,   default=920)
    p.add_argument("--no-align",            action="store_true",
                   help="left-align rows instead of using the true horizontal offset")
    p.add_argument("--align-accept",        type=float, default=22.0,
                   help="max seam MAD to trust a horizontal alignment")
    p.add_argument("--no-terminal",         action="store_true",
                   help="skip the terminal-text pass")
    p.add_argument("--term-top",            type=int,   default=60,
                   help="max number of terminal strips in the viewer")
    p.add_argument("--term-min",            type=int,   default=2,
                   help="min tiles per terminal strip (lower = more material to stack)")
    p.add_argument("--no-ocr",              action="store_true",
                   help="do not generate an OCR transcript of the terminal reconstruction")
    p.add_argument("--jigsaw",              action="store_true",
                   help="2D best-buddy assembly of the terminal tiles (extra blocks)")
    p.add_argument("--jigsaw-all",          action="store_true",
                   help="2D best-buddy assembly of ALL content tiles instead of just terminal")
    p.add_argument("--no-scroll-dedup",     action="store_true",
                   help="skip scroll/2D deduplication before the jigsaw")
    p.add_argument("--assembler",           action="store_true",
                   help="generate assembler.html: interactively choose tiles to the right/below/above")
    p.add_argument("--transcribe",          nargs="+", metavar="IMAGE",
                   help="standalone mode: OCR transcript of the given image(s) "
                        "(e.g. your own manual montage) to --out/transcript.txt")
    args = p.parse_args()

    if not args.no_banner:
        print_banner()

    # ── standalone mode: transcribe existing image(s) ──
    if args.transcribe:
        out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
        specs = [(Image.open(p_).convert("RGB"), 6, Path(p_).name)
                 for p_ in args.transcribe]
        tp = out_dir / "transcript.txt"
        res = build_transcript(specs, str(tp))
        if res is None:
            print("OCR not available: the 'tesseract' binary was not found.\n"
                  "Install it with:  sudo apt-get install tesseract-ocr   "
                  "(and: pip install pytesseract)")
        else:
            print(f"Transcript: {tp}  ({res['flagged']} flagged / "
                  f"{res['total']} lines)")
            ind = res["indicators"]
            print("  emails :", ind["emails"] or "(none)")
            print("  hosts  :", ind["hosts"] or "(none)")
            print("  cmds   :", ind["commands"] or "(none)")
        return

    out_dir   = Path(args.out)
    tiles_dir = out_dir / "tiles"
    recon_dir = out_dir / "reconstructed"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/5] Extracting tiles from {args.bin_file}...")
    bmp_paths = extract_tiles_bmc(args.bin_file, str(tiles_dir))
    print(f"      {len(bmp_paths)} tiles extracted")

    print("\n[2/5] Loading and deduplicating tiles...")
    tiles, tile_sizes = load_tiles(bmp_paths)
    names = list(tiles.keys())

    print("\n[3/5] Building edge-match graph (Phase 1)...")
    h_nbr, v_nbr = build_neighbor_graph(
        tiles, tile_sizes,
        thr_normal=args.threshold,
        thr_terminal=args.threshold_terminal)

    print("\n[4/5] Building and validating chains (Phase 2)...")
    chains   = build_chains_mutual(h_nbr, names)
    n_before = len(chains)
    print(f"      After mutual-best: {n_before} chains, "
          f"top-5: {[len(c) for c in chains[:5]]}")

    chains  = validate_and_repair_chains(
        chains, tiles, h_nbr, corr_threshold=args.corr_threshold)
    n_after = len(chains)
    print(f"      After correlation validation: {n_after} chains "
          f"({n_after-n_before:+d})")

    groups = build_groups(chains, v_nbr, tile_sizes)
    print(f"      {len(groups)} groups, "
          f"top-5: {[sum(len(chains[ci]) for ci in g) for g in groups[:5]]}")

    print(f"\n[5/5] Exporting top {args.top} groups...")
    entries = []
    for gi, group in enumerate(groups[:args.top]):
        n_tiles = sum(len(chains[ci]) for ci in group)
        img     = render_group(group, chains, tiles, tile_sizes,
                               align=not args.no_align,
                               accept=args.align_accept)
        if img.width < 64 or img.height < 32: continue

        png_path = recon_dir / f"group_{gi:03d}_{n_tiles}t_{img.width}x{img.height}.png"
        img.save(str(png_path), "PNG")

        scale  = min(1.0, args.preview_width / img.width)
        pw, ph = int(img.width * scale), int(img.height * scale)
        buf    = io.BytesIO()
        img.resize((pw, ph), Image.LANCZOS).convert("RGB").save(buf, "JPEG", quality=88)
        b64    = base64.b64encode(buf.getvalue()).decode()

        entries.append({"id": gi, "tiles": n_tiles, "w": img.width, "h": img.height,
                         "pw": pw, "ph": ph, "rows": len(group), "b64": b64, "term": 0})
        print(f"      group_{gi:03d}: {n_tiles}t {img.width}x{img.height}px")

    # ── Terminal-text pass ──
    if not args.no_terminal:
        print("\n[+] Terminal-text reconstruction (console/PowerShell)...")
        term_dir = out_dir / "terminal_strips"
        term_dir.mkdir(parents=True, exist_ok=True)
        strips = reconstruct_terminal_strips(tiles, tile_sizes, min_len=args.term_min)
        strips = [s for s in strips if len(s) >= args.term_min]
        print(f"      {len(strips)} text-line strips "
              f"(longest: {[len(s) for s in strips[:8]]} tiles)")

        # write individual strips (reference)
        for si, strip in enumerate(strips[:args.term_top]):
            render_strip(strip, tiles, tile_sizes).save(
                str(term_dir / f"term_{si:03d}_{len(strip)}t.png"), "PNG")

        # stack vertically into multi-line console blocks
        blocks = stack_terminal_blocks(strips, tiles, tile_sizes)
        print(f"      {len(blocks)} multi-line blocks "
              f"(tallest: {[h for _, h in blocks[:8]]} rows)")
        block_dir = out_dir / "terminal_blocks"
        block_dir.mkdir(parents=True, exist_ok=True)

        # in the viewer we show the blocks (and individual strips that did not stack)
        shown = blocks[:args.term_top]
        for bi, (img, nrows) in enumerate(shown):
            img.save(str(block_dir / f"block_{bi:03d}_{nrows}r_{img.width}x{img.height}.png"), "PNG")
            scale = min(1.0, args.preview_width / img.width)
            pw, ph = max(1, int(img.width * scale)), max(1, int(img.height * scale))
            buf = io.BytesIO()
            img.resize((pw, ph), Image.LANCZOS).convert("RGB").save(buf, "JPEG", quality=92)
            b64 = base64.b64encode(buf.getvalue()).decode()
            entries.append({"id": 10000 + bi, "tiles": nrows,
                            "w": img.width, "h": img.height, "pw": pw, "ph": ph,
                            "rows": nrows, "b64": b64, "term": 1})

        # ── OCR transcript of the terminal reconstruction ──
        if not args.no_ocr:
            specs = [(render_strip(s, tiles, tile_sizes), 7, "strip")
                     for s in strips[:args.term_top]]
            specs += [(img, 6, "block") for img, _ in shown]
            tp = out_dir / "transcript.txt"
            res = build_transcript(specs, str(tp))
            if res is None:
                print("      (OCR skipped: the 'tesseract' binary was not found.\n"
                      "       Install it with:  sudo apt-get install tesseract-ocr )")
            else:
                ind = res["indicators"]
                print(f"      OCR transcript: {tp.name} "
                      f"({res['flagged']} flagged / {res['total']} lines)")
                print(f"        indicators - emails:{len(ind['emails'])} "
                      f"hosts:{len(ind['hosts'])} cmds:{len(ind['commands'])}")

    # ── 2D best-buddy jigsaw assembly ──
    if args.jigsaw or args.jigsaw_all:
        print("\n[+] 2D jigsaw assembly (best-buddy puzzle placement)...")
        if args.jigsaw_all:
            subset = [n for n in tiles if tile_sizes[n] == (64, 64)]
        else:
            subset = [n for n in tiles
                      if tile_sizes[n] == (64, 64)
                      and 4 < _term_feats(tiles[n])["ink"] < 140
                      and _term_feats(tiles[n])["bg"].mean() < 75
                      and _term_feats(tiles[n])["sharp"] > 0.018]
        if not args.no_scroll_dedup and len(subset) <= 2500:
            before = len(subset)
            subset = scroll_dedup(subset, tiles, tile_sizes)
            print(f"      scroll/2D dedup: {before} -> {len(subset)} representatives")
        elif not args.no_scroll_dedup:
            print(f"      (scroll dedup skipped: {len(subset)} tiles > 2500)")
        jigs = jigsaw_blocks(subset, tiles, tile_sizes)
        print(f"      {len(jigs)} 2D blocks (largest: {[n for _, n in jigs[:8]]} tiles)")
        jdir = out_dir / "jigsaw_blocks"
        jdir.mkdir(parents=True, exist_ok=True)
        for ji, (img, ntiles) in enumerate(jigs[:args.term_top]):
            img.save(str(jdir / f"jigsaw_{ji:03d}_{ntiles}t_{img.width}x{img.height}.png"), "PNG")
            scale = min(1.0, args.preview_width / img.width)
            pw, ph = max(1, int(img.width * scale)), max(1, int(img.height * scale))
            buf = io.BytesIO()
            img.resize((pw, ph), Image.LANCZOS).convert("RGB").save(buf, "JPEG", quality=92)
            b64 = base64.b64encode(buf.getvalue()).decode()
            entries.append({"id": 20000 + ji, "tiles": ntiles,
                            "w": img.width, "h": img.height, "pw": pw, "ph": ph,
                            "rows": 0, "b64": b64, "term": 1})

    # ── Interactive assembler (hand-guided reconstruction) ──
    if args.assembler:
        print("\n[+] Generating interactive assembler (assembler.html)...")
        ap = out_dir / "assembler.html"
        ntiles = build_assembler(tiles, tile_sizes, str(ap),
                                 content=("all" if args.jigsaw_all else "terminal"),
                                 scroll=not args.no_scroll_dedup)
        print(f"      assembler.html: {ntiles} tiles  "
              f"(open in a browser, choose a start tile and click the + cells)")

    if not args.no_viewer and entries:
        viewer = out_dir / "viewer.html"
        build_viewer(entries, str(viewer))
        print(f"\nHTML viewer: {viewer}")

    print(f"\nDone! Output in: {out_dir}/")




# ── Interactive assembler ────────────────────────────────────────────────────
def build_assembler(tiles, sizes, out_path, content="terminal", k=12, scroll=True):
    if content=="terminal":
        subset=[n for n in tiles if sizes[n]==(64,64)
                and 4<_term_feats(tiles[n])['ink']<140
                and _term_feats(tiles[n])['bg'].mean()<75
                and _term_feats(tiles[n])['sharp']>0.018]
    else:
        subset=[n for n in tiles if sizes[n]==(64,64)]
    if scroll and len(subset)<=2500:
        subset=scroll_dedup(subset,tiles,sizes)
    CAND,names=_jigsaw_candidates(subset,tiles,sizes,k=k,h_gate=0.65,v_gate=0.25)
    N=len(names)
    # base64 thumbnails
    b64=[]
    for n in names:
        buf=io.BytesIO(); Image.fromarray(tiles[n],"RGB").save(buf,"PNG"); b64.append(base64.b64encode(buf.getvalue()).decode())
    def arr(d):
        out=[[] for _ in range(N)]
        for i,lst in CAND[d].items():
            out[i]=[[int(j),round(float(s),1)] for s,j in lst]
        return out
    data=dict(b64=b64, R=arr("R"), L=arr("L"), U=arr("U"), D=arr("D"))
    html=_ASSEMBLER_HTML.replace("/*__DATA__*/", json.dumps(data))
    open(out_path,"w",encoding="utf-8").write(html)
    return N

_ASSEMBLER_HTML=r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>RDP Terminal Assembler</title>
<style>
:root{--bg:#0e0e12;--panel:#1a1a22;--line:#2a2a36;--txt:#d8d8e0;--accent:#ffd24a;--accent2:#4aa3ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:13px system-ui,Segoe UI,Arial}
header{padding:8px 12px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{background:var(--panel);color:var(--txt);border:1px solid var(--line);padding:5px 10px;border-radius:6px;cursor:pointer}
button:hover{border-color:var(--accent2)}
#main{display:flex;height:calc(100vh - 46px)}
#stage{flex:1;overflow:auto;position:relative;background:
  linear-gradient(90deg,#15151c 1px,transparent 1px) 0 0/64px 64px,
  linear-gradient(#15151c 1px,transparent 1px) 0 0/64px 64px,#0a0a0e}
#canvas{position:relative;transform-origin:0 0}
.cell{position:absolute;width:64px;height:64px}
.cell img{width:64px;height:64px;display:block;image-rendering:pixelated}
.cell.sel{outline:2px solid var(--accent);outline-offset:-2px;z-index:5}
.plus{position:absolute;width:64px;height:64px;border:1px dashed #3a3a4a;color:#5a5a6a;
  display:flex;align-items:center;justify-content:center;font-size:22px;cursor:pointer;background:rgba(74,163,255,.04)}
.plus:hover{border-color:var(--accent2);color:var(--accent2);background:rgba(74,163,255,.12)}
#side{width:300px;border-left:1px solid var(--line);overflow:auto;padding:10px;background:var(--panel)}
#side h3{margin:6px 0;font-size:12px;color:#9a9aa8;text-transform:uppercase;letter-spacing:.5px}
.cands{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
.cands .c{border:1px solid var(--line);border-radius:4px;cursor:pointer;position:relative}
.cands .c:hover{border-color:var(--accent)}
.cands .c img{width:96px;height:96px;display:block;image-rendering:pixelated}
.cands .c span{position:absolute;right:1px;bottom:1px;background:#000a;padding:0 3px;border-radius:3px;font-size:10px}
#palette{position:fixed;inset:0;background:#000c;display:none;align-items:center;justify-content:center;z-index:50}
#palbox{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;width:80vw;height:80vh;display:flex;flex-direction:column}
#palgrid{display:flex;flex-wrap:wrap;gap:3px;overflow:auto;align-content:flex-start}
#palgrid img{width:64px;height:64px;image-rendering:pixelated;border:1px solid var(--line);cursor:pointer}
#palgrid img:hover{border-color:var(--accent)}
.hint{color:#9a9aa8;font-size:12px;max-width:520px}
small{color:#7a7a88}
</style></head><body>
<header>
  <b style="color:var(--accent)">Terminal Assembler</b>
  <button onclick="startPick()">+ Start tile</button>
  <button onclick="delSel()">Delete selection</button>
  <button onclick="clearAll()">Clear</button>
  <span style="flex:1"></span>
  <label>zoom <input type="range" min="0.5" max="2" step="0.1" value="1" oninput="setZoom(this.value)"></label>
  <button onclick="exportPNG()">Export PNG</button>
</header>
<div id="main">
  <div id="stage"><div id="canvas"></div></div>
  <div id="side">
    <div class="hint">Click a <b>+</b> cell next to a placed tile. You then get the best-matching candidates; pick the one that continues the text logically. Click a placed tile to select it (and optionally delete it).</div>
    <div id="panel"></div>
  </div>
</div>
<div id="palette"><div id="palbox"><h3>Choose a start tile</h3><div id="palgrid"></div>
  <div style="text-align:right;margin-top:6px"><button onclick="closePal()">Close</button></div></div></div>
<script>
const DATA=/*__DATA__*/;
const T=DATA.b64, R=DATA.R, L=DATA.L, U=DATA.U, D=DATA.D;
const placed=new Map();      // "x,y" -> tile index
let sel=null, zoom=1, target=null;
const key=(x,y)=>x+","+y;
function setZoom(z){zoom=+z;document.getElementById('canvas').style.transform=`scale(${zoom})`;}
function neighborsCands(x,y){
  // which list of the neighbor points toward (x,y):
  const map=[[-1,0,R],[1,0,L],[0,-1,D],[0,1,U]];
  const sc={},pr={}; let nb=0;
  for(const [dx,dy,LST] of map){
    const k=key(x+dx,y+dy);
    if(placed.has(k)){nb++;const ni=placed.get(k);for(const [j,s] of (LST[ni]||[])){sc[j]=(sc[j]||0)+s;pr[j]=(pr[j]||0)+1;}}
  }
  let arr=Object.keys(sc).map(j=>({j:+j,s:sc[j],p:pr[j]}));
  arr=arr.filter(o=>![...placed.values()].includes(o.j)); // not already placed
  arr.sort((a,b)=>(b.p-a.p)||(a.s-b.s));
  return arr.slice(0,12);
}
function render(){
  const cv=document.getElementById('canvas');cv.innerHTML='';
  if(placed.size===0){cv.style.width=cv.style.height='64px';return;}
  let xs=[...placed.keys()].map(k=>+k.split(',')[0]);
  let ys=[...placed.keys()].map(k=>+k.split(',')[1]);
  const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
  const pad=1;
  const W=(maxx-minx+1+2*pad)*64,H=(maxy-miny+1+2*pad)*64;
  cv.style.width=W+'px';cv.style.height=H+'px';
  const ox=(minx-pad),oy=(miny-pad);
  // plus cells (empty neighbors of placed tiles)
  const empties=new Set();
  for(const k of placed.keys()){const [x,y]=k.split(',').map(Number);
    for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]){const nk=key(x+dx,y+dy);if(!placed.has(nk))empties.add(nk);}}
  for(const k of empties){const [x,y]=k.split(',').map(Number);
    const d=document.createElement('div');d.className='plus';d.textContent='+';
    d.style.left=((x-ox)*64)+'px';d.style.top=((y-oy)*64)+'px';
    d.onclick=()=>openTarget(x,y);cv.appendChild(d);}
  for(const [k,ti] of placed){const [x,y]=k.split(',').map(Number);
    const d=document.createElement('div');d.className='cell'+(sel===k?' sel':'');
    d.style.left=((x-ox)*64)+'px';d.style.top=((y-oy)*64)+'px';
    const im=document.createElement('img');im.src='data:image/png;base64,'+T[ti];d.appendChild(im);
    d.onclick=()=>{sel=k;render();showPanelForSel();};cv.appendChild(d);}
}
function openTarget(x,y){
  target=[x,y];sel=null;
  const cands=neighborsCands(x,y);
  const p=document.getElementById('panel');
  p.innerHTML=`<h3>Candidates for position (${x},${y})</h3>`+
    `<small>${cands.length} opties — klik de juiste</small><div class="cands" id="cl"></div>`;
  const cl=document.getElementById('cl');
  if(cands.length===0){cl.innerHTML='<small>No candidates (no matching edge). Try another cell or place manually via Start tile.</small>';}
  cands.forEach(o=>{const c=document.createElement('div');c.className='c';
    c.innerHTML=`<img src="data:image/png;base64,${T[o.j]}"><span>${o.p}×·${o.s.toFixed(0)}</span>`;
    c.onclick=()=>{placed.set(key(x,y),o.j);target=null;render();};cl.appendChild(c);});
  render();
}
function showPanelForSel(){
  const p=document.getElementById('panel');
  if(!sel){p.innerHTML='';return;}
  const [x,y]=sel.split(',').map(Number);
  p.innerHTML=`<h3>Selected tile (${x},${y})</h3>`+
    `<div style="display:flex;gap:6px;flex-wrap:wrap">
      <button onclick="suggestDir(${x},${y},1,0,'right')">→ right</button>
      <button onclick="suggestDir(${x},${y},-1,0,'left')">← left</button>
      <button onclick="suggestDir(${x},${y},0,-1,'up')">↑ up</button>
      <button onclick="suggestDir(${x},${y},0,1,'down')">↓ down</button>
      <button onclick="delSel()">🗑 delete</button>
     </div><div id="dirpanel"></div>`;
}
function suggestDir(x,y,dx,dy,label){
  const tx=x+dx,ty=y+dy;
  if(placed.has(key(tx,ty))){sel=key(tx,ty);render();showPanelForSel();return;}
  openTarget(tx,ty);
}
function startPick(){const g=document.getElementById('palgrid');g.innerHTML='';
  T.forEach((b,i)=>{const im=document.createElement('img');im.src='data:image/png;base64,'+b;
    im.title=i;im.onclick=()=>{const x=placed.size?9999:0;
      // place at (0,0) if empty, otherwise next to selection? Here: empty canvas -> (0,0)
      if(placed.size===0){placed.set('0,0',i);}else{
        // place loosely to the right of the right-most tile
        let xs=[...placed.keys()].map(k=>+k.split(',')[0]);let mx=Math.max(...xs);placed.set(key(mx+2,0),i);}
      closePal();render();};g.appendChild(im);});
  document.getElementById('palette').style.display='flex';}
function closePal(){document.getElementById('palette').style.display='none';}
function delSel(){if(sel){placed.delete(sel);sel=null;document.getElementById('panel').innerHTML='';render();}}
function clearAll(){placed.clear();sel=null;document.getElementById('panel').innerHTML='';render();}
function exportPNG(){
  if(placed.size===0)return;
  let xs=[...placed.keys()].map(k=>+k.split(',')[0]);let ys=[...placed.keys()].map(k=>+k.split(',')[1]);
  const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
  const W=(maxx-minx+1)*64,H=(maxy-miny+1)*64;
  const cv=document.createElement('canvas');cv.width=W;cv.height=H;const ctx=cv.getContext('2d');
  ctx.fillStyle='#000';ctx.fillRect(0,0,W,H);let pending=placed.size;
  for(const [k,ti] of placed){const [x,y]=k.split(',').map(Number);const im=new Image();
    im.onload=()=>{ctx.drawImage(im,(x-minx)*64,(y-miny)*64);if(--pending===0){
      const a=document.createElement('a');a.download='terminal_reconstruction.png';a.href=cv.toDataURL('image/png');a.click();}};
    im.src='data:image/png;base64,'+T[ti];}
}
render();
</script></body></html>"""


if __name__ == "__main__":
    main()