"""
Hex-retina sampler demo (single image)

What it does
- Loads an image from disk
- Builds a pointy-top hex grid over the entire image
- Computes per-hex aggregated features (avg RGB/HSV, edge energy)
- Applies an attention/focus profile (fovea + mid + periphery weighting)
- Visualizes:
  1) Original image
  2) Hex overlay
  3) "What the hexes take in" as a hex-colored mosaic (avg color)
  4) Focus/weight heatmap
  5) Edge-energy heatmap (optional)

How to use
- Put an image file path into IMAGE_PATH below and run.
- Then tweak the CONFIG dict (hex size, fovea radius, gaze point, etc).
- Optional: enable interactive sliders (Matplotlib) by setting USE_WIDGETS=True.

Dependencies: numpy, pillow, matplotlib
"""

import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Optional interactive UI (matplotlib widgets)
from matplotlib.widgets import Slider, Button, CheckButtons


# -----------------------------
# Configuration (edit these)
# -----------------------------
IMAGE_PATH = "temp_dir/IMG_0041.JPG"  # <-- set this

CONFIG = {
    # Hex geometry
    "hex_radius_px": 18,        # distance from hex center to vertex (controls hex size)
    "pointy_top": True,         # True: pointy-top, False: flat-top

    # Focus profile (weights)
    "fovea_radius_hex": 2.5,    # in "hex steps" (approx; computed from pixel distance / hex spacing)
    "mid_radius_hex": 6.0,
    "outer_radius_hex": 12.0,   # beyond this, weight ~0 or minimal
    "outer_floor": 0.05,        # minimum weight for far periphery
    "alpha": 0.35,              # radial decay factor (bigger = faster dropoff
    "focus_stretch_x": 1.0,     # >1 widens focus horizontally
    "focus_stretch_y": 1.0,     # >1 widens focus vertically

    # Gaze center (in image pixel coordinates). If None, use image center.
    "gaze_x": None,
    "gaze_y": None,

    # Feature computation
    "compute_hsv": True,
    "compute_edges": True,
    "send_rgb_uint8": True,     # estimate packet sizes assuming RGB sent as 3 bytes
    "include_hsv": True,        # estimate: include avg_hsv
    "include_edges": True,      # estimate: include edge_energy
    "include_weight": True,     # estimate: include focus weight
    "include_motion": False,    # estimate: include motion magnitude (placeholder)
    "include_patch": False,     # estimate: include raw pixel patches for selected hexes
    "patch_size": 32,           # patch is patch_size x patch_size RGB
    "patch_max_hexes": 24,      # maximum hexes per frame with patches

    # Visualization
    "overlay_linewidth": 0.8,
    "show_hex_ids": False,
    "hex_id_step": 8,           # label every Nth hex if show_hex_ids
    "show_interest": True,
    "interest_top_k": 12,
    "interest_mode": "periphery_edges",  # options: periphery_edges, center_edges, edges_only
    "show_recommendation": True,
}

USE_WIDGETS = True  # set False if you just want static plots


# -----------------------------
# Utilities
# -----------------------------
def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)

def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    # rgb: (...,3) uint8 -> hsv (...,3) float in [0,1]
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    mx = np.max(rgb_f, axis=-1)
    mn = np.min(rgb_f, axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    s = np.zeros_like(mx)
    v = mx

    mask = diff > 1e-6
    s[mask] = diff[mask] / (mx[mask] + 1e-12)

    # Hue calculation
    r_eq = (mx == r) & mask
    g_eq = (mx == g) & mask
    b_eq = (mx == b) & mask

    h[r_eq] = (g[r_eq] - b[r_eq]) / (diff[r_eq] + 1e-12)
    h[g_eq] = 2.0 + (b[g_eq] - r[g_eq]) / (diff[g_eq] + 1e-12)
    h[b_eq] = 4.0 + (r[b_eq] - g[b_eq]) / (diff[b_eq] + 1e-12)
    h = (h / 6.0) % 1.0

    return np.stack([h, s, v], axis=-1)

def simple_edge_energy(gray: np.ndarray) -> np.ndarray:
    """
    Cheap edge magnitude: |dx| + |dy| using finite differences.
    gray: HxW float in [0,1]
    """
    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])

    # pad back to HxW
    dx = np.pad(dx, ((0, 0), (0, 1)), mode="edge")
    dy = np.pad(dy, ((0, 1), (0, 0)), mode="edge")
    return dx + dy

def polygon_vertices_pointy(cx: float, cy: float, r: float) -> np.ndarray:
    # pointy-top: first vertex at angle 30°
    angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330], dtype=np.float32))
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)
    return np.stack([xs, ys], axis=1)

def polygon_vertices_flat(cx: float, cy: float, r: float) -> np.ndarray:
    # flat-top: first vertex at angle 0°
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300], dtype=np.float32))
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)
    return np.stack([xs, ys], axis=1)

def point_in_poly_mask(H: int, W: int, poly: np.ndarray) -> np.ndarray:
    """
    Rasterize polygon (6-vertex hex) into a boolean mask using a vectorized ray casting test.
    poly: (6,2) vertices
    Returns: mask HxW boolean
    Note: for demo use; not optimized.
    """
    xv = poly[:, 0]
    yv = poly[:, 1]
    x = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    y = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)

    inside = np.zeros((H, W), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = xv[i], yv[i]
        xj, yj = xv[j], yv[j]
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)
        inside ^= intersect
        j = i
    return inside


# -----------------------------
# Hex grid builder
# -----------------------------
@dataclass
class HexCell:
    idx: int
    center: Tuple[float, float]
    poly: np.ndarray  # (6,2) vertices
    # features
    avg_rgb: np.ndarray  # (3,) float in [0,1]
    avg_hsv: np.ndarray  # (3,) float in [0,1]
    edge_energy: float   # scalar
    # focus
    weight: float

def build_hex_centers(W: int, H: int, r: float, pointy_top: bool) -> List[Tuple[float, float]]:
    """
    Build hex centers that cover the image. Uses standard hex spacing.
    """
    centers = []

    if pointy_top:
        # pointy-top:
        # horizontal spacing = sqrt(3)*r
        # vertical spacing = 1.5*r
        dx = math.sqrt(3) * r
        dy = 1.5 * r
        # offset columns every other row
        row = 0
        y = r
        while y <= H + r:
            x_offset = (dx / 2.0) if (row % 2 == 1) else 0.0
            x = r + x_offset
            while x <= W + r:
                centers.append((x, y))
                x += dx
            y += dy
            row += 1
    else:
        # flat-top:
        # horizontal spacing = 1.5*r
        # vertical spacing = sqrt(3)*r
        dx = 1.5 * r
        dy = math.sqrt(3) * r
        col = 0
        x = r
        while x <= W + r:
            y_offset = (dy / 2.0) if (col % 2 == 1) else 0.0
            y = r + y_offset
            while y <= H + r:
                centers.append((x, y))
                y += dy
            x += dx
            col += 1

    return centers

def focus_weight_for_center(
    cx: float, cy: float,
    gx: float, gy: float,
    r: float, pointy_top: bool,
    cfg: Dict[str, Any]
) -> float:
    """
    Computes a radial attention weight based on approximate "hex steps".
    We convert pixel distance to an approximate hex distance by dividing by center spacing.
    """
    if pointy_top:
        spacing = math.sqrt(3) * r  # roughly one hex step in x
    else:
        spacing = math.sqrt(3) * r  # roughly one hex step in y (similar magnitude)
    # Anisotropic distance
    sx = float(cfg.get("focus_stretch_x", 1.0))
    sy = float(cfg.get("focus_stretch_y", 1.0))
    # Prevent divide-by-zero and keep behavior stable
    sx = sx if sx > 1e-6 else 1.0
    sy = sy if sy > 1e-6 else 1.0
    dx = (cx - gx) / sx
    dy = (cy - gy) / sy
    dist_px = math.hypot(dx, dy)
    d_hex = dist_px / (spacing + 1e-9)

    fovea = cfg["fovea_radius_hex"]
    mid = cfg["mid_radius_hex"]
    outer = cfg["outer_radius_hex"]

    # Piecewise: high in fovea, medium in mid, decays to outer_floor
    if d_hex <= fovea:
        base = 1.0
    elif d_hex <= mid:
        # smoothly drop from 1.0 to ~0.5 across [fovea, mid]
        t = (d_hex - fovea) / max(1e-6, (mid - fovea))
        base = 1.0 - 0.5 * t
    else:
        # exponential decay beyond mid
        base = 0.5 * math.exp(-cfg["alpha"] * (d_hex - mid))

    # clamp and floor out to outer
    if d_hex > outer:
        base = cfg["outer_floor"]

    return float(max(cfg["outer_floor"], min(1.0, base)))


######################
# Backend payload estimator and interest/recommendation helpers
######################

def estimate_backend_payload_bytes(cells: List[HexCell], cfg: Dict[str, Any]) -> Dict[str, int]:
    """
    Returns dict with:
        - per_hex_bytes
        - hex_feature_bytes_total
        - patch_bytes_total
        - total_bytes
    """
    n_hexes = len(cells)
    # Per-hex feature size
    per_hex = 0
    if cfg.get("send_rgb_uint8", True):
        per_hex += 3  # 3 bytes for uint8 RGB
    else:
        per_hex += 12  # 3 float32
    if cfg.get("include_hsv", True):
        per_hex += 12  # 3 float32
    if cfg.get("include_edges", True):
        per_hex += 4   # float32
    if cfg.get("include_weight", True):
        per_hex += 4   # float32
    if cfg.get("include_motion", False):
        per_hex += 4   # float32
    per_hex += 4  # id
    hex_feature_bytes_total = per_hex * n_hexes
    patch_bytes_total = 0
    if cfg.get("include_patch", False):
        patch_size = int(cfg.get("patch_size", 32))
        patch_max_hexes = int(cfg.get("patch_max_hexes", 24))
        patch_bytes_total = patch_max_hexes * patch_size * patch_size * 3
    total_bytes = hex_feature_bytes_total + patch_bytes_total
    return {
        "per_hex_bytes": per_hex,
        "hex_feature_bytes_total": hex_feature_bytes_total,
        "patch_bytes_total": patch_bytes_total,
        "total_bytes": total_bytes
    }

def compute_interest_scores(cells: List[HexCell], cfg: Dict[str, Any]) -> np.ndarray:
    """
    Computes interest scores for each cell using heuristics.
    Returns: np.ndarray of shape (num_cells,)
    """
    e_vals = np.array([c.edge_energy for c in cells], dtype=np.float32)
    # Normalize edge energy to 0-1
    e_min, e_max = float(e_vals.min()), float(e_vals.max())
    denom = (e_max - e_min) if (e_max > e_min) else 1.0
    norm_edges = (e_vals - e_min) / denom
    weights = np.array([c.weight for c in cells], dtype=np.float32)
    mode = cfg.get("interest_mode", "periphery_edges")
    if mode == "periphery_edges":
        interest = norm_edges * (1.0 - weights)
    elif mode == "center_edges":
        interest = norm_edges * weights
    elif mode == "edges_only":
        interest = norm_edges
    else:
        interest = norm_edges
    # Normalize again to [0,1] for safety
    interest = interest - interest.min()
    if interest.max() > 1e-8:
        interest = interest / interest.max()
    return interest

def recommend_next_gaze(
    cells: List[HexCell],
    interest: np.ndarray,
    cfg: Dict[str, Any]
) -> Tuple[float, float, Dict[str, float]]:
    """
    Selects top-k interest cells, computes recommended gaze point (weighted avg of centers),
    returns (rec_x, rec_y, profile_dict)
    """
    k = int(cfg.get("interest_top_k", 12))
    if len(interest) == 0:
        # fallback to image center
        W = cfg.get("_img_W", 512)
        H = cfg.get("_img_H", 512)
        return (W / 2.0, H / 2.0, {
            "suggested_fovea_radius_hex": cfg.get("fovea_radius_hex", 2.5),
            "suggested_mid_radius_hex": cfg.get("mid_radius_hex", 6.0),
            "suggested_include_patch": False
        })
    idxs = np.argpartition(-interest, min(k, len(interest)-1))[:k]
    idxs = idxs[np.argsort(-interest[idxs])]  # sort top-k by interest descending
    centers = np.array([cells[i].center for i in idxs], dtype=np.float32)
    scores = interest[idxs]
    if np.sum(scores) > 1e-8:
        weights = scores / np.sum(scores)
    else:
        weights = np.ones_like(scores) / len(scores)
    rec_xy = np.sum(centers * weights[:, None], axis=0)
    rec_x, rec_y = float(rec_xy[0]), float(rec_xy[1])
    max_interest = float(np.max(interest))
    # Compute "far" as distance from current gaze
    W = cfg.get("_img_W", 512)
    H = cfg.get("_img_H", 512)
    cur_gx = cfg.get("gaze_x", None)
    cur_gy = cfg.get("gaze_y", None)
    if cur_gx is None:
        cur_gx = W / 2.0
    if cur_gy is None:
        cur_gy = H / 2.0
    dist = math.hypot(rec_x - float(cur_gx), rec_y - float(cur_gy))
    far_thresh = 0.25 * min(W, H)
    is_high = max_interest > 0.65
    is_far = dist > far_thresh
    suggested_fovea = 3.0 if (is_high and is_far) else 2.0
    suggested_include_patch = is_high
    profile = {
        "suggested_fovea_radius_hex": suggested_fovea,
        "suggested_mid_radius_hex": cfg.get("mid_radius_hex", 6.0),
        "suggested_include_patch": suggested_include_patch
    }
    return (rec_x, rec_y, profile)

def compute_cells(img_rgb_u8: np.ndarray, cfg: Dict[str, Any]) -> List[HexCell]:
    H, W = img_rgb_u8.shape[:2]
    r = float(cfg["hex_radius_px"])
    pointy = bool(cfg["pointy_top"])

    gx = cfg["gaze_x"] if cfg["gaze_x"] is not None else (W / 2.0)
    gy = cfg["gaze_y"] if cfg["gaze_y"] is not None else (H / 2.0)

    img_rgb = img_rgb_u8.astype(np.float32) / 255.0
    img_hsv = rgb_to_hsv(img_rgb_u8) if cfg["compute_hsv"] else None

    gray = (0.2989 * img_rgb[..., 0] + 0.5870 * img_rgb[..., 1] + 0.1140 * img_rgb[..., 2]).astype(np.float32)
    edges = simple_edge_energy(gray) if cfg["compute_edges"] else None

    centers = build_hex_centers(W, H, r, pointy)
    cells: List[HexCell] = []

    for idx, (cx, cy) in enumerate(centers):
        poly = polygon_vertices_pointy(cx, cy, r) if pointy else polygon_vertices_flat(cx, cy, r)

        # quick reject: if polygon bbox doesn't intersect image, skip
        minx, miny = np.floor(poly.min(axis=0)).astype(int)
        maxx, maxy = np.ceil(poly.max(axis=0)).astype(int)
        if maxx < 0 or maxy < 0 or minx >= W or miny >= H:
            continue

        # clip bbox to image
        minx_c = max(0, minx)
        miny_c = max(0, miny)
        maxx_c = min(W - 1, maxx)
        maxy_c = min(H - 1, maxy)

        # rasterize mask within bbox
        bbox_W = maxx_c - minx_c + 1
        bbox_H = maxy_c - miny_c + 1

        poly_shifted = poly.copy()
        poly_shifted[:, 0] -= minx_c
        poly_shifted[:, 1] -= miny_c
        mask = point_in_poly_mask(bbox_H, bbox_W, poly_shifted)

        patch_rgb = img_rgb[miny_c:maxy_c + 1, minx_c:maxx_c + 1, :]
        if mask.sum() < 10:  # too small to be meaningful
            continue

        avg_rgb = patch_rgb[mask].mean(axis=0)

        if img_hsv is not None:
            patch_hsv = img_hsv[miny_c:maxy_c + 1, minx_c:maxx_c + 1, :]
            avg_hsv = patch_hsv[mask].mean(axis=0)
        else:
            avg_hsv = np.zeros(3, dtype=np.float32)

        if edges is not None:
            patch_e = edges[miny_c:maxy_c + 1, minx_c:maxx_c + 1]
            edge_energy = float(patch_e[mask].mean())
        else:
            edge_energy = 0.0

        w = focus_weight_for_center(cx, cy, gx, gy, r, pointy, cfg)

        cells.append(HexCell(
            idx=idx,
            center=(cx, cy),
            poly=poly,
            avg_rgb=avg_rgb,
            avg_hsv=avg_hsv,
            edge_energy=edge_energy,
            weight=w
        ))

    return cells

def render_hex_mosaic(H: int, W: int, cells: List[HexCell], mode: str) -> np.ndarray:
    """
    mode:
      - "avg_rgb": each hex filled with its avg color
      - "weight": grayscale from weight
      - "edges": grayscale from edge energy
      - "weighted_rgb": avg_rgb * weight + background*(1-weight)
    Returns an RGB float image HxWx3 in [0,1].
    """
    out = np.zeros((H, W, 3), dtype=np.float32)

    # Normalize edges for display if needed
    if mode == "edges":
        e_vals = np.array([c.edge_energy for c in cells], dtype=np.float32)
        e_min, e_max = float(e_vals.min()), float(e_vals.max())
        denom = (e_max - e_min) if (e_max > e_min) else 1.0

    for c in cells:
        poly = c.poly
        minx, miny = np.floor(poly.min(axis=0)).astype(int)
        maxx, maxy = np.ceil(poly.max(axis=0)).astype(int)

        if maxx < 0 or maxy < 0 or minx >= W or miny >= H:
            continue
        minx_c = max(0, minx)
        miny_c = max(0, miny)
        maxx_c = min(W - 1, maxx)
        maxy_c = min(H - 1, maxy)

        bbox_W = maxx_c - minx_c + 1
        bbox_H = maxy_c - miny_c + 1

        poly_shifted = poly.copy()
        poly_shifted[:, 0] -= minx_c
        poly_shifted[:, 1] -= miny_c
        mask = point_in_poly_mask(bbox_H, bbox_W, poly_shifted)

        if mode == "avg_rgb":
            color = c.avg_rgb
        elif mode == "weight":
            g = c.weight
            color = np.array([g, g, g], dtype=np.float32)
        elif mode == "edges":
            g = (c.edge_energy - e_min) / denom
            color = np.array([g, g, g], dtype=np.float32)
        elif mode == "weighted_rgb":
            # Put a dark neutral background under low weight
            bg = np.array([0.08, 0.08, 0.08], dtype=np.float32)
            color = c.avg_rgb * c.weight + bg * (1.0 - c.weight)
        else:
            raise ValueError(f"Unknown mode {mode}")

        patch = out[miny_c:maxy_c + 1, minx_c:maxx_c + 1, :]
        patch[mask] = color
        out[miny_c:maxy_c + 1, minx_c:maxx_c + 1, :] = patch

    return np.clip(out, 0.0, 1.0)

def plot_all(img_rgb_u8: np.ndarray, cells: List[HexCell], cfg: Dict[str, Any]) -> None:
    H, W = img_rgb_u8.shape[:2]
    img_rgb = img_rgb_u8.astype(np.float32) / 255.0

    mosaic = render_hex_mosaic(H, W, cells, mode="avg_rgb")
    weighted = render_hex_mosaic(H, W, cells, mode="weighted_rgb")
    weightmap = render_hex_mosaic(H, W, cells, mode="weight")
    edgemap = render_hex_mosaic(H, W, cells, mode="edges") if cfg["compute_edges"] else None

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, wspace=0.05, hspace=0.15)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_rgb)
    ax0.set_title("Original")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(img_rgb)
    for c in cells:
        poly = c.poly
        ax1.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]],
                 linewidth=cfg["overlay_linewidth"], color="white", alpha=0.7)
        if cfg["show_hex_ids"] and (c.idx % cfg["hex_id_step"] == 0):
            ax1.text(c.center[0], c.center[1], str(c.idx), color="yellow",
                     fontsize=7, ha="center", va="center")
    gx = cfg["gaze_x"] if cfg["gaze_x"] is not None else (W / 2.0)
    gy = cfg["gaze_y"] if cfg["gaze_y"] is not None else (H / 2.0)
    ax1.scatter([gx], [gy], s=50, c="red")
    ax1.set_title("Hex overlay (red = gaze center)")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(mosaic)
    ax2.set_title("Hex mosaic (avg color per hex)")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(weighted)
    ax3.set_title("What’s 'taken in' (avg color × focus weight)")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(weightmap, cmap="gray")
    ax4.set_title("Focus/weight heatmap")
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 2])
    if edgemap is not None:
        ax5.imshow(edgemap, cmap="gray")
        ax5.set_title("Edge energy heatmap (per hex)")
    else:
        ax5.text(0.5, 0.5, "Edges disabled", ha="center", va="center")
        ax5.set_title("Edge energy heatmap")
    ax5.axis("off")

    plt.show()


# -----------------------------
# Interactive UI (sliders)
# -----------------------------
def run_with_widgets(img_rgb_u8: np.ndarray, cfg: Dict[str, Any]) -> None:
    H, W = img_rgb_u8.shape[:2]

    # For recommendation logic, store image shape in cfg temporarily
    cfg["_img_W"] = W
    cfg["_img_H"] = H
    cells = compute_cells(img_rgb_u8, cfg)
    img_rgb = img_rgb_u8.astype(np.float32) / 255.0

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 3, wspace=0.05)

    axA = fig.add_subplot(gs[0, 0])
    axA.set_title("Original + overlay")
    axA.axis("off")

    axB = fig.add_subplot(gs[0, 1])
    axB.set_title("Taken in (avg×weight)")
    axB.axis("off")

    axC = fig.add_subplot(gs[0, 2])
    axC.set_title("Weight heatmap")
    axC.axis("off")

    # initial images
    axA.imshow(img_rgb)
    taken_im = axB.imshow(render_hex_mosaic(H, W, cells, "weighted_rgb"))
    weight_im = axC.imshow(render_hex_mosaic(H, W, cells, "weight"), cmap="gray")

    # --- Sliders ---
    axcolor = "lightgoldenrodyellow"
    ax_hex = plt.axes([0.12, 0.14, 0.25, 0.03], facecolor=axcolor)
    ax_fov = plt.axes([0.12, 0.10, 0.25, 0.03], facecolor=axcolor)
    ax_mid = plt.axes([0.45, 0.14, 0.25, 0.03], facecolor=axcolor)
    ax_out = plt.axes([0.45, 0.10, 0.25, 0.03], facecolor=axcolor)
    # New: focus stretch sliders
    ax_stx = plt.axes([0.12, 0.06, 0.25, 0.03], facecolor=axcolor)
    ax_sty = plt.axes([0.45, 0.06, 0.25, 0.03], facecolor=axcolor)

    s_hex = Slider(ax_hex, "hex_radius_px", 6, 60, valinit=cfg["hex_radius_px"], valstep=1)
    s_fov = Slider(ax_fov, "fovea_r_hex", 0.5, 8.0, valinit=cfg["fovea_radius_hex"], valstep=0.1)
    s_mid = Slider(ax_mid, "mid_r_hex", 1.0, 16.0, valinit=cfg["mid_radius_hex"], valstep=0.1)
    s_out = Slider(ax_out, "outer_r_hex", 2.0, 30.0, valinit=cfg["outer_radius_hex"], valstep=0.1)
    s_stx = Slider(ax_stx, "focus_stretch_x", 0.5, 3.0, valinit=cfg.get("focus_stretch_x", 1.0), valstep=0.05)
    s_sty = Slider(ax_sty, "focus_stretch_y", 0.5, 3.0, valinit=cfg.get("focus_stretch_y", 1.0), valstep=0.05)

    ax_btn = plt.axes([0.80, 0.08, 0.12, 0.07])
    btn = Button(ax_btn, "Center gaze", color=axcolor, hovercolor="0.975")

    # --- CheckButtons for toggles ---
    ax_checks = plt.axes([0.80, 0.18, 0.13, 0.17], facecolor='whitesmoke')
    check_labels = [
        "Show interest",
        "Show recommendation",
        "Include patches",
        "Include HSV",
        "Include edges"
    ]
    check_states = [
        cfg.get("show_interest", True),
        cfg.get("show_recommendation", True),
        cfg.get("include_patch", False),
        cfg.get("include_hsv", True),
        cfg.get("include_edges", True)
    ]
    checks = CheckButtons(ax_checks, check_labels, check_states)

    # --- Text block for stats and explanations ---
    fig_text = fig.text(0.12, 0.01, "", fontsize=10, va="bottom", ha="left", family="monospace")

    def redraw_overlay():
        """Safe redraw: clear the axis and draw everything fresh."""
        axA.clear()
        axA.imshow(img_rgb)
        axA.set_title("Original + overlay")
        axA.axis("off")

        # draw hex outlines
        for c in cells:
            poly = c.poly
            axA.plot(
                np.r_[poly[:, 0], poly[0, 0]],
                np.r_[poly[:, 1], poly[0, 1]],
                linewidth=cfg["overlay_linewidth"],
                color="white",
                alpha=0.6,
            )

        gx = cfg["gaze_x"] if cfg["gaze_x"] is not None else (W / 2.0)
        gy = cfg["gaze_y"] if cfg["gaze_y"] is not None else (H / 2.0)
        axA.scatter([gx], [gy], s=50, c="red")

        # Show interest markers
        if cfg.get("show_interest", True):
            interest = compute_interest_scores(cells, cfg)
            k = int(cfg.get("interest_top_k", 12))
            if len(interest) > 0:
                idxs = np.argpartition(-interest, min(k, len(interest)-1))[:k]
                for i in idxs:
                    c = cells[i]
                    axA.plot(c.center[0], c.center[1], 'o', color='yellow', markersize=7, markeredgewidth=0.8, markeredgecolor='black', alpha=0.7)
        # Show recommendation cross
        if cfg.get("show_recommendation", True):
            interest = compute_interest_scores(cells, cfg)
            rec_x, rec_y, profile = recommend_next_gaze(cells, interest, cfg)
            axA.plot([rec_x], [rec_y], marker='x', color='cyan', markersize=13, markeredgewidth=2.3)
            label = f"fovea_r={profile['suggested_fovea_radius_hex']:.1f}, patch={profile['suggested_include_patch']}"
            axA.text(rec_x+10, rec_y+10, label, color='cyan', fontsize=9, bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.2'))

    def update_fig_text():
        # Estimate bytes
        stats = estimate_backend_payload_bytes(cells, cfg)
        n_hexes = len(cells)
        per_hex = stats["per_hex_bytes"]
        hex_total = stats["hex_feature_bytes_total"]
        patch_total = stats["patch_bytes_total"]
        total = stats["total_bytes"]
        kib = total / 1024.0
        mib = total / (1024.0 * 1024.0)
        mode = "feature-only stream" if patch_total == 0 else "feature + patches"
        # Slider explanations
        expl = (
            "hex_radius_px: larger = fewer hexes\n"
            "fovea/mid/outer: radii of focus profile\n"
            "focus_stretch_x/y: makes focus horizontally/vertically wider\n"
        )
        stats_str = (
            f"#hexes: {n_hexes}\n"
            f"bytes/hex: {per_hex}\n"
            f"hex features: {hex_total} bytes\n"
            f"patches: {patch_total} bytes\n"
            f"total: {total} bytes ({kib:.1f} KiB, {mib:.3f} MiB)\n"
            f"mode: {mode}\n"
        )
        fig_text.set_text(expl + stats_str)

    def recompute(_=None):
        nonlocal cells
        cfg["hex_radius_px"] = float(s_hex.val)
        cfg["fovea_radius_hex"] = float(s_fov.val)
        cfg["mid_radius_hex"] = float(s_mid.val)
        cfg["outer_radius_hex"] = float(s_out.val)
        cfg["focus_stretch_x"] = float(s_stx.val)
        cfg["focus_stretch_y"] = float(s_sty.val)
        # CheckButtons toggles
        cfg["show_interest"] = checks.get_status()[0]
        cfg["show_recommendation"] = checks.get_status()[1]
        cfg["include_patch"] = checks.get_status()[2]
        cfg["include_hsv"] = checks.get_status()[3]
        cfg["include_edges"] = checks.get_status()[4]
        # For feature computation
        cfg["compute_hsv"] = cfg["include_hsv"]
        cfg["compute_edges"] = cfg["include_edges"]
        # Recompute cells and images
        cells = compute_cells(img_rgb_u8, cfg)
        taken_im.set_data(render_hex_mosaic(H, W, cells, "weighted_rgb"))
        weight_im.set_data(render_hex_mosaic(H, W, cells, "weight"))
        redraw_overlay()
        update_fig_text()
        fig.canvas.draw_idle()

    def center_gaze(_event):
        cfg["gaze_x"] = W / 2.0
        cfg["gaze_y"] = H / 2.0
        recompute()

    def checkbuttons_callback(label):
        recompute()

    s_hex.on_changed(recompute)
    s_fov.on_changed(recompute)
    s_mid.on_changed(recompute)
    s_out.on_changed(recompute)
    s_stx.on_changed(recompute)
    s_sty.on_changed(recompute)
    btn.on_clicked(center_gaze)
    checks.on_clicked(checkbuttons_callback)

    def onclick(event):
        if event.inaxes != axA:
            return
        if event.xdata is None or event.ydata is None:
            return
        cfg["gaze_x"] = float(event.xdata)
        cfg["gaze_y"] = float(event.ydata)
        recompute()

    redraw_overlay()
    update_fig_text()
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

# -----------------------------
# Main
# -----------------------------
def main():
    img = load_image_rgb(IMAGE_PATH)

    if USE_WIDGETS:
        run_with_widgets(img, CONFIG)
    else:
        cells = compute_cells(img, CONFIG)
        plot_all(img, cells, CONFIG)

if __name__ == "__main__":
    main()