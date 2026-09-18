"""Regenerates sharding-colored{3,4,5,6}.png for the X-is-rows convention.

The originals draw the device mesh with X along the columns and Y along the
rows. Under the new convention (X = row axis, Y = column axis, matching
sharding-colored1.png and JAX's Mesh semantics) the shard held by device
(r, c) is exactly the shard the old figures gave device (c, r). A block's
position inside its gray square encodes the shard's *global* position in the
array, which travels with the content — so transposing the device-square
contents of the original PNGs produces the corrected figures pixel-perfectly:

* colored3 / colored4 / the single-axis panels of colored5: swap the top-right
  and bottom-left square contents.
* colored5's multi-axis panels (I_xy and J_xy) and both colored6 panels are
  NOT transposed: flattened shardings like I_XY assign block i to device i
  with the first-named axis major (matching jax.P(('X', 'Y')) reshape order),
  which reads the same under either axis convention, so the original panels
  were already correct.

sharding-example.png instead has its green axis labels swapped: the horizontal
arrow was labeled I_X and the vertical J_Y; the ink is rotated 90 degrees and
moved to the other arrow. (sharding-colored1.png is generated from scratch by
sharding_figure.py instead.)

Run against pristine originals (e.g. from git), not already-converted files.
Pass step names to run a subset:
  uv run --with pillow --with numpy python _scripts/transpose_sharding_figures.py [3 4 5 example subscripts]
"""

import sys

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

IMG = 'assets/img'
GRAY = np.array([239, 236, 234])  # device-square fill


def cluster(v: np.ndarray, gap: int = 5) -> list[tuple[int, int]]:
  """Groups sorted pixel indices into (start, end) runs separated by > gap."""
  out, start, prev = [], v[0], v[0]
  for x in v[1:]:
    if x > prev + gap:
      out.append((start, prev))
      start = x
    prev = x
  out.append((start, prev))
  return out


def square_ranges(a: np.ndarray, min_run: int = 40) -> tuple[list, list]:
  """Returns (x_ranges, y_ranges) of the gray device squares in the image."""
  gray = (np.abs(a.astype(int) - GRAY).sum(axis=2) < 20)
  xs = cluster(np.where(gray.sum(axis=0) > min_run)[0])
  ys = cluster(np.where(gray.sum(axis=1) > min_run)[0])
  return xs, ys


def swap_regions(a: np.ndarray, xa: tuple, ya: tuple, xb: tuple,
                 yb: tuple, pad: int = 4) -> None:
  """Swaps two equal-sized rectangles (given as inclusive ranges), in place."""
  ih, iw = a.shape[:2]
  w = max(xa[1] - xa[0], xb[1] - xb[0]) + 1
  h = max(ya[1] - ya[0], yb[1] - yb[0]) + 1
  # Shrink the pad if a region touches the image border.
  pad = min(pad, xa[0], ya[0], xb[0], yb[0],
            iw - 1 - max(xa[0], xb[0]) - w, ih - 1 - max(ya[0], yb[0]) - h)
  pad = max(pad, 0)
  w, h = w + 2 * pad, h + 2 * pad
  ax, ay, bx, by = xa[0] - pad, ya[0] - pad, xb[0] - pad, yb[0] - pad
  tmp = a[ay:ay + h, ax:ax + w].copy()
  a[ay:ay + h, ax:ax + w] = a[by:by + h, bx:bx + w]
  a[by:by + h, bx:bx + w] = tmp


def transpose_panels(name: str, skip: frozenset = frozenset()) -> None:
  """Swaps top-right and bottom-left squares of every 2x2 panel in a figure.

  Args:
    name: Figure filename under assets/img.
    skip: (panel_row, panel_col) pairs to leave untouched — used for the
      multi-axis (flattened) sharding panels, which are convention-agnostic.
  """
  path = f'{IMG}/{name}'
  a = np.array(PIL.Image.open(path).convert('RGB'))
  xs, ys = square_ranges(a)
  assert len(xs) % 2 == 0 and len(ys) % 2 == 0, (name, xs, ys)
  for pr in range(len(ys) // 2):
    for pc in range(len(xs) // 2):
      if (pr, pc) in skip:
        continue
      r0, r1 = ys[2 * pr], ys[2 * pr + 1]
      c0, c1 = xs[2 * pc], xs[2 * pc + 1]
      swap_regions(a, c1, r0, c0, r1)  # top-right <-> bottom-left
  PIL.Image.fromarray(a).save(path)
  print(f'{name}: transposed panels (skipped {len(skip)})')


def swap_example_labels() -> None:
  """Swaps the green I_X / J_Y axis labels of sharding-example.png."""
  path = f'{IMG}/sharding-example.png'
  a = np.array(PIL.Image.open(path).convert('RGB'))
  ai = a.astype(int)
  green = ((ai[:, :, 1] > 90) & (ai[:, :, 1] - ai[:, :, 0] > 30)
           & (ai[:, :, 1] - ai[:, :, 2] > 30))
  green[:760] = False  # keep only the horizontal label; TPU 2's fill is green
  hy, hx = np.where(green)
  vx, vy = (914, 975), (460, 511)  # vertical J_Y label (measured; the green
  # mask can't isolate it from the TPU square by thresholding alone)
  pad = 2
  vlab = a[vy[0] - pad:vy[1] + 1 + pad, vx[0] - pad:vx[1] + 1 + pad].copy()
  hlab = a[hy.min() - pad:hy.max() + 1 + pad,
           hx.min() - pad:hx.max() + 1 + pad].copy()
  hc = ((hx.min() + hx.max()) / 2, (hy.min() + hy.max()) / 2)
  vc = ((vx[0] + vx[1]) / 2, (vy[0] + vy[1]) / 2)
  a[vy[0] - pad:vy[1] + 1 + pad, vx[0] - pad:vx[1] + 1 + pad] = 255
  a[hy.min() - pad:hy.max() + 1 + pad, hx.min() - pad:hx.max() + 1 + pad] = 255
  # The vertical label is upright text rotated 90 CCW, so rotate it CW to sit
  # under the horizontal arrow, and rotate the horizontal label CCW to stand
  # beside the vertical arrow. Paste each centered where the other one was.
  for ink, (cx, cy) in [(np.rot90(vlab, -1), hc), (np.rot90(hlab, 1), vc)]:
    h, w = ink.shape[:2]
    y0, x0 = round(cy - h / 2), round(cx - w / 2)
    a[y0:y0 + h, x0:x0 + w] = ink
  PIL.Image.fromarray(a).save(path)
  print('sharding-example.png: swapped axis labels')


# Panel titles as (text, is_subscript) runs, reading order. None = leave as is.
TITLES5 = [
    None, [('I', 0), ('x', 1), (', J', 0)], [('I', 0), (', J', 0), ('x', 1)],
    [('I', 0), ('y', 1), (', J', 0)], [('I', 0), ('xy', 1), (', J', 0)],
    [('I', 0), ('y', 1), (', J', 0), ('x', 1)],
    [('I', 0), (', J', 0), ('y', 1)],
    [('I', 0), ('x', 1), (', J', 0), ('y', 1)],
    [('I', 0), (', J', 0), ('xy', 1)],
]
TITLES6 = [[('I', 0), ('xy', 1), (', J', 0)], [('I', 0), ('yx', 1), (', J', 0)]]
SUB_SCALE = 0.62   # subscript size relative to the main font
SUB_DROP = 0.20    # subscript baseline drop, in cap heights


def pt_sans(size: int) -> PIL.ImageFont.FreeTypeFont:
  return PIL.ImageFont.truetype(
      '/System/Library/Fonts/Supplemental/PTSans.ttc', size=size, index=0)


def scrub_below_band(a: np.ndarray, xs: list, y1: int, tx0: int,
                     tx1: int) -> None:
  """Removes title-descender ink that spilled past the band into the squares.

  Titles sit above the left square of their panel; the panel's right square is
  pixel-identical (rounded corners included) and has nothing above it, so any
  pixel darker than its counterpart there is ink and gets replaced by it.
  """
  pc = next(i // 2 for i, (x0, x1) in enumerate(xs) if x0 <= tx0 <= x1)
  off = xs[2 * pc + 1][0] - xs[2 * pc][0]
  strip = a[y1:y1 + 40, tx0 - 4:tx1 + 5].astype(int)
  src = a[y1:y1 + 40, tx0 - 4 + off:tx1 + 5 + off].astype(int)
  ink = strip.sum(axis=2) < src.sum(axis=2) - 25
  a[y1:y1 + 40, tx0 - 4:tx1 + 5][ink] = a[y1:y1 + 40,
                                          tx0 - 4 + off:tx1 + 5 + off][ink]


def subscript_titles(name: str, titles: list, bands: int) -> None:
  """Redraws 'Ix, J'-style panel titles with the axis letters as subscripts.

  Args:
    name: Figure filename under assets/img.
    titles: Per-title run lists, reading order (row-major over title bands).
    bands: Number of title bands (rows of panels) in the figure.
  """
  path = f'{IMG}/{name}'
  img = PIL.Image.open(path).convert('RGB')
  a = np.array(img)
  dark = a.astype(int).sum(axis=2) < 250
  sq_xs, ys = square_ranges(a)
  for b in range(bands):
    band0, band1 = (0 if b == 0 else ys[2 * b - 1][1]), ys[2 * b][0]
    xs = np.where(dark[band0:band1].any(axis=0))[0]
    for tx0, tx1 in cluster(xs, gap=100):
      scrub_below_band(a, sq_xs, band1, tx0, tx1)
  img = PIL.Image.fromarray(a)
  draw = PIL.ImageDraw.Draw(img)
  it = iter(titles)
  for b in range(bands):
    y0, y1 = (0 if b == 0 else ys[2 * b - 1][1]), ys[2 * b][0]
    band = dark[y0:y1]
    xs = np.where(band.any(axis=0))[0]
    for tx0, tx1 in cluster(xs, gap=100):
      runs = next(it)
      if runs is None:
        continue
      ink = np.where(band[:, tx0:tx1 + 1].any(axis=1))[0]
      stem = np.where(band[:, tx0:tx0 + 20].any(axis=1))[0]  # leading 'I'
      cap = stem.max() - stem.min() + 1
      size = round(cap * 1000 / 700)  # PT Sans capHeight = 700/1000 em
      fonts = {0: pt_sans(size), 1: pt_sans(round(size * SUB_SCALE))}
      base = {0: y0 + stem.max() + 1, 1: y0 + stem.max() + 1 + SUB_DROP * cap}
      # If a subscript descender would hit the squares below, lift the title.
      bottom = max(draw.textbbox((0, base[s]), t, font=fonts[s], anchor='ls')[3]
                   for t, s in runs)
      lift = max(0, bottom - (y1 - 8))
      # Clamp to the title band so the erase can't nick the squares below.
      draw.rectangle((tx0 - 4, max(y0 + ink.min() - 4, 0), tx1 + 4,
                      min(y0 + ink.max() + 4, y1 - 1)), fill='white')
      x = tx0
      for text, s in runs:
        draw.text((x, base[s] - lift), text, font=fonts[s], fill='black',
                  anchor='ls')
        x += fonts[s].getlength(text)
  PIL.Image.fromarray(np.array(img)).save(path)
  print(f'{name}: subscripted titles')


if __name__ == '__main__':
  steps = sys.argv[1:] or ['3', '4', '5', 'example', 'subscripts']
  for step in steps:
    if step == 'example':
      swap_example_labels()
    elif step == 'subscripts':
      subscript_titles('sharding-colored5.png', TITLES5, bands=3)
      subscript_titles('sharding-colored6.png', TITLES6, bands=1)
    elif step == '5':
      # The I_xy (row 1, col 1) and J_xy (row 2, col 2) panels shard over a
      # flattened multi-axis dimension and are correct as originally drawn.
      transpose_panels('sharding-colored5.png',
                       skip=frozenset({(1, 1), (2, 2)}))
    else:
      transpose_panels(f'sharding-colored{step}.png')
