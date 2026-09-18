"""Reproduces assets/img/sharding-colored1.png (Data / Devices sharding figure).

Draws the figure in the original's pixel coordinates (1757x1058 @ 100 dpi) so it
can be compared 1:1 against the PNG. Also emits a variant with the mesh axes X
and Y swapped.

Usage:
  uv run --with matplotlib --with fonttools python _scripts/sharding_figure.py [-o OUTDIR]
"""

import argparse
import pathlib
import tempfile

import fontTools.ttLib
import matplotlib.font_manager
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

W, H = 1757, 1058
DPI = 100

# Cell fills sampled from the original PNG, row-major.
CELL_COLORS = [
    ['#ccff99', '#99ff33', '#ea9cf6', '#d439ed'],
    ['#b3e580', '#8abd58', '#d083dc', '#a85bb4'],
    ['#f4e19f', '#e8c33e', '#9fe0f6', '#3fc1ed'],
    ['#dac885', '#b29f5d', '#86c7dc', '#5e9eb4'],
]
DEVICE_GRAY = '#efecea'

SANS = 'PT Sans'  # big labels: J, I, X, Y, Data, Devices, device numbers


def mono_font() -> matplotlib.font_manager.FontProperties:
  """PT Mono Regular, for the cell digits (slashed zero).

  macOS ships PTMono.ttc with the Bold face first, and matplotlib only ever
  loads face 0 of a .ttc — so asking for family 'PT Mono' silently renders
  bold. Extract the regular face to a .ttf and reference it by path instead.
  """
  out = pathlib.Path(tempfile.gettempdir()) / 'PTMono-Regular.ttf'
  if not out.exists():
    tc = fontTools.ttLib.TTCollection(
        '/System/Library/Fonts/Supplemental/PTMono.ttc', lazy=True)
    face = next(f for f in tc.fonts if f['OS/2'].usWeightClass == 400)
    face.save(str(out))
  return matplotlib.font_manager.FontProperties(fname=str(out))


MONO = mono_font()

# Geometry measured from the original (pixels, y increases downward).
GRID_X0, GRID_X1 = 175, 754
GRID_Y0, GRID_Y1 = 238, 816
SQ_XS = [(1154, 1438), (1471, 1755)]  # device square x-extents per column
SQ_YS = [(220, 499), (527, 806)]      # device square y-extents per row
ARROW_Y = 179                          # y of both horizontal arrows
I_ARROW_X, Y_ARROW_X = 106, 1112       # x of the two vertical arrows


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
  ax.add_patch(mpatches.FancyArrowPatch(
      start, end, arrowstyle='-|>', mutation_scale=35, lw=2.88,
      color='black', shrinkA=0, shrinkB=0, joinstyle='miter'))


def make_figure(col_label: str, row_label: str,
                device_labels: list[list[int]]) -> plt.Figure:
  """Draws the Data/Devices figure.

  Args:
    col_label: Mesh axis name along the device columns (horizontal arrow).
    row_label: Mesh axis name along the device rows (vertical arrow).
    device_labels: 2x2 device ids, row-major.
  """
  fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
  ax = fig.add_axes((0, 0, 1, 1))
  ax.set_xlim(0, W)
  ax.set_ylim(H, 0)
  ax.set_aspect('equal')
  ax.axis('off')

  # --- Left panel: data grid ---
  cw = (GRID_X1 - GRID_X0) / 4
  ch = (GRID_Y1 - GRID_Y0) / 4
  for i in range(4):
    for j in range(4):
      x, y = GRID_X0 + j * cw, GRID_Y0 + i * ch
      ax.add_patch(mpatches.Rectangle((x, y), cw, ch,
                                      facecolor=CELL_COLORS[i][j],
                                      edgecolor='black', lw=3))
      ax.text(x + cw / 2, y + ch / 2 + 7, f'{i}{j}', fontproperties=MONO,
              fontsize=59, ha='center', va='center', color='black')

  arrow(ax, (GRID_X0 + 2, ARROW_Y), (GRID_X1 - 5, ARROW_Y))     # J
  arrow(ax, (I_ARROW_X, GRID_Y0 + 14), (I_ARROW_X, GRID_Y1 - 15))  # I
  ax.text(228, 97, 'J', family=SANS, fontsize=112, ha='center', va='center')
  ax.text(40, 340, 'I', family=SANS, fontsize=112, ha='center', va='center')
  ax.text(450, 980, 'Data', family=SANS, fontsize=112, ha='center',
          va='center')

  # --- Right panel: device mesh ---
  for r, (sy0, sy1) in enumerate(SQ_YS):
    for c, (sx0, sx1) in enumerate(SQ_XS):
      ax.add_patch(mpatches.FancyBboxPatch(
          (sx0 + 35, sy0 + 35), sx1 - sx0 - 70, sy1 - sy0 - 70,
          boxstyle='round,pad=35,rounding_size=28',
          facecolor=DEVICE_GRAY, edgecolor='none'))
      ax.text((sx0 + sx1) / 2, (sy0 + sy1) / 2 + 11, str(device_labels[r][c]),
              family=SANS, fontsize=110, ha='center', va='center')

  arrow(ax, (SQ_XS[0][0] + 10, ARROW_Y), (SQ_XS[1][1] - 5, ARROW_Y))
  arrow(ax, (Y_ARROW_X, SQ_YS[0][0] + 12), (Y_ARROW_X, SQ_YS[1][1] - 3))
  ax.text(1210, 96, col_label, family=SANS, fontsize=112, ha='center',
          va='center')
  ax.text(1025, 306, row_label, family=SANS, fontsize=112, ha='center',
          va='center')
  ax.text(1463, 980, 'Devices', family=SANS, fontsize=112, ha='center',
          va='center')
  return fig


def axis_label(ax, x: float, y: float, main: str, sub: str,
               size: float = 58) -> None:
  """Draws an axis label like I_X at baseline (x, y), main glyph centered."""
  ax.text(x, y, main, family=SANS, fontsize=size, ha='center', va='baseline')
  ax.text(x + size * (0.28 if main == 'I' else 0.38), y + size * 0.21, sub,
          family=SANS, fontsize=size * 0.62, ha='left', va='baseline')


def make_example_figure() -> plt.Figure:
  """Draws the intro sharding example: 4x4 data grid -> per-TPU 2x2 blocks."""
  W, H = 1580, 745
  fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
  ax = fig.add_axes((0, 0, 1, 1))
  ax.set_xlim(0, W)
  ax.set_ylim(H, 0)
  ax.set_aspect('equal')
  ax.axis('off')

  # Left: the unsharded 4x4 array.
  cell = 130
  gx, gy = 60, 165
  for i in range(4):
    for j in range(4):
      x, y = gx + j * cell, gy + i * cell
      ax.add_patch(mpatches.Rectangle((x, y), cell, cell,
                                      facecolor=CELL_COLORS[i][j],
                                      edgecolor='black', lw=3))
      ax.text(x + cell / 2, y + cell / 2 + 6, f'{i}{j}', fontproperties=MONO,
              fontsize=53, ha='center', va='center', color='black')

  arrow(ax, (685, 425), (795, 425))

  # Right: the same array sharded A[I_X, J_Y] over a 2x2 mesh; device (r, c)
  # holds row-block r and column-block c, numbered row-major like colored1.
  # The mesh spans the same height as the data grid on the left.
  sq, gap, bcell = 250, 20, 80
  mx, my = 1000, 165
  arrow(ax, (mx + 5, my - 45), (mx + 2 * sq + gap - 4, my - 45))       # J_Y
  arrow(ax, (mx - 45, my + 5), (mx - 45, my + 2 * sq + gap - 6))       # I_X
  axis_label(ax, mx + 60, my - 78, 'J', 'Y', size=48)
  axis_label(ax, mx - 112, my + 80, 'I', 'X', size=48)
  for r in range(2):
    for c in range(2):
      sx, sy = mx + c * (sq + gap), my + r * (sq + gap)
      ax.add_patch(mpatches.FancyBboxPatch(
          (sx + 26, sy + 26), sq - 52, sq - 52,
          boxstyle='round,pad=26,rounding_size=21',
          facecolor=DEVICE_GRAY, edgecolor='none'))
      ax.text(sx + 18, sy + 36, f'TPU:{2 * r + c}', family=SANS, fontsize=15,
              ha='left', va='baseline')
      bx, by = sx + (sq - 2 * bcell) / 2, sy + 60
      for i in range(2):
        for j in range(2):
          x, y = bx + j * bcell, by + i * bcell
          ax.add_patch(mpatches.Rectangle(
              (x, y), bcell, bcell, facecolor=CELL_COLORS[2 * r + i][2 * c + j],
              edgecolor='black', lw=2.3))
          ax.text(x + bcell / 2, y + bcell / 2 + 4, f'{2 * r + i}{2 * c + j}',
                  fontproperties=MONO, fontsize=33, ha='center', va='center')
  return fig


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('-o', '--outdir', default='.')
  args = parser.parse_args()

  fig = make_figure('X', 'Y', [[0, 1], [2, 3]])
  fig.savefig(f'{args.outdir}/sharding-colored1-repro.png')

  # Swapped variant: Y runs along columns, X along rows. Device numbers stay
  # put (they label physical devices); change device_labels to [[0, 2], [1, 3]]
  # if you want the numbering to follow the mesh axes instead.
  fig = make_figure('Y', 'X', [[0, 1], [2, 3]])
  fig.savefig(f'{args.outdir}/sharding-colored1-swapped.png')

  fig = make_example_figure()
  fig.savefig(f'{args.outdir}/sharding-example.png')


if __name__ == '__main__':
  main()
