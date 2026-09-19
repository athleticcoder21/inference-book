#!/usr/bin/env ruby
# Generate a small explanatory GIF for the shared-memory tiled GEMM section.

require "fileutils"

ROOT = File.expand_path("..", __dir__)
FRAME_DIR = File.join(ROOT, "_tmp", "gemm_tiling_frames")
OUTPUT = File.join(ROOT, "assets", "gemm", "shared-memory-tiling.gif")

WIDTH = 960
HEIGHT = 540
CELL = 34
GAP = 5

COLORS = {
  bg: "#ffffff",
  ink: "#2a2a2a",
  muted: "#6b7280",
  border: "#d1d5db",
  grid: "#e5e7eb",
  a: "#dbeafe",
  a_hot: "#60a5fa",
  b: "#dcfce7",
  b_hot: "#4ade80",
  c: "#fee2e2",
  c_hot: "#fb7185",
  shared: "#fef3c7",
  shared_hot: "#fbbf24",
  thread0: "#bfdbfe",
  thread1: "#bbf7d0",
  thread2: "#fecaca",
  thread3: "#ddd6fe",
  barrier: "#111827",
  arrow: "#374151"
}.freeze

STEPS = [
  {
    title: "One block owns one 2x2 tile of C",
    note: "Four threads keep four partial sums. The output tile will stay fixed.",
    a: [],
    b: [],
    c: [[0, 0], [0, 1], [1, 0], [1, 1]],
    shared_a: [],
    shared_b: [],
    stage: :own
  },
  {
    title: "tile_start = 0: cooperative load",
    note: "Each thread loads one A value and one B value into shared memory.",
    a: [[0, 0], [0, 1], [1, 0], [1, 1]],
    b: [[0, 0], [0, 1], [1, 0], [1, 1]],
    c: [[0, 0], [0, 1], [1, 0], [1, 1]],
    shared_a: [[0, 0], [0, 1], [1, 0], [1, 1]],
    shared_b: [[0, 0], [0, 1], [1, 0], [1, 1]],
    stage: :load_first
  },
  {
    title: "Barrier 1: do not read half-loaded tiles",
    note: "All threads wait until both shared tiles are complete.",
    a: [[0, 0], [0, 1], [1, 0], [1, 1]],
    b: [[0, 0], [0, 1], [1, 0], [1, 1]],
    c: [[0, 0], [0, 1], [1, 0], [1, 1]],
    shared_a: [[0, 0], [0, 1], [1, 0], [1, 1]],
    shared_b: [[0, 0], [0, 1], [1, 0], [1, 1]],
    stage: :barrier_load
  },
  {
    title: "Use shared tiles for the first partial dot products",
    note: "Example: C00 += a00*b00 + a01*b10.",
    a: [[0, 0], [0, 1], [1, 0], [1, 1]],
    b: [[0, 0], [0, 1], [1, 0], [1, 1]],
    c: [[0, 0]],
    shared_a: [[0, 0], [0, 1]],
    shared_b: [[0, 0], [1, 0]],
    stage: :compute_first
  },
  {
    title: "Barrier 2: do not overwrite tiles too early",
    note: "Fast threads wait until every thread has finished reading this pair.",
    a: [[0, 0], [0, 1], [1, 0], [1, 1]],
    b: [[0, 0], [0, 1], [1, 0], [1, 1]],
    c: [[0, 0], [0, 1], [1, 0], [1, 1]],
    shared_a: [[0, 0], [0, 1], [1, 0], [1, 1]],
    shared_b: [[0, 0], [0, 1], [1, 0], [1, 1]],
    stage: :barrier_overwrite
  },
  {
    title: "tile_start = 2: reuse the same shared arrays",
    note: "The output tile is still the same. Only the K-slice moved.",
    a: [[0, 2], [0, 3], [1, 2], [1, 3]],
    b: [[2, 0], [2, 1], [3, 0], [3, 1]],
    c: [[0, 0], [0, 1], [1, 0], [1, 1]],
    shared_a: [[0, 0], [0, 1], [1, 0], [1, 1]],
    shared_b: [[0, 0], [0, 1], [1, 0], [1, 1]],
    stage: :load_second
  },
  {
    title: "Add the second partial dot products",
    note: "Example: C00 += a02*b20 + a03*b30.",
    a: [[0, 2], [0, 3], [1, 2], [1, 3]],
    b: [[2, 0], [2, 1], [3, 0], [3, 1]],
    c: [[0, 0]],
    shared_a: [[0, 0], [0, 1]],
    shared_b: [[0, 0], [1, 0]],
    stage: :compute_second
  },
  {
    title: "Now the fixed C tile is complete",
    note: "The block walked across K while accumulating into the same outputs.",
    a: [],
    b: [],
    c: [[0, 0], [0, 1], [1, 0], [1, 1]],
    shared_a: [],
    shared_b: [],
    stage: :done
  }
].freeze

def esc(text)
  text.to_s.gsub("&", "&amp;").gsub("<", "&lt;").gsub(">", "&gt;")
end

def matrix_cell(x, y, label, fill, stroke = COLORS[:grid], text_color = COLORS[:ink])
  <<~SVG
    <rect x="#{x}" y="#{y}" width="#{CELL}" height="#{CELL}" rx="4" fill="#{fill}" stroke="#{stroke}" />
    <text x="#{x + CELL / 2}" y="#{y + 22}" text-anchor="middle" class="cell" fill="#{text_color}">#{esc(label)}</text>
  SVG
end

def draw_matrix(name, origin_x, origin_y, rows, columns, prefix, base_fill, hot_fill, hot_cells)
  hot = hot_cells.to_h { |row, col| [[row, col], true] }
  body = +""
  body << %(<text x="#{origin_x}" y="#{origin_y - 12}" class="label">#{esc(name)}</text>\n)
  rows.times do |row|
    columns.times do |col|
      fill = hot[[row, col]] ? hot_fill : base_fill
      label = "#{prefix}#{row}#{col}"
      body << matrix_cell(origin_x + col * (CELL + GAP), origin_y + row * (CELL + GAP), label, fill)
    end
  end
  body
end

def draw_shared(name, origin_x, origin_y, prefix, hot_fill, hot_cells)
  hot = hot_cells.to_h { |row, col| [[row, col], true] }
  body = +""
  body << %(<text x="#{origin_x}" y="#{origin_y - 12}" class="label">#{esc(name)}</text>\n)
  2.times do |row|
    2.times do |col|
      fill = hot[[row, col]] ? hot_fill : COLORS[:shared]
      body << matrix_cell(origin_x + col * (CELL + GAP), origin_y + row * (CELL + GAP), "#{prefix}#{row}#{col}", fill)
    end
  end
  body
end

def arrow(x1, y1, x2, y2, color = COLORS[:arrow])
  <<~SVG
    <path d="M #{x1} #{y1} C #{(x1 + x2) / 2} #{y1}, #{(x1 + x2) / 2} #{y2}, #{x2} #{y2}" fill="none" stroke="#{color}" stroke-width="2.5" marker-end="url(#arrow)" opacity="0.8" />
  SVG
end

def barrier(x, y, title, subtitle)
  <<~SVG
    <g>
      <rect x="#{x}" y="#{y}" width="300" height="76" rx="10" fill="#{COLORS[:barrier]}" opacity="0.94" />
      <text x="#{x + 150}" y="#{y + 30}" text-anchor="middle" class="barrier-title">#{esc(title)}</text>
      <text x="#{x + 150}" y="#{y + 54}" text-anchor="middle" class="barrier-note">#{esc(subtitle)}</text>
    </g>
  SVG
end

def threads(x, y, active = true)
  fills = [COLORS[:thread0], COLORS[:thread1], COLORS[:thread2], COLORS[:thread3]]
  labels = ["T00", "T01", "T10", "T11"]
  body = +""
  body << %(<text x="#{x}" y="#{y - 12}" class="label">threads in one block</text>\n)
  2.times do |row|
    2.times do |col|
      idx = row * 2 + col
      fill = active ? fills[idx] : "#f3f4f6"
      body << matrix_cell(x + col * (CELL + GAP), y + row * (CELL + GAP), labels[idx], fill)
    end
  end
  body
end

def stage_badges(stage)
  labels = [
    ["load", [:load_first, :load_second].include?(stage)],
    ["barrier 1", stage == :barrier_load],
    ["compute", [:compute_first, :compute_second].include?(stage)],
    ["barrier 2", stage == :barrier_overwrite],
    ["next K tile", stage == :load_second],
  ]
  body = +""
  labels.each_with_index do |(label, active), i|
    x = 110 + i * 144
    fill = active ? COLORS[:barrier] : "#f3f4f6"
    color = active ? "#ffffff" : COLORS[:muted]
    body << %(<rect x="#{x}" y="488" width="118" height="30" rx="15" fill="#{fill}" stroke="#{COLORS[:border]}" />\n)
    body << %(<text x="#{x + 59}" y="508" text-anchor="middle" class="badge" fill="#{color}">#{esc(label)}</text>\n)
  end
  body
end

def render(step)
  svg = +""
  svg << <<~SVG
    <svg xmlns="http://www.w3.org/2000/svg" width="#{WIDTH}" height="#{HEIGHT}" viewBox="0 0 #{WIDTH} #{HEIGHT}">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
          <path d="M 0 0 L 10 4 L 0 8 z" fill="#{COLORS[:arrow]}" />
        </marker>
        <style>
          text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
          .title { font-size: 26px; font-weight: 700; fill: #{COLORS[:ink]}; }
          .note { font-size: 16px; fill: #{COLORS[:muted]}; }
          .label { font-size: 15px; font-weight: 700; fill: #{COLORS[:ink]}; }
          .cell { font-size: 12px; font-weight: 650; }
          .badge { font-size: 13px; font-weight: 700; }
          .barrier-title { font-size: 18px; font-weight: 800; fill: #ffffff; }
          .barrier-note { font-size: 13px; fill: #d1d5db; }
        </style>
      </defs>
      <rect width="100%" height="100%" fill="#{COLORS[:bg]}" />
      <text x="48" y="48" class="title">#{esc(step[:title])}</text>
      <text x="48" y="76" class="note">#{esc(step[:note])}</text>
  SVG

  svg << draw_matrix("A in global memory", 48, 130, 4, 4, "a", "#f8fafc", COLORS[:a_hot], step[:a])
  svg << draw_matrix("B in global memory", 736, 130, 4, 4, "b", "#f8fafc", COLORS[:b_hot], step[:b])
  svg << draw_shared("shared A tile", 342, 142, "A", COLORS[:a_hot], step[:shared_a])
  svg << draw_shared("shared B tile", 496, 142, "B", COLORS[:b_hot], step[:shared_b])
  svg << draw_matrix("fixed C tile", 403, 334, 2, 2, "c", COLORS[:c], COLORS[:c_hot], step[:c])
  svg << threads(48, 334, step[:stage] != :done)

  if [:load_first, :load_second].include?(step[:stage])
    svg << arrow(203, 196, 342, 176)
    svg << arrow(736, 196, 574, 176)
  elsif [:compute_first, :compute_second].include?(step[:stage])
    svg << arrow(419, 224, 430, 334)
    svg << arrow(535, 224, 462, 334)
  end

  if step[:stage] == :barrier_load
    svg << barrier(330, 280, "Barrier 1", "loading must finish before reading")
  elsif step[:stage] == :barrier_overwrite
    svg << barrier(330, 280, "Barrier 2", "reading must finish before overwrite")
  elsif step[:stage] == :done
    svg << barrier(330, 170, "done", "write the accumulated sums to C")
  end

  svg << stage_badges(step[:stage])
  svg << "</svg>\n"
  svg
end

FileUtils.rm_rf(FRAME_DIR)
FileUtils.mkdir_p(FRAME_DIR)
FileUtils.mkdir_p(File.dirname(OUTPUT))

frame_number = 0
STEPS.each do |step|
  2.times do
    path = File.join(FRAME_DIR, format("frame_%03d.svg", frame_number))
    File.write(path, render(step))
    png_path = File.join(FRAME_DIR, format("frame_%03d.png", frame_number))
    system("sips", "-s", "format", "png", path, "--out", png_path, out: File::NULL, err: File::NULL) ||
      abort("sips failed")
    frame_number += 1
  end
end

system(
  "ffmpeg",
  "-y",
  "-framerate", "1",
  "-i", File.join(FRAME_DIR, "frame_%03d.png"),
  "-vf", "scale=#{WIDTH}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
  "-loop", "0",
  OUTPUT
) || abort("ffmpeg failed")

FileUtils.rm_rf(File.join(ROOT, "_tmp"))
puts OUTPUT
