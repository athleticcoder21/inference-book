#!/usr/bin/env ruby
# Generate explanatory GIFs for 1D and 2D register tiling in the GEMM chapter.

require "fileutils"

ROOT = File.expand_path("..", __dir__)
FRAME_DIR = File.join(ROOT, "_tmp", "gemm_register_tiling_frames")
OUTPUT_1D = File.join(ROOT, "assets", "gemm", "one-dimensional-register-tiling.gif")
OUTPUT_2D = File.join(ROOT, "assets", "gemm", "two-dimensional-register-tiling.gif")

WIDTH = 960
HEIGHT = 540

COLORS = {
  bg: "#ffffff",
  ink: "#1f2937",
  muted: "#6b7280",
  border: "#d1d5db",
  grid: "#f8fafc",
  tile: "#ffffff",
  a: "#dbeafe",
  a_hot: "#1d4ed8",
  b: "#dcfce7",
  b_hot: "#15803d",
  reg: "#fee2e2",
  reg_hot: "#fb7185",
  thread: "#fef3c7",
  arrow: "#374151",
  dark: "#111827"
}.freeze

def esc(text)
  text.to_s.gsub("&", "&amp;").gsub("<", "&lt;").gsub(">", "&gt;")
end

def box(x, y, w, h, text, fill, stroke = COLORS[:border], klass = "cell")
  <<~SVG
    <rect x="#{x}" y="#{y}" width="#{w}" height="#{h}" rx="8" fill="#{fill}" stroke="#{stroke}" />
    <text x="#{x + w / 2}" y="#{y + h / 2 + 5}" text-anchor="middle" class="#{klass}">#{esc(text)}</text>
  SVG
end

def matrix_cell(x, y, label, fill, klass = "small")
  <<~SVG
    <rect x="#{x}" y="#{y}" width="42" height="34" rx="6" fill="#{fill}" stroke="#{COLORS[:border]}" />
    <text x="#{x + 21}" y="#{y + 22}" text-anchor="middle" class="#{klass}">#{esc(label)}</text>
  SVG
end

def draw_shared_matrix(name, x, y, prefix, fragment_cells, active_cells, light_color, dark_color)
  fragment = fragment_cells.to_h { |row, col| [[row, col], true] }
  active = active_cells.to_h { |row, col| [[row, col], true] }
  body = +""
  body << %(<text x="#{x}" y="#{y - 14}" class="label">#{esc(name)}</text>\n)
  4.times do |row|
    4.times do |col|
      fill =
        if active[[row, col]]
          dark_color
        elsif fragment[[row, col]]
          light_color
        else
          COLORS[:tile]
        end
      body << matrix_cell(x + col * 48, y + row * 40, "#{prefix}#{row}#{col}", fill)
    end
  end
  body
end

def arrow(x1, y1, x2, y2, color = COLORS[:arrow])
  <<~SVG
    <path d="M #{x1} #{y1} C #{(x1 + x2) / 2} #{y1}, #{(x1 + x2) / 2} #{y2}, #{x2} #{y2}" fill="none" stroke="#{color}" stroke-width="2.8" marker-end="url(#arrow)" opacity="0.85" />
  SVG
end

def base(title, note)
  <<~SVG
    <svg xmlns="http://www.w3.org/2000/svg" width="#{WIDTH}" height="#{HEIGHT}" viewBox="0 0 #{WIDTH} #{HEIGHT}">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
          <path d="M 0 0 L 10 4 L 0 8 z" fill="#{COLORS[:arrow]}" />
        </marker>
        <style>
          text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #{COLORS[:ink]}; }
          .title { font-size: 27px; font-weight: 750; }
          .note { font-size: 16px; fill: #{COLORS[:muted]}; }
          .label { font-size: 15px; font-weight: 750; }
          .cell { font-size: 14px; font-weight: 700; }
          .small { font-size: 12px; font-weight: 700; }
          .badge { font-size: 14px; font-weight: 750; fill: #ffffff; }
        </style>
      </defs>
      <rect width="100%" height="100%" fill="#{COLORS[:bg]}" />
      <text x="48" y="48" class="title">#{esc(title)}</text>
      <text x="48" y="78" class="note">#{esc(note)}</text>
  SVG
end

def draw_badges(active)
  labels = ["own outputs", "load values", "reuse", "result"]
  labels.each_with_index.map do |label, index|
    x = 145 + index * 170
    fill = label == active ? COLORS[:dark] : "#f3f4f6"
    color = label == active ? "#ffffff" : COLORS[:muted]
    <<~SVG
      <rect x="#{x}" y="488" width="130" height="30" rx="15" fill="#{fill}" stroke="#{COLORS[:border]}" />
      <text x="#{x + 65}" y="508" text-anchor="middle" class="small" fill="#{color}">#{esc(label)}</text>
    SVG
  end.join
end

def render_1d(step)
  titles = {
    own: ["1D register tiling: one thread owns a vertical strip", "The thread keeps several partial sums in registers."],
    load_b: ["Load one B value", "All outputs in this column need the same B value at this inner position."],
    update0: ["Use the same B value for result0", "A0 is different, but B is reused."],
    update1: ["Use the same B value for result1", "The B register is still live inside the same thread."],
    update2: ["Use the same B value for result2", "One B load is now feeding multiple multiply-adds."],
    update3: ["Use the same B value for result3", "The vertical strip is updated one row at a time."],
    done: ["What changed?", "Four multiply-adds used four A loads and one B load."]
  }
  title, note = titles.fetch(step)
  svg = base(title, note)
  active_index = [:update0, :update1, :update2, :update3].index(step)
  a_fragment = [[0, 1], [1, 1], [2, 1], [3, 1]]
  a_active = active_index ? [[active_index, 1]] : []
  b_fragment = [[1, 2]]
  b_active = [:load_b, :update0, :update1, :update2, :update3].include?(step) ? b_fragment : []
  svg << draw_shared_matrix("shared A tile", 48, 150, "A", a_fragment, a_active, COLORS[:a], COLORS[:a_hot])
  svg << draw_shared_matrix("shared B tile", 280, 150, "B", b_fragment, b_active, COLORS[:b], COLORS[:b_hot])
  svg << %(<text x="510" y="130" class="label">one thread</text>\n)
  svg << box(485, 155, 150, 240, "registers", COLORS[:thread])
  svg << %(<text x="740" y="130" class="label">vertical strip owned by thread</text>\n)
  4.times do |i|
    active = [:update0, :update1, :update2, :update3].index(step) == i || step == :done
    svg << box(740, 155 + i * 58, 92, 42, "result#{i}", active ? COLORS[:reg_hot] : COLORS[:reg])
  end

  if step == :load_b
    svg << arrow(418, 213, 485, 285, COLORS[:b_hot])
  elsif [:update0, :update1, :update2, :update3].include?(step)
    i = active_index
    svg << arrow(114, 167 + i * 40, 485, 215 + i * 28, COLORS[:a_hot])
    svg << arrow(418, 213, 485, 300, COLORS[:b_hot])
    svg << arrow(635, 225 + i * 28, 740, 176 + i * 58, COLORS[:reg_hot])
  end

  if step == :done
    svg << box(350, 420, 270, 48, "B12 loaded once, reused 4 times", COLORS[:dark], COLORS[:dark], "badge")
  end

  active = case step
           when :own then "own outputs"
           when :load_b then "load values"
           when :done then "result"
           else "reuse"
           end
  svg << draw_badges(active)
  svg << "</svg>\n"
  svg
end

def render_2d(step)
  titles = {
    own: ["2D register tiling: one thread owns a small rectangle", "Now the thread has outputs in both row and column directions."],
    load: ["Load A and B fragments", "Two A values and two B values are kept in registers."],
    top: ["A0 meets both B values", "One A value updates the top row of the micro-tile."],
    bottom: ["A1 meets both B values", "One B value is also reused down the column."],
    done: ["What changed?", "The loaded A and B values are reused in both directions."]
  }
  title, note = titles.fetch(step)
  svg = base(title, note)
  a_fragment = [[1, 1], [2, 1]]
  a_active = step == :top ? [[1, 1]] : step == :bottom ? [[2, 1]] : step == :load ? a_fragment : []
  b_fragment = [[1, 2], [1, 3]]
  b_active = [:load, :top, :bottom].include?(step) ? b_fragment : []
  svg << draw_shared_matrix("shared A tile", 48, 150, "A", a_fragment, a_active, COLORS[:a], COLORS[:a_hot])
  svg << draw_shared_matrix("shared B tile", 300, 150, "B", b_fragment, b_active, COLORS[:b], COLORS[:b_hot])
  svg << %(<text x="715" y="135" class="label">2x2 register micro-tile</text>\n)
  labels = [["r00", 716, 180], ["r01", 820, 180], ["r10", 716, 258], ["r11", 820, 258]]
  labels.each_with_index do |(label, x, y), idx|
    active =
      step == :done ||
      (step == :top && idx < 2) ||
      (step == :bottom && idx >= 2)
    svg << box(x, y, 82, 48, label, active ? COLORS[:reg_hot] : COLORS[:reg])
  end

  if step == :load
    svg << arrow(114, 207, 716, 204, COLORS[:a_hot])
    svg << arrow(414, 207, 716, 204, COLORS[:b_hot])
  elsif step == :top
    svg << arrow(114, 207, 716, 204, COLORS[:a_hot])
    svg << arrow(414, 207, 716, 204, COLORS[:b_hot])
    svg << arrow(114, 207, 820, 204, COLORS[:a_hot])
    svg << arrow(462, 207, 820, 204, COLORS[:b_hot])
  elsif step == :bottom
    svg << arrow(114, 247, 716, 282, COLORS[:a_hot])
    svg << arrow(414, 207, 716, 282, COLORS[:b_hot])
    svg << arrow(114, 247, 820, 282, COLORS[:a_hot])
    svg << arrow(462, 207, 820, 282, COLORS[:b_hot])
  elsif step == :done
    svg << box(330, 400, 330, 48, "4 loads -> 4 multiply-adds", COLORS[:dark], COLORS[:dark], "badge")
  end

  active = case step
           when :own then "own outputs"
           when :load then "load values"
           when :done then "result"
           else "reuse"
           end
  svg << draw_badges(active)
  svg << "</svg>\n"
  svg
end

def write_gif(name, steps, output)
  dir = File.join(FRAME_DIR, name)
  FileUtils.rm_rf(dir)
  FileUtils.mkdir_p(dir)
  FileUtils.mkdir_p(File.dirname(output))

  index = 0
  steps.each do |step|
    2.times do
      svg = name == "one_d" ? render_1d(step) : render_2d(step)
      svg_path = File.join(dir, format("frame_%03d.svg", index))
      png_path = File.join(dir, format("frame_%03d.png", index))
      File.write(svg_path, svg)
      system("sips", "-s", "format", "png", svg_path, "--out", png_path, out: File::NULL, err: File::NULL) ||
        abort("sips failed")
      index += 1
    end
  end

  system(
    "ffmpeg",
    "-y",
    "-framerate", "1",
    "-i", File.join(dir, "frame_%03d.png"),
    "-vf", "scale=#{WIDTH}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
    "-loop", "0",
    output,
    out: File::NULL,
    err: File::NULL
  ) || abort("ffmpeg failed")
end

FileUtils.mkdir_p(FRAME_DIR)
write_gif("one_d", [:own, :load_b, :update0, :update1, :update2, :update3, :done], OUTPUT_1D)
write_gif("two_d", [:own, :load, :top, :bottom, :done], OUTPUT_2D)
FileUtils.rm_rf(FRAME_DIR)

puts OUTPUT_1D
puts OUTPUT_2D
