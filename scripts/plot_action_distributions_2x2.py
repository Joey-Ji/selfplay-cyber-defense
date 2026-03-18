"""Rearrange the existing 1x4 red_action_distributions.png into a 2x2 grid.

The original image is 3583x887 with 4 equal-width subplots side by side.
This script splits it into 4 panels and arranges them in a 2x2 grid so
the figure is taller and more readable in a single-column layout.

Usage:
    python scripts/plot_action_distributions_2x2.py
"""

from pathlib import Path
from PIL import Image

INPUT = Path("docs/final_report/figures/red_action_distributions.png")
OUTPUT = Path("docs/final_report/figures/red_action_distributions.png")


def main():
    img = Image.open(INPUT)
    w, h = img.size
    # The image has a shared title at the top (~50px) and 4 equal subplots below.
    # We'll split into 4 equal vertical strips and arrange 2x2.
    panel_w = w // 4

    panels = []
    for i in range(4):
        box = (i * panel_w, 0, (i + 1) * panel_w, h)
        panels.append(img.crop(box))

    # Create 2x2 grid with a small gap
    gap = 10
    grid_w = panel_w * 2 + gap
    grid_h = h * 2 + gap
    grid = Image.new("RGB", (grid_w, grid_h), "white")

    positions = [
        (0, 0),
        (panel_w + gap, 0),
        (0, h + gap),
        (panel_w + gap, h + gap),
    ]
    for panel, (x, y) in zip(panels, positions):
        grid.paste(panel, (x, y))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    grid.save(OUTPUT, dpi=(200, 200))
    print(f"Saved 2x2 grid to {OUTPUT} ({grid_w}x{grid_h})")


if __name__ == "__main__":
    main()
