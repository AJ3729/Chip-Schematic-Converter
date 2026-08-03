"""zoom.py <stem> <x> <y> [win=60] [zoom=6] [root=/home/claude/val]

Crop a window of the 1024x1024 drawing around (x,y), upscale it with
nearest-neighbour, draw a light coordinate grid every 10 source px, and write
a PNG.  Prints the path.
"""
from __future__ import annotations
import sys, os
import cv2
import numpy as np

OUT = "/home/claude/zoom"


def zoom(stem, cx, cy, win=60, zm=6, root="/home/claude/val", grid=10):
    p = f"{root}/img1024/{stem}.jpg"
    g = cv2.imread(p, 0)
    H, W = g.shape
    x0, y0 = max(0, int(cx) - win), max(0, int(cy) - win)
    x1, y1 = min(W, int(cx) + win), min(H, int(cy) + win)
    crop = cv2.cvtColor(g[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    big = cv2.resize(crop, None, fx=zm, fy=zm, interpolation=cv2.INTER_NEAREST)
    # grid + labels in source coords
    for x in range(x0 - x0 % grid, x1, grid):
        X = int((x - x0) * zm)
        if 0 <= X < big.shape[1]:
            col = (255, 190, 190) if x % (grid * 5) else (255, 120, 120)
            big[:, X] = np.minimum(big[:, X], np.array(col, np.uint8))
            if x % (grid * 5) == 0:
                cv2.putText(big, str(x), (X + 2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, 0, 220), 1, cv2.LINE_AA)
    for y in range(y0 - y0 % grid, y1, grid):
        Y = int((y - y0) * zm)
        if 0 <= Y < big.shape[0]:
            col = (255, 190, 190) if y % (grid * 5) else (255, 120, 120)
            big[Y, :] = np.minimum(big[Y, :], np.array(col, np.uint8))
            if y % (grid * 5) == 0:
                cv2.putText(big, str(y), (2, Y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, 0, 220), 1, cv2.LINE_AA)
    os.makedirs(OUT, exist_ok=True)
    o = f"{OUT}/{stem}_{int(cx)}_{int(cy)}_w{win}_z{zm}.png"
    cv2.imwrite(o, big)
    return o


if __name__ == "__main__":
    a = sys.argv[1:]
    stem, cx, cy = a[0], float(a[1]), float(a[2])
    win = int(a[3]) if len(a) > 3 else 60
    zm = int(a[4]) if len(a) > 4 else 6
    root = a[5] if len(a) > 5 else "/home/claude/val"
    print(zoom(stem, cx, cy, win, zm, root))
