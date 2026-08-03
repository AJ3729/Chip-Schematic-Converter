"""ASCII ink dump - pixel-exact view of a small region. Stroke widths and
1-px gaps are visible here in a way no rendered PNG can show.

usage: python3 inkmap.py <stem> <x> <y> [win=22] [thresh=160] [root]
"""
import sys, cv2, numpy as np
stem = sys.argv[1]; x = int(sys.argv[2]); y = int(sys.argv[3])
win = int(sys.argv[4]) if len(sys.argv) > 4 else 22
th = int(sys.argv[5]) if len(sys.argv) > 5 else 160
root = sys.argv[6] if len(sys.argv) > 6 else "/home/claude/val"
g = cv2.imread(f"{root}/img1024/{stem}.jpg", 0)
H, W = g.shape
x0, y0 = max(0, x - win), max(0, y - win)
x1, y1 = min(W, x + win + 1), min(H, y + win + 1)
sub = g[y0:y1, x0:x1]
print(f"{stem}  x[{x0},{x1})  y[{y0},{y1})   '#'<{th}  '+'<205  '.'=paper   * = ({x},{y})")
hdr = "     " + "".join(str((x0 + i) // 100 % 10) for i in range(x1 - x0))
print(hdr)
print("     " + "".join(str((x0 + i) // 10 % 10) for i in range(x1 - x0)))
print("     " + "".join(str((x0 + i) % 10) for i in range(x1 - x0)))
for j in range(y1 - y0):
    row = ""
    for i in range(x1 - x0):
        v = sub[j, i]
        ch = "#" if v < th else ("+" if v < 205 else ".")
        if (y0 + j) == y and (x0 + i) == x and ch == ".":
            ch = "*"
        row += ch
    print(f"{y0+j:4d} {row}")
