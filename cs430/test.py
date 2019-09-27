#!/usr/bin/env python3
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", help="the input \"Postscript\" file", type=str, default="hw1.ps")
parser.add_argument("-s", help="a float specifying the scaling factor in both dimensions about the world origin", type=float, default=1.0)
parser.add_argument("-r", help="an integer specifying the number of degrees for a counter-clockwise rotation about the world origin", type=int, default=0)
parser.add_argument("-m", help="an integer specifying a translation in the x dimension", type=int, default=0)
parser.add_argument("-n", help="an integer specifying a translation in the y dimension", type=int, default=0)

parser.add_argument("-a", help="an integer lower bound in the x dimension of the world window", type=int, default=0)
parser.add_argument("-b", help="an integer lower bound in the y dimension of the world window", type=int, default=0)
parser.add_argument("-c", help="an integer upper bound in the x dimension of the world window", type=int, default=499)
parser.add_argument("-d", help="an integer upper bound in the y dimension of the world window", type=int, default=499)

args = parser.parse_args()

import numpy as np

## DEFINE GLOBAL VARIABLES
picture = [[]]
xpm = ""

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

def write_xpm_header():
    global xpm
    xpm_header = \
'''
/* XPM */
static char *sco100[] = {{
/* width height num_colors chars_per_pixel */
"{width} {height} {num_colors} {chars_per_pixel}",
/* colors */
"0 c #ffffff",
"1 c #000000",
/* pixels */
'''.format(
        width=args.c-args.a+1, 
        height=args.d-args.b+1,
        num_colors=2,
        chars_per_pixel=1,
        )
    xpm += xpm_header

def write_xpm_pixels():
    global picture, xpm
    xpm_pixel = ""
    for r in picture:
        s = "\"{}\",\r\n".format("".join(map(str, r)))
        xpm_pixel += s

    xpm += xpm_pixel
    xpm += \
'''
};
'''

# Using bresenham
def create_line(l):
    # print(l)
    global picture
    line = []
    x1, x2, y1, y2 = int(l[0]), int(l[2]), int(l[1]), int(l[3])
    if x1 <= x2:
        q = Point(x1, y1)
        r = Point(x2, y2)
    else:
        q = Point(x2, y2)
        r = Point(x1, y1)
    dx, dy = r.x - q.x, r.y - q.y
    y = q.y
    x = q.x
    D = 2*dy - dx
    while True:
        if dy >= 0:
            if x > args.c or y > args.d or x > r.x or y > r.y:
                break
            if x >= args.a and y >= args.b:
                picture[499-y][x] = 1
            if D <= 0:
                D += 2*dy
                if dx > 0:
                    x += 1
                if dx < 0:
                    x -= 1
            else:
                D -= 2*dx
                if dy > 0:
                    y += 1
                if dy < 0:
                    y -= 1
        else:
            if x > args.c or y < args.b or x > r.x or y < r.y:
                break
            if x >= args.a and y <= args.d:
                picture[499-y][x] = 1
            if D <= 0:
                D -= 2*dy
                if dx > 0:
                    x += 1
                if dx < 0:
                    x -= 1
            else:
                D -= 2*dx
                if dy > 0:
                    y += 1
                if dy < 0:
                    y -= 1

def parse_ps():
    with open(args.f) as fp:
        is_started = 0
        for cnt, line in enumerate(fp):
            line = line.strip()
            if line == "%%%END":
                break
            if is_started:
                line_args = line.split()
                if line_args[-1].upper() == "LINE":
                    create_line(line_args[:-1])
            if line == "%%%BEGIN":
                is_started = 1

                

def parse_window():
    global picture
    picture = np.zeros(shape=(args.c-args.a+1, args.d-args.b+1), dtype=int)

def main():
    global picture, xpm
    parse_window()
    parse_ps()
    write_xpm_header()
    write_xpm_pixels()
    print(xpm)

if __name__ == "__main__":
    main()
