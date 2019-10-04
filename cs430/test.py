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
parser.add_argument("--stdout", help="print output or not", type=int, default=1)

args = parser.parse_args()

import numpy as np
import math

## DEFINE GLOBAL VARIABLES
window_size = (0,0)
lowest_x, lowest_y = (0, 0)
picture = []
xpm = ""
trans_mat = []


class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def less_than(self, P):
        if self.x > P.x and self.y > P.y:
            return False
        return True
    
    def greater_than(self, P):
        if self.x < P.x and self.y < P.y:
            return False
        return True

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
        # width=args.c-args.a+1, 
        # height=args.d-args.b+1,
        width=picture.shape[1], 
        height=picture.shape[0],
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
def create_line(lines):
    global picture

    for l in lines:
        x1, x2, y1, y2 = int(l[0])-lowest_x, int(l[2])-lowest_y, int(l[1])-lowest_x, int(l[3])-lowest_y
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
                # if x > args.c or y > args.d or x > r.x or y > r.y:
                if x > r.x or y > r.y:
                    break
                # if x >= args.a and y >= args.b:
                picture[picture.shape[0]-1-y][x-1] = 1
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
                # if x > args.c or y < args.b or x > r.x or y < r.y:
                if x > r.x or y < r.y:
                    break
                # if x >= args.a and y <= args.d:
                picture[picture.shape[0]-1-y][x-1] = 1
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


def get_max_dim(lines):
    global picture, lowest_x, lowest_y
    lowest_x = 0
    lowest_y = 0
    highest_x = 0
    highest_y = 0

    for l in lines:
        l = list(map(int, l))
        if l[0] < l[2]:
            if l[0] < lowest_x:
                lowest_x = l[0]
            if l[2] > highest_x:
                highest_x = l[2]
        else:
            if l[2] < lowest_x:
                lowest_x = l[2]
            if l[0] > highest_x:
                highest_x = l[0]
        if l[1] < l[3]:
            if l[1] < lowest_y:
                lowest_y = l[1]
            if l[3] > highest_y:
                highest_y = l[3]
        else:
            if l[3] < lowest_y:
                lowest_y = l[3]
            if l[1] > highest_y:
                highest_y = l[1]
    print (lowest_x, lowest_y, highest_x, highest_y)
    return (highest_x-lowest_x, highest_y-lowest_y)

def parse_ps():
    global picture
    # /home/sweet/2-coursework/cs430/hw1.ps
    # with open(args.f) as fp:
    lines = []
    with open("/home/sweet/2-coursework/cs430/hw1.ps") as fp:
        is_started = 0
        for cnt, line in enumerate(fp):
            line = line.strip()
            if line == "%%%END":
                break
            if is_started:
                line_args = line.split()
                if line_args[-1].upper() == "LINE":
                    lines.append(line_args[:-1])
            if line == "%%%BEGIN":
                is_started = 1
    max_x, max_y = get_max_dim(lines)
    if window_size[0] > max_x:
        max_x = window_size[0]
    if window_size[1] > max_y:
        max_y = window_size[1]
                
    picture = np.zeros((max_y, max_x), dtype=int)
    create_line(lines)
    print(picture.shape)

def parse_window():
    global window_size
    window_size = (args.c-args.a, args.d-args.b)

def parse_transf():
    global trans_mat, picture
    scale_mat = [[args.s,0,0], [0,args.s,0], [0,0,1]]   
    rad = np.radians(args.r)
    cos = np.cos(rad)
    sin = np.sin(rad)
    rot_mat = [[cos,sin,0], [-sin,cos,0], [0,0,1]]
    transl_mat = [[1,0,args.m], [0,1,-args.n], [0,0,1]]
    trans_mat = np.matmul(transl_mat, rot_mat)
    trans_mat = np.matmul(trans_mat, scale_mat)

    result_pic = np.zeros(picture.shape, dtype=int)

    # k = picture.shape[0]-window_size[1]-lowest_y
    k = window_size[1]-lowest_y
    l = 0-lowest_x

    for r in range(0, len(picture)):
        for c in range(0, len(picture[0])):
            idx_mat = [[c-l],[r-k],[1]]
            idx_mat = np.matmul(trans_mat, idx_mat)
            idx_mat = idx_mat.astype(int)
            new_r = idx_mat[1][0] + k
            new_c = idx_mat[0][0] + l
            if (0 <= new_r < picture.shape[0]) and (0 <= new_c < picture.shape[1]):
                result_pic[new_r][new_c] = picture[r][c]
    
    picture = result_pic


def crop_final_pic():
    global picture
    # picture = picture[args.b-lowest_y:window_size[1]-lowest_y, args.a-lowest_x:args.c-lowest_x]
    picture = picture[115:115+349, 0:700]

def main():
    global xpm
    parse_window()
    parse_ps()
    parse_transf()
    crop_final_pic()
    write_xpm_header()
    write_xpm_pixels()
    if args.stdout:
        print(xpm)

if __name__ == "__main__":
    main()
