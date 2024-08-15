#!/usr/bin/env python3
'''
Given several g-code files, compute the overall bounding box of all models.
The first g-code is for the original orientation.
'''
import argparse
import re
from pathlib import Path
import numpy as np
import math
import sys

def triple(arg):
    return [float(n) for n in arg.split(",")]

def extract_MinX(input_line):
    m = re.match(r";MINX:(?P<x>(\d*)?(\.)?(\d*)?)", input_line)
    minX = float(m.group('x'))
    return minX

def extract_MinY(input_line):
    m = re.match(r";MINY:(?P<x>(\d*)?(\.)?(\d*)?)", input_line)
    minY = float(m.group('x'))
    return minY

def extract_MaxX(input_line):
    m = re.match(r";MAXX:(?P<x>(\d*)?(\.)?(\d*)?)", input_line)
    maxX = float(m.group('x'))
    return maxX

def extract_MaxY(input_line):
    m = re.match(r";MAXY:(?P<x>(\d*)?(\.)?(\d*)?)", input_line)
    maxY = float(m.group('x'))
    return maxY

def extract_MaxZ(input_line):
    m = re.match(r";MAXZ:(?P<x>(\d*)?(\.)?(\d*)?)", input_line)
    maxZ = float(m.group('x'))
    return maxZ

def parse_bbox(gcode):
    minx, miny, minz, maxx, maxy, maxz = 0, 0, 0, 0, 0, 0
    p = Path(gcode)
    with p.open() as file:
        for instr in file:
            if instr.startswith(';MINX'):
                minx = extract_MinX(instr)
            elif instr.startswith(';MINY'):
                miny = extract_MinY(instr)
            elif instr.startswith(';MAXX'):
                maxx = extract_MaxX(instr)
            elif instr.startswith(';MAXY'):
                maxy = extract_MaxY(instr)
            elif instr.startswith(';MAXZ'):
                maxz = extract_MaxZ(instr)
                break
    return [[minx, miny, minz], [maxx, maxy, maxz]]

def rotation_matrix(agl):
    rot_x = np.asarray([[1, 0, 0], [0, math.cos(agl[0]), -math.sin(agl[0])], [0, math.sin(agl[0]), math.cos(agl[0])]])
    # in cura need to negate the y angle
    rot_y = np.asarray([[math.cos(agl[1]), 0, math.sin(-agl[1])], [0, 1, 0], [-math.sin(-agl[1]), 0, math.cos(agl[1])]])
    rot_z = np.asarray([[math.cos(agl[2]), -math.sin(agl[2]), 0], [math.sin(agl[2]), math.cos(agl[2]), 0], [0, 0, 1]])
    return rot_x, rot_y, rot_z

# compute overall bbox from a list of bounding boxes
def overall_bbox(bbox_list):
    coord_dict = {f'c{i}': [] for i in range(6)}
    for bbox in bbox_list:
        for i in range(6):
            coord_dict[f'c{i}'].append(bbox[i//3][i%3])
    return [[min(coord_dict[f'c{i}']) for i in range(3)],
            [max(coord_dict[f'c{i}']) for i in range(3, 6)]]


def align_bbox(bbox_1, bbox_2, rot, center):
    agl = np.radians(rot)
    agl_back = [-a for a in agl]

    bbox_1 = np.array(bbox_1)
    rot_x, rot_y, rot_z = rotation_matrix(agl)
    translation = np.transpose(bbox_1 - center)
    rotated = np.matmul(rot_z, np.matmul(rot_y, np.matmul(rot_x, translation))).transpose() + center
    min_z = np.min(rotated[:, -1])
    print(min_z)
    
    bbox_2 = np.array(bbox_2)
    bbox_2[:, -1] += min_z
    rot_x, rot_y, rot_z = rotation_matrix(agl_back)
    translation = np.transpose(bbox_2 - center)
    rotated = np.matmul(rot_x, np.matmul(rot_y, np.matmul(rot_z, translation))).transpose() + center
    min_z = np.min(rotated[:, -1])
    print(min_z)  # should be approximately 0
    
    rotated[:, -1] -= min_z
    
    bmin = rotated[0] if rotated[0][0] < rotated[1][0] else rotated[1]
    bmax = rotated[1] if rotated[0][0] < rotated[1][0] else rotated[0]
    return [bmin, bmax]

# e.g., oa_bbox.py gcode1 150,150,5 -p gcode2 90,0,0 -p gcode3 0,90,0
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute overall bounding box")
    p.add_argument("orig_gcode", help="G-code for the original orientation")
    p.add_argument("center", help="rotation center in x,y,z")
    p.add_argument("-p", "--pair", action="append", nargs=2, metavar=('G-code', 'rotation'),
                   help="Specify a pair of g-code and its rotation")
    args = p.parse_args()
    
    orig_bbox = parse_bbox(args.orig_gcode)
    bb_list = [orig_bbox]
    for gc, rot in args.pair:
        bbox = parse_bbox(gc)
        rot_bbox = align_bbox(orig_bbox, bbox, triple(rot), triple(args.center))
        bb_list.append(rot_bbox)
    
    sys.exit(overall_bbox(bb_list))
    
    
