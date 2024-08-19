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
import json

def parse_str(arg):
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
    return np.matmul(rot_z, np.matmul(rot_y, rot_x))

def rotate_points(pts, rot, ct, undo):
    agl = np.radians(rot)
    rot_mat = rotation_matrix(agl).transpose() if undo else rotation_matrix(agl)
    translation = np.transpose(pts - np.array(ct))
    rotated = np.matmul(rot_mat, translation).transpose() + ct
    rotated[:, -1] -= np.min(rotated[:, -1])
    return rotated

# compute overall bbox from a list of bounding boxes
def overall_bbox(bbox_list):
    coord_dict = {f'c{i}': [] for i in range(6)}
    for bbox in bbox_list:
        for i in range(6):
            coord_dict[f'c{i}'].append(bbox[i//3][i%3])
    return [[min(coord_dict[f'c{i}']) for i in range(3)],
            [max(coord_dict[f'c{i}']) for i in range(3, 6)]]

def align_bbox(bbox_1, rot_1, center_1, rot_2, center_2):
    undo_rotated = rotate_points(bbox_1, rot_1, center_1, undo=1)
    if rot_2 == [0., 0., 0.]:
        rotated = undo_rotated
    else:
        rotated = rotate_points(undo_rotated, rot_2, center_2, undo=0)

    bmin = rotated[0] if rotated[0][0] < rotated[1][0] else rotated[1]
    bmax = rotated[1] if rotated[0][0] < rotated[1][0] else rotated[0]
    
    return [bmin, bmax]

    
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute overall bounding box")
    p.add_argument("gcode")
    p.add_argument("rot", default="0,0,0", type=parse_str,
                   help="rotation applied on the original model to get sliced")
    p.add_argument("center", type=parse_str, help="xy coordinates of rotation center")
    p.add_argument("--height", type=float, help="provide the height of the original model if gcode is not original")
    p.add_argument("-t", "--triple", action="append", nargs=3,
                   metavar=('G-code', 'rotation', 'height'),
                   help="Specify a triple of g-code, rotation, and height")
    args = p.parse_args()
    
    orig_bbox = parse_bbox(args.gcode)
    bb_list = [orig_bbox]
    
    if args.triple is not None:
        for gc, rot, h in args.triple:
            bbox = parse_bbox(gc)
            center_back = args.center + [float(h) / 2]
            center_fw = None if args.height is None else args.center + [args.height / 2]
            rot_bbox = align_bbox(bbox, parse_str(rot), center_back, 
                                args.rot, center_fw)
            bb_list.append(rot_bbox)
    
    bbox_json = json.dumps(overall_bbox(bb_list), indent=4)
    print(bbox_json)
    
    
