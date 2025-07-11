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
import sys
#sys.path.append('../Gcode-Checking-Project/')
from gcode_comp_Z import initialize_global, align_bbox, overall_bbox


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
    return np.array([[minx, miny, minz], [maxx, maxy, maxz]])

    
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute overall bounding box")
    p.add_argument("gcode")
    p.add_argument("imd_rot", default="0,0,0", type=parse_str,
                   help="rotation applied on the original model to get sliced")
    p.add_argument("-p", "--pair", action="append", nargs=2,
                   metavar=('G-code', 'rotation'),
                   help="Specify a pair of g-code and rotation")
    args = p.parse_args()
    
    initialize_global() # use float by default
    
    orig_bbox = parse_bbox(args.gcode)
    bb_list = [orig_bbox]
    
    if args.pair is not None:
        for gc, rot in args.pair:
            bbox = parse_bbox(gc)
            rot_bbox = align_bbox(bbox, args.imd_rot, parse_str(rot))
            bb_list.append(rot_bbox)
    
    bbox_json = json.dumps(overall_bbox(bb_list), indent=4)
    print(bbox_json)
    
    
