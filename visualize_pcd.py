#!/usr/bin/env python3

import open3d as o3d
import argparse
from pathlib import Path
import json
import sys
import os

parser = argparse.ArgumentParser(
                    description='Read and visualize a point cloud')
parser.add_argument("fin_pcd", help="Path to the point cloud file")
parser.add_argument("--out", help="PNG path for the screenshot")
parser.add_argument("--up", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                    help='Set the up direction of the view')
parser.add_argument("--front", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                    help='Set the front direction of the view')
parser.add_argument("--zoom", type=float, default=None,
                    help='Set the zoom level of the view')
parser.add_argument("--ni", dest="non_interactive", action="store_true", help="Use in non-interactive mode")

args = parser.parse_args()

PCD = o3d.io.read_point_cloud(args.fin_pcd)
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
visualizer.add_geometry(PCD)
view_ctl = visualizer.get_view_control()

if view_ctl is None:
    print("ERROR: Couldn't get view control. Failed to render.", file=sys.stderr)
    print("If you're using Wayland, setting XDG_SESSION_TYPE=x11 for this command might help", file=sys.stderr)
    if 'DISPLAY' not in os.environ:
        print("DISPLAY variable is not set.", file=sys.stderr)

    sys.exit(1)

view_ctl.set_up   ([float(x) for x in args.up])
view_ctl.set_front([float(x) for x in args.front])

if args.zoom: view_ctl.set_zoom(args.zoom)

if args.out:
    visualizer.capture_screen_image(args.out, do_render=True)
    print(f"Screenshot written to {Path(args.out).resolve()}")

if args.non_interactive:
    visualizer.close()
else:
    visualizer.run()
    st = json.loads(visualizer.get_view_status())
    print(st)
    print("--front", " ".join([str(x) for x in st['trajectory'][0]['front']]), end=' ')
    print("--up", " ".join([str(x) for x in st['trajectory'][0]['up']]), end=' ')
    print("--zoom", str(st['trajectory'][0]['zoom']))

