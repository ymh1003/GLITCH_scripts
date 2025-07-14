import open3d as o3d
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
                    description='Read and visualize a point cloud')
parser.add_argument("fin_pcd", help="Path to the point cloud file")
parser.add_argument("--out", default="out.png", help="PNG path for the screenshot")
parser.add_argument("--up", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                    help='Set the up direction of the view')
parser.add_argument("--front", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                    help='Set the front direction of the view')
parser.add_argument("--zoom", type=float, default=1.0,
                    help='Set the zoom level of the view')
args = parser.parse_args()

PCD = o3d.io.read_point_cloud(args.fin_pcd)
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
visualizer.add_geometry(PCD)
view_ctl = visualizer.get_view_control()

view_ctl.set_up   ([float(x) for x in args.up])
view_ctl.set_front([float(x) for x in args.front])
visualizer.capture_screen_image(args.out, do_render=True)
print(f"Screenshot written to {Path(args.out).resolve()}")
visualizer.close()