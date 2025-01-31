import open3d as o3d
import argparse

parser = argparse.ArgumentParser(
                    description='Read and visualize a point cloud')
parser.add_argument('filepath')
args = parser.parse_args()

PCD = o3d.io.read_point_cloud(args.filepath)
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
visualizer.add_geometry(PCD)
view_ctl = visualizer.get_view_control()
view_ctl.set_up((0, 0, 1))  # set the positive direction of the z-axis as the up direction
view_ctl.set_front((1, 0, 0))  # set the positive direction of the x-axis toward you
visualizer.run()