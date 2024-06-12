#!/usr/bin/env python3

import argparse
import yaml
import path
import logging
import tempfile

#./comp_script.sh [stlfile] [rotx] [roty] [rotz] [sampling-gap] [center-x] [center-y] [boxsize-x] [boxsize-y] [boxsize-z] [percentile]

class SlicerPj3d:
    def generate_gcode(self, stlfile, rotation):
        # pj3d projectname create "Printer Name" "Print Settings"
        # pj3d add stlfile
        # pj3d printpart stlfile
        # pj3d printpart stlfile --rotation --suffix
        # return the two gcode files


class Model:
    def __init__(self):
        self.rotation = {'x': 0, 'y': 0, 'z': 0}
        self.scaling = 1
        self.sampling = None
        self.boxsize = None # TODO: Is this x,y,z?
        self.threshold = None

    def generate_gcode(self, slicer):
        # scale the stl file if needed / TODO: cache?

        # call the slicer

        pass

    def invoke_glitch(self, gcode_orig, gcode_rotated):
        pass


    @staticmethod
    def load(d):
        x = Model()
        x = d['name']
        x.rotation = d.get('rotation', x.rotation)
        x.scaling = d.get('scaling', x.scaling)
        x.sampling = d.get('sampling', x.sampling)
        x.boxsize = d.get('boxsize', x.boxsize)
        x.threshold = d.get('threshold', x.threshold)

        return x

    def as_dict(self):
        return {'name': self.name,
                'rotation': self.rotation,
                'scaling': self.scaling,
                'sampling': self.sampling,
                'cubesize': self.cubesize,
                'threshold': self.threshold}

def load_yaml(yamlfile):
    out = []
    with open(yamlfile, "r") as f:
        models = yaml.safe_load(f)
        for m in models:
            out.append(Model.load(m))

    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run glitch on models specified in file")

    p.add_argument("models.yaml")

    args = p.parse_args()
