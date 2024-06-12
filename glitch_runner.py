#!/usr/bin/env python3

import argparse
import yaml
from pathlib import Path
import logging
import tempfile
import sys
import stlinfo
import shutil
import subprocess

#./comp_script.sh [stlfile] [rotx] [roty] [rotz] [sampling-gap] [center-x] [center-y] [boxsize-x] [boxsize-y] [boxsize-z] [percentile]

class SlicerPj3d:
    def __init__(self, printername, printsettings, pj3dbin='pj3d'):
        self.printername = printername
        self.printsettings = Path(printsettings).absolute()
        self.pj3d = pj3dbin

    def run_pj3d(self, *args):
        cmd = [self.pj3d]
        cmd.extend(args)
        print(cmd)
        return subprocess.run(cmd, check=True)

    def generate_gcode(self, model, storage):
        p = storage / model.name

        self.run_pj3d(p, "create", self.printername, str(self.printsettings))
        self.run_pj3d(p, "add", model.path)
        self.run_pj3d(p, "pack")
        self.run_pj3d(p, "printpart", model.path)
        self.run_pj3d(p, "printpart", model.path, "--rotxyz",
                      f"{model.rotation['x']},{model.rotation['y']},{model.rotation['z']}", "--suffix", "_rotated")

        jobpath = p.with_suffix('.job')
        prefix1 = jobpath / Path(model.path).with_suffix('.gcode')
        prefix2 = str(prefix1.with_suffix('')) + "_rotated.gcode"
        return  prefix1, Path(prefix2)

class Model:
    def __init__(self):
        self.rotation = {'x': 0, 'y': 0, 'z': 0}
        self.path = None
        self.scaling = 1
        self.dimensions = (None, None, None)
        self.sampling = None
        self.boxsize = {'length': 1, 'width': 1, 'height': 1}
        self.threshold = None

    def generate_gcode(self, slicer):
        # scale the stl file if needed / TODO: cache?

        # call the slicer

        pass

    def invoke_glitch(self, gcode_orig, gcode_rotated):
        pass


    @staticmethod
    def from_stl(stlfile: Path, name: str):
        x = Model()
        x.name = name
        x.path = str(stlfile)
        sf = stlinfo.STLFile()
        sf.load_file(stlfile)
        bmin, bmax = sf.bounds()
        x.dimensions = (float(bmax.x - bmin.x + 1),
                        float(bmax.y - bmin.y + 1),
                        float(bmax.z - bmin.z + 1))

        return x

    @staticmethod
    def load(d):
        x = Model()
        x.name = d['name']
        x.path = d['path']
        x.rotation = d.get('rotation', x.rotation)
        x.scaling = d.get('scaling', x.scaling)
        x.dimensions = tuple(d.get('dimensions', x.dimensions))
        x.sampling = d.get('sampling', x.sampling)
        x.boxsize = d.get('boxsize', x.boxsize)
        x.threshold = d.get('threshold', x.threshold)

        return x

    def to_dict(self):
        return {'name': self.name,
                'path': self.path,
                'rotation': self.rotation,
                'dimensions': list(self.dimensions),
                'scaling': self.scaling,
                'sampling': self.sampling,
                'boxsize': self.boxsize,
                'threshold': self.threshold}

class GlitchExpt:
    """A class for representing information for Glitch Experiments"""
    def __init__(self):
        self.models = [] # maybe dictionary or orderdict?

    def add_model(self, model):
        mnames = set([m.name for m in self.models])

        if model.name in mnames:
            raise KeyError(f"Duplicate name {model.name}")

        self.models.append(model)

    def get_model(self, model):
        for m in self.models:
            if m.name == model:
                return m

        raise KeyError

    @staticmethod
    def from_dict(d):
        assert d["format"] == "glitch-expt"
        assert d["version"] == 1
        x = GlitchExpt()
        x.models = list([Model.load(m) for m in d["models"]])
        return x

    def to_dict(self):
        return {"format": "glitch-expt",
                "version": 1,
                "models": list([m.to_dict() for m in self.models])}

def load_yaml(yamlfile):
    with open(yamlfile, "r") as f:
        expt = yaml.safe_load(f)
        return GlitchExpt.from_dict(expt)

def save_yaml(expt, yamlfile):
    with open(yamlfile, "w") as f:
        yaml.dump(expt.to_dict(), f)

def do_create(args):
    ey = Path(args.exptyaml)

    if ey.exists():
        print(f"ERROR: {ey} already exists. Not overwriting")
        return 1

    ge = GlitchExpt()
    save_yaml(ge, ey)
    return 0

def parse_csnum(s, ty, k, default=None):
    v = s.split(",")

    if len(v) > len(k) or (len(v) < len(k) and default is None):
        # improper list of values
        raise ValueError

    v = [ty(vv) if vv else default for vv in v]

    if default is None and default in v:
        # list contains empty element and no default supplied
        raise ValueError

    return dict(zip(k, v))

def do_add(args):
    ge = load_yaml(args.exptyaml)

    stlfile = Path(args.stlfile)

    if not stlfile.exists():
        print(f"ERROR: {stlfile} does not exist.")
        return 1

    n = args.name or stlfile.with_suffix('').name
    m = Model.from_stl(stlfile, n)

    m.sampling = args.sampling
    m.threshold = args.threshold
    try:
        m.boxsize = parse_csnum(args.boxsize, float, ('length',
                                                      'width',
                                                      'height'),
                                0)
    except ValueError:
        print(f"ERROR: {args.boxsize} is improperly formatted")
        return 1

    try:
        print(f"Adding model with name={m.name} to {args.exptyaml}")
        ge.add_model(m)
    except KeyError:
        print(f"ERROR: Model with name={m.name} already exists, use -n to change name if adding a new model")
        return 1


    save_yaml(ge, args.exptyaml)
    return 0

def do_runone(args):
    ge = load_yaml(args.exptyaml)
    m = ge.get_model(args.name)

    sl = SlicerPj3d(args.printer, args.printsettings, args.pj3d)

    with tempfile.TemporaryDirectory(prefix="glitch") as d:
        print(d)
        gcode_orig, gcode_rotated = sl.generate_gcode(m, Path(d))
        print(gcode_orig, gcode_rotated)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run glitch on models specified in file")

    p.add_argument("exptyaml")

    sp = p.add_subparsers(dest="cmd")

    cm = sp.add_parser('create', help="Create an experiment file")

    am = sp.add_parser('add', help="Add a model")
    am.add_argument("stlfile", help="path to STL file")
    am.add_argument("--rot", help="Glitch Rotation as X,Y,Z in degrees")
    am.add_argument("-s", "--sampling", help="Sampling interval", type=float, default=0.1)
    am.add_argument("-t", "--threshold", help="Threshold, in percentile", type=float, default=90)
    am.add_argument("-b", "--boxsize", help="Box/cube size, for visualization", default="1,1,1") # Is this also used for HD?
    am.add_argument("-n", "--name", help="Unique name for model, by default just the part before .stl")

    gm = sp.add_parser('runone', help="Run Glitch on a single model")
    gm.add_argument("name", help="Name of model")
    gm.add_argument("printer", help="Printer name")
    gm.add_argument("printsettings", help="Printer settings")
    gm.add_argument("--pj3d", help="Path to pj3d binary", default=shutil.which('pj3d') or 'pj3d')

    args = p.parse_args()

    if args.cmd == "create":
        sys.exit(do_create(args))
    elif args.cmd == "add":
        sys.exit(do_add(args))
    elif args.cmd == "runone":
        sys.exit(do_runone(args))
