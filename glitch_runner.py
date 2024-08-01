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
from collections import namedtuple
import re
import logging
import shlex

logger = logging.getLogger()
XYZTuple = namedtuple('XYZTuple', 'x y z')

class SlicerPj3d:
    MACHINE_DIM = re.compile(r'^machine_(depth|height|width)="([0-9\.]+)"$')

    def __init__(self, printername, printsettings, pj3dbin='pj3d'):
        self.printername = printername
        self.printsettings = Path(printsettings).absolute()
        self.pj3d = pj3dbin
        self._load_printer_dimensions()

    def _load_printer_dimensions(self):
        out = {}
        with open(self.printsettings, "r") as f:
            for l in f:
                m = SlicerPj3d.MACHINE_DIM.match(l)
                if m is not None:
                    out[m.group(1)] = m.group(2)

        assert len(out) == 3, "Unable to find one dimension. Found {out.keys()}"
        #TODO: check the mapping of X, Y, Z
        self.dimensions = XYZTuple(x=float(out['width']),
                                   y=float(out['depth']),
                                   z=float(out['height']))

        logger.info('Printer dimensions: {self.dimensions}')

    def run_pj3d(self, *args):
        cmd = [self.pj3d]
        cmd.extend(args)
        logger.info(shlex.join([str(s) for s in cmd]))
        return subprocess.run(cmd, check=True)

    def generate_gcode(self, model, storage):
        p = storage / model.name

        self.run_pj3d(p, "create", self.printername, str(self.printsettings))
        self.run_pj3d(p, "add", model.path)
        self.run_pj3d(p, "pack")
        self.run_pj3d(p, "printpart", model.path)

        jobpath = p.with_suffix('.job')
        prefix1 = jobpath / Path(model.path).with_suffix('.gcode').name

        # in preparation for when we will do multiple rotations
        out = []

        # backward compat
        if isinstance(model.rotation, dict):
            rot = [model.rotation]
        else:
            rot = model.rotation

        for r in rot:
            rotation = XYZTuple(x = r['x'],
                                y = r['y'],
                                z = r['z'])

            suffix = f"__x{rotation.x},y{rotation.y},z{rotation.z}__".replace(".", "_")
            self.run_pj3d(p, "printpart", model.path, "--rotxyz",
                          f"{rotation.x},{rotation.y},{rotation.z}",
                          "--suffix", f"{suffix}_rotated")

            prefix2 = str(prefix1.with_suffix('')) + f"{suffix}_rotated.gcode"
            out.append((rotation, prefix2))

        return  prefix1, out

class Model:
    def __init__(self):
        self.rotation = [{'x': 0, 'y': 0, 'z': 0}]
        self.path = None
        self.scaling = 1
        self.dimensions = (None, None, None)
        self.sampling = None
        self.boxsize = {'length': 1, 'width': 1, 'height': 1}
        self.threshold = None
        self.density = None

    def generate_gcode(self, slicer, storage):
        gcode_orig, gcode_rotated = slicer.generate_gcode(self, storage)
        return gcode_orig, gcode_rotated

    def run_glitch(self,
                   gcode_orig,
                   gcode_rotated_file,
                   printer_dims,
                   rotation,
                   use_float = True,
                   glitch = 'gcode_comp_Z.py',
                   dry_run = False
                   ):

        center_x, center_y = printer_dims.x / 2, printer_dims.y / 2

        dim = XYZTuple(*self.dimensions)
        height = dim.z / 2

        boxlwh = [self.boxsize["length"], self.boxsize["width"], self.boxsize["height"]]

        cmd = [glitch, "-c", str(center_x), str(center_y),
               "-s", str(self.sampling),
               "-b"] + [str(s) for s in boxlwh] + \
               ["-t", str(self.threshold),
                "-m", str(self.density),
                "-d", str(0 if use_float else 1),
                str(gcode_orig),
                str(gcode_rotated_file),
                str(rotation.x),
                str(rotation.y),
                str(rotation.z),
                str(height)]

        print(shlex.join(cmd))
        if not dry_run: return subprocess.run(cmd, check=True)
        return None

    def invoke_glitch(self, slicer, gcode_orig, gcode_rotated,
                      glitch='gcode_comp_Z.py',
                      dry_run = False):
        for (rot, gcode) in gcode_rotated:
            self.run_glitch(gcode_orig, gcode,
                            slicer.dimensions, rot,
                            glitch = glitch,
                            dry_run = dry_run)
            print(gcode_orig, rot, gcode)

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
        x.density = d.get('density', x.density)

        return x

    def to_dict(self):
        return {'name': self.name,
                'path': self.path,
                'rotation': self.rotation,
                'dimensions': list(self.dimensions),
                'scaling': self.scaling,
                'sampling': self.sampling,
                'boxsize': self.boxsize,
                'threshold': self.threshold,
                'density': self.density}

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
    m.density = args.density

    try:
        m.rotation = [parse_csnum(rot, float, ('x','y','z'), 0) for rot in args.rot]
    except ValueError:
        print(f"ERROR: {args.rot} is improperly formatted")
        return 1

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

    sl = SlicerPj3d(args.printer, args.printsettings, pj3dbin = args.pj3d)

    with tempfile.TemporaryDirectory(prefix="glitch") as d:
        print(d)
        gcode_orig, gcode_rotated = m.generate_gcode(sl, Path(d))
        print(gcode_orig, gcode_rotated)
        m.invoke_glitch(sl, gcode_orig, gcode_rotated, glitch = args.glitch, dry_run=args.dryrun)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run glitch on models specified in file")

    p.add_argument("exptyaml")

    sp = p.add_subparsers(dest="cmd")

    cm = sp.add_parser('create', help="Create an experiment file")

    am = sp.add_parser('add', help="Add a model")
    am.add_argument("stlfile", help="path to STL file")
    am.add_argument("--rot", help="Glitch Rotation as X,Y,Z in degrees",
                    action="append", default=[])
    am.add_argument("-s", "--sampling", help="Sampling interval", type=float, default=0.1)
    am.add_argument("-t", "--threshold", help="Threshold, in percentile", type=float, default=90)
    am.add_argument("-d", "--density", help="Density for each box, in float", type=float, default=0.3)
    am.add_argument("-b", "--boxsize", help="Box/cube size, for visualization", default="1,1,1") # Is this also used for HD?
    am.add_argument("-n", "--name", help="Unique name for model, by default just the part before .stl")

    gm = sp.add_parser('runone', help="Run Glitch on a single model")
    gm.add_argument("name", help="Name of model")
    gm.add_argument("printer", help="Printer name")
    gm.add_argument("printsettings", help="Printer settings")
    gm.add_argument("--pj3d", help="Path to pj3d binary", default=shutil.which('pj3d') or 'pj3d')
    gm.add_argument("--glitch", help="Path to glitch", default=shutil.which('gcode_comp_Z.py') or 'gcode_comp_Z.py')
    gm.add_argument("-n", dest="dryrun", help="Dry-run, don't actually run glitch", action="store_true")

    args = p.parse_args()

    if args.cmd == "create":
        sys.exit(do_create(args))
    elif args.cmd == "add":
        sys.exit(do_add(args))
    elif args.cmd == "runone":
        sys.exit(do_runone(args))
