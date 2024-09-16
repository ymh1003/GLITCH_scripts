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
import json
import configparser
import platform
import os

logger = logging.getLogger()
XYZTuple = namedtuple('XYZTuple', 'x y z')

def get_config_dir():
    path = None
    if platform.system() == 'Windows':
        path = os.environ.get('APPDATA', '') or '~/AppData/Roaming'
    else:
        path = os.environ.get('XDG_CONFIG_HOME', '') or '~/.config'

    return Path(os.path.abspath(os.path.expanduser(path)))

class Config:
    def __init__(self, path = None):
        if path is None:
            path = Config.default_path()

        self.configfile = path / 'glitch_runner.cfg'
        self.config = configparser.ConfigParser()
        if self.configfile.exists():
            logger.info(f"Loaded configuration file: {self.configfile}")
            self.config.read(self.configfile)

        if not self.config.has_section("paths"):
            self.config.add_section("paths")

    @staticmethod
    def default_path():
        return get_config_dir() / 'glitch_runner'

    def write_config(self):
        if not self.configfile.parent.exists():
            os.makedirs(self.configfile.parent)

        with open(self.configfile, "w") as f:
            self.config.write(f)

    def get_path(self, key, default=None):
        return self.config.get("paths", key, fallback=default)

    def set_path(self, key, value):
        self.config.set("paths", key, value)


class Paths:
    pj3d = 'pj3d'
    glitch = 'gcode_comp_Z.py'
    oa_bbox = 'oa_bbox.py'
    stlrotate = 'stlrotate'
    stlscale = 'stlscale'
    heatmap_merge = 'heatmap_merge.py'

    # these don't work since unqualified paths must also be in PATH
    def check(self, to_check = ['pj3d', 'stlrotate', 'stlscale',
                                'glitch', 'oa_bbox', 'heatmap_merge']):
        for i in to_check:
            p = Path(getattr(self, i))

            if not p.exists():
                logger.error(f"Path '{p}' does not exist for '{i}'. Use config to set a path or supply it as an argument.")
                return False

        return True

    def check_gcode(self):
        return self.check(['pj3d', 'stlrotate', 'stlscale'])

    def check_glitch(self):
        return self.check(['glitch', 'oa_bbox', 'heatmap_merge'])


def get_paths(args, config):
    p = Paths()

    p.pj3d = (args.pj3d if hasattr(args, 'pj3d') else None) or config.get_path('pj3d') or p.pj3d
    p.glitch = (args.glitch if hasattr(args, 'glitch') else None) or config.get_path('glitch') or p.glitch
    p.oa_bbox = (args.oabbox if hasattr(args, 'oabbox') else None) or  config.get_path('oa_bbox') or p.oa_bbox
    p.stlrotate = (args.stlrotate if hasattr(args, 'stlrotate') else None) or config.get_path('stlrotate') or p.stlrotate
    p.stlscale = (args.stlscale if hasattr(args, 'stlscale') else None) or config.get_path('stlscale') or p.stlscale
    p.heatmap_merge = (args.hmerge if hasattr(args, 'hmerge') else None) or config.get_path('heatmap_merge') or p.heatmap_merge

    return p

def xyz2str(xyzt):
    if isinstance(xyzt, dict):
        return f"{xyzt['x']},{xyzt['y']},{xyzt['z']}"
    elif isinstance(xyzt, XYZTuple):
        return f"{xyzt.x},{xyzt.y},{xyzt.z}"
    else:
        raise NotImplementedError

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

        logger.info(f'Printer dimensions: {self.dimensions}')

    def run_pj3d(self, *args):
        cmd = [self.pj3d]
        cmd.extend(args)
        logger.info(shlex.join([str(s) for s in cmd]))
        return subprocess.run(cmd, capture_output=True,
                              encoding='utf-8', check=True)

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

            prefix2 = Path(str(prefix1.with_suffix('')) + f"{suffix}_rotated.gcode")
            out.append((rotation, prefix2))

        return  prefix1, out

    def generate_gcode2(self, model, storage):
        assert hasattr(model, 'rotated_stl')

        p = storage / model.name

        self.run_pj3d(p, "create", self.printername, str(self.printsettings))
        for m in model.rotated_stl: # this contains the original as well
            self.run_pj3d(p, "add", m)

        self.run_pj3d(p, "pack")

        jobpath = p.with_suffix('.job')
        prefix1 = jobpath / Path(model.path).with_suffix('.gcode').name

        out = []

        assert isinstance(model.rotation, list)

        for r, f in zip(model.rotation, model.rotated_stl):
            rotation = XYZTuple(x = r['x'],
                                y = r['y'],
                                z = r['z'])

            self.run_pj3d(p, "printpart", f, "--rotxyz",
                          f"{rotation.x},{rotation.y},{rotation.z}",
                          #"--suffix", f"{suffix}_rotated" (inherits suffix from rotated stl)
                          )

            prefix2 = jobpath / Path(f).with_suffix('.gcode').name
            out.append((rotation, prefix2))

        return out[0][1], out[1:]

class Model:
    def __init__(self):
        self.rotation = [{'x': 0, 'y': 0, 'z': 0}]
        self.path = None
        self.path_scaled = None
        self.scaling = 1
        self.dimensions = (None, None, None)
        self.sampling = None
        self.boxsize = {'length': 1, 'width': 1, 'height': 1}
        self.threshold = None
        self.density = None

    def do_scale(self, slicer, storage, stlscale = 'stlscale'):
        logger.debug(f"Ignoring scaling value {self.scaling} in Yaml file for {self.name}")

        # compute the scale by running
        volume = "%d,%d,%d" % (slicer.dimensions.x,
                               slicer.dimensions.y,
                               slicer.dimensions.z)

        cmd = [stlscale, str(self.path), 'compute', '-v', volume]
        logger.info(f"Invoking stlscale: {shlex.join(cmd)}")
        output = subprocess.check_output(cmd, encoding='utf-8')
        logger.info(f"stlscale returned {output} for {self.name}")
        output = output.strip()

        if output == "1":
            logger.info(f"Not scaling model {self.name}")
            return
        else:
            self.scaling = float(output)
            self.path_scaled = storage / Path(self.path).with_suffix(".scaled.stl").name
            if self.path_scaled.exists():
                logger.info(f"Already found scaled model {self.path_scaled}, not rescaling")
            else:
                #TODO: will we need multiple scaled versions?
                logger.info(f"Scaling model '{self.name}' by '{self.scaling}' to '{self.path_scaled}'")

                subprocess.run([stlscale,
                                str(self.path),
                                'scale',
                                '-o', self.path_scaled,
                                str(self.scaling)], encoding='utf-8',
                               check=True)


    def load_heights(self, stlinfo = 'stlinfo'):
        assert hasattr(self, 'rotated_stl')
        rm = []
        for p in self.rotated_stl:
            rm.append(Model.from_stl(p, self.name))

        self.rotated_model = rm

    def get_scaled_model(self):
        if self.scaling == 1:
            return self
        else:
            # this isn't there on the scaled model
            assert self.path_scaled is not None

            x = Model()
            x.name = self.name
            x.rotation = self.rotation
            x.scaling = self.scaling
            x.dimensions = self.dimensions
            x.sampling = self.sampling
            x.boxsize = self.boxsize
            x.threshold = self.threshold
            x.density = self.density
            x.path = self.path_scaled

            return x

    def generate_rotations(self, storage, stlrotate='stlrotate'):
        m = self.get_scaled_model()
        rotated_stls = []

        for rot in m.rotation:
            rotation = XYZTuple(**rot)
            if rotation.x == rotation.y and rotation.y == rotation.z and rotation.x == 0:
                logging.info("Not generating rotated version")
                rotated_stls.append(m.path)
                continue

            suffix = f"__x{rotation.x},y{rotation.y},z{rotation.z}__".replace(".", "_")

            op = storage / f"{self.name}_{suffix}.stl"
            logging.info(f"Rotating {m.path} by {rot} and storing to {op}")
            cmd = [stlrotate, str(m.path), str(op),
                   str(rotation.x), str(rotation.y), str(rotation.z)]
            logging.info(f"Running {shlex.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(e)
                logging.info("stlrotate: " + e.stdout)
                return False

            rotated_stls.append(op)

        m.rotated_stl = rotated_stls
        return m

    def generate_gcode2(self, slicer, storage, stlrotate):
        """For updated glitch"""

        m = self.generate_rotations(storage, stlrotate = stlrotate)
        if not m:
            return None, None, None

        try:
            gcode_orig, gcode_rotated = slicer.generate_gcode2(m, storage)
        except subprocess.CalledProcessError as e:
            logging.error(e)
            logging.info("pj3d:" + e.stdout)

            return None, None, None

        return gcode_orig, gcode_rotated, m

    def generate_gcode(self, slicer, storage):
        try:
            gcode_orig, gcode_rotated = slicer.generate_gcode(self.get_scaled_model(), storage)
        except subprocess.CalledProcessError as e:
            logging.error(e)
            logging.info("pj3d:" + e.stdout)

            return None, None

        return gcode_orig, gcode_rotated

    def run_glitch(self,
                   gcode_orig,
                   gcode_rotated_file,
                   printer_dims,
                   rotation,
                   use_float = True,
                   output_dir = None,
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

        if output_dir is not None:
            json_name = f"{gcode_orig.stem}-{gcode_rotated_file.stem}.json"
            cmd += ["--collect",
                    str(output_dir / json_name)]

        logger.info(f"Running glitch: {shlex.join(cmd)}")
        if not dry_run: return subprocess.run(cmd, check=True)
        return None

    def run_glitch2(self,
                    gcode_orig,
                    gcode_rotated_file,
                    printer_dims,
                    rotation,
                    oa_bbox,
                    use_float = True,
                    output_dir = None,
                    glitch = 'gcode_comp_Z.py',
                    dry_run = False
                    ):

        center_x, center_y = printer_dims.x / 2, printer_dims.y / 2

        dim = XYZTuple(*self.rotated_model[0].dimensions)
        height = dim.z / 2

        boxlwh = [self.boxsize["length"], self.boxsize["width"], self.boxsize["height"]]

        cmd = [glitch, "-c", str(center_x), str(center_y),
               "-s", str(self.sampling),
               "-b"] + [str(s) for s in boxlwh] + \
               ["-t", str(self.threshold),
                "-m", str(self.density),
                "-d", str(0 if use_float else 1),
                "--bound",
                ",".join([str(s) for s in oa_bbox['bbminxyz']]),
                ",".join([str(s) for s in oa_bbox['bbmaxxyz']]),
                str(gcode_orig),
                xyz2str(self.rotation[0]),
                str(gcode_rotated_file),
                xyz2str(rotation),
                str(height)]

        if output_dir is not None:
            json_name = f"{gcode_orig.stem}-{gcode_rotated_file.stem}.json"
            cmd += ["--collect",
                    str(output_dir / json_name)]

        logger.info(f"Running glitch: {shlex.join(cmd)}")
        if not dry_run: return output_dir / json_name, subprocess.run(cmd, check=True)
        return None, None


    def invoke_glitch(self, slicer, gcode_orig, gcode_rotated,
                      output_dir = None,
                      glitch='gcode_comp_Z.py',
                      dry_run = False):

        #TODO: obtain overall bounding box using tool for ...

        for (rot, gcode) in gcode_rotated:
            logger.debug(f"Running glitch for rotation={rot} with rotated gcode='{gcode}' (original: {gcode_orig})")
            self.run_glitch(gcode_orig, gcode,
                            slicer.dimensions, rot,
                            output_dir = output_dir,
                            glitch = glitch,
                            dry_run = dry_run)

    def invoke_glitch2(self, printerdim, gcode_orig, gcode_rotated,
                       oabbox,
                       output_dir = None,
                       glitch='gcode_comp_Z.py',
                       dry_run = False):

        json_files = []
        for (rot, gcode), m in zip(gcode_rotated, self.rotated_model[1:]):
            logger.debug(f"Running glitch for rotation={rot} with rotated gcode='{gcode}' (original: {gcode_orig})")
            out, res = self.run_glitch2(gcode_orig, gcode,
                                        printerdim, rot,
                                        oabbox,
                                        output_dir = output_dir,
                                        glitch = glitch,
                                        dry_run = dry_run)
            if dry_run: continue
            if not res: return None
            json_files.append(out)

        return json_files

    def invoke_heatmap_merge(self, json_files,
                             heatmap_merge='heatmap_merge.py',
                             dry_run = False):

        cmd = [heatmap_merge] + [str(s) for s in json_files]
        logger.info(f"Running heatmap_merge: {shlex.join(cmd)}")

        if not dry_run:
            return subprocess.run(cmd, check=True)

        return None

    def get_oabbox(self, printer_dims,
                   gcode_orig, gcode_rotated,
                   oa_bbox = 'oa_bbox.py'):
        center_x, center_y = printer_dims.x / 2, printer_dims.y / 2
        assert hasattr(self, 'rotated_model')

        dim = XYZTuple(*self.rotated_model[0].dimensions)
        height = dim.z / 2

        cmd = [oa_bbox,
               str(gcode_orig),
               xyz2str(self.rotation[0]),
               "--height",
               str(height),
               f"{center_x},{center_y}"]

        for (rot, gcode), rm in  zip(gcode_rotated, self.rotated_model[1:]):
            height = rm.dimensions[2] / 2
            cmd.extend(['-t', str(gcode),
                        ",".join([str(s) for s in [rot.x, rot.y, rot.z]]),
                        str(height)
                        ])


        logger.info(shlex.join(cmd))

        output = subprocess.check_output(cmd, encoding='utf-8')
        logger.info(f"oa_bbox output: {output}")
        bounds = json.loads(output) # returned as two arrays
        assert len(bounds) == 2, len(bounds)
        assert len(bounds[0]) == len(bounds[1]) == 3

        return {'bbminxyz': bounds[0],
                'bbmaxxyz': bounds[1]}


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

    def override(self, sampling, boxsize, threshold, density):
        if all([x is None for x in [sampling, boxsize, threshold, density]]):
            return self

        x = Model()

        x.name = self.name
        x.rotation = [dict(k) for k in self.rotation]
        x.scaling = self.scaling
        x.dimensions = self.dimensions
        x.sampling = self.sampling if sampling is None else sampling
        x.boxsize = self.boxsize if boxsize is None else boxsize
        x.threshold = self.threshold if threshold is None else threshold
        x.density = self.density if density is None else density
        x.path = self.path

        ov = []
        if x.sampling != self.sampling: ov.append(f"s={x.sampling}")
        if x.boxsize != self.boxsize:
            sboxsize = ",".join([f"{k}_{v}" for k, v in x.boxsize.items()])
            ov.append(f"b={sboxsize}")
        if x.threshold != self.threshold: ov.append(f"t={x.threshold}")
        if x.density != self.density: ov.append(f"m={x.density}")
        x._override = ov

        return x

    @property
    def has_override(self):
        return hasattr(self, '_override')

    @property
    def override_suffix(self):
        if not self.has_override: return ""
        return "+".join(self._override)

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

    def rm_model(self, model):
        self.models.remove(model)

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
    logger.info(f"Loading Glitch experiment yaml file: {yamlfile}")

    with open(yamlfile, "r") as f:
        expt = yaml.safe_load(f)
        return GlitchExpt.from_dict(expt)

def save_yaml(expt, yamlfile):
    with open(yamlfile, "w") as f:
        yaml.dump(expt.to_dict(), f)

def do_create(args):
    ey = Path(args.exptyaml)

    if ey.exists():
        print(f"ERROR: {ey} already exists. Not overwriting",
              file=sys.stderr)
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
        print(f"ERROR: {stlfile} does not exist.", file=sys.stderr)
        return 1

    n = args.name or stlfile.with_suffix('').name
    m = Model.from_stl(stlfile, n)

    m.sampling = args.sampling
    m.threshold = args.threshold
    m.density = args.density

    try:
        m.rotation = [parse_csnum(rot, float, ('x','y','z'), 0) for rot in args.rot]
    except ValueError:
        print(f"ERROR: {args.rot} is improperly formatted", file=sys.stderr)
        return 1

    try:
        m.boxsize = parse_csnum(args.boxsize, float, ('length',
                                                      'width',
                                                      'height'),
                                0)
    except ValueError:
        print(f"ERROR: {args.boxsize} is improperly formatted", file=sys.stderr)
        return 1

    try:
        print(f"Adding model with name={m.name} to {args.exptyaml}",
              file=sys.stderr)
        ge.add_model(m)
    except KeyError:
        print(f"ERROR: Model with name={m.name} already exists, use -n to change name if adding a new model", file=sys.stderr)
        return 1


    save_yaml(ge, args.exptyaml)
    return 0

def do_rm(args):
    ge = load_yaml(args.exptyaml)
    try:
        m = ge.get_model(args.name)
        ge.rm_model(m)
        print(f"Removed model '{args.name}'", file=sys.stderr)
    except KeyError:
        print(f"ERROR: Model with name={args.name} does not exist",
              file=sys.stderr)
        return 1

    save_yaml(ge, args.exptyaml)
    return 0

def do_runone(args):
    opdir = Path(args.outputdir)
    if opdir.exists():
        print(f"ERROR: Output directory {args.outputdir} exists.",
              file=sys.stderr)
        return 1

    ge = load_yaml(args.exptyaml)
    m = ge.get_model(args.name)

    sl = SlicerPj3d(args.printer, args.printsettings, pj3dbin = args.pj3d)

    if not args.dryrun:
        opdir.mkdir()

    with tempfile.TemporaryDirectory(prefix="glitch") as d:
        logger.info(f"Using temporary directory: {d}")
        gcode_orig, gcode_rotated = m.generate_gcode(sl, Path(d))

        if gcode_orig is None:
            logging.info("Gcode generation failed.")
            return 1
        else:
            m.invoke_glitch(sl, gcode_orig, gcode_rotated,
                            output_dir = opdir,
                            glitch = args.glitch, dry_run=args.dryrun)

    return 0

def do_gcode(args):
    edir = Path(args.exptdir)
    if edir.exists():
        print(f"ERROR: Experiment output directory {args.exptdir} already exists. Remove it if a previous gcode failed.",
              file=sys.stderr)
        return 1

    cfg = Config()
    paths = get_paths(args, cfg)
    ge = load_yaml(args.exptyaml)
    sl = SlicerPj3d(args.printer, args.printsettings, pj3dbin = paths.pj3d)

    models = list([m.name for m in ge.models])
    if args.models:
        include = set(args.models)
        wrong = include - set(models)
        if len(wrong):
            print(f"ERROR: Model(s) {wrong} do not exist", file=sys.stderr)
            return 1
        models = list(args.models)
        if len(include) != len(models):
            #TODO: identify duplicate model
            print(f"ERROR: Duplicate model names in list", file=sys.stderr)
            return 1

    logger.info(f"Generating gcode for {models}")

    if not args.dryrun:
        edir.mkdir()

    logger.info(f"Storing gcode in {edir}")

    out = {'slicer':
           {'printer': sl.printername,
            'settings': str(sl.printsettings),
            'dimensions': sl.dimensions._asdict()}}

    modelinfo = {}
    for mn in models:
        m = ge.get_model(mn)

        logger.info(f"Generating gcode for {m.name}")
        if args.dryrun: continue

        model_edir = edir / m.name
        model_edir.mkdir()

        m.do_scale(sl, Path(model_edir), stlscale=paths.stlscale)
        gcode_orig, gcode_rotated, xm = m.generate_gcode2(sl,
                                                          Path(model_edir),
                                                          stlrotate =
                                                          paths.stlrotate)
        if gcode_orig is None:
            logger.error(f"Gcode generation failed for {m.name}.")
            return 1

        modelinfo[m.name] = {'original':
                             {'rotation': m.rotation[0],
                              'path': str(gcode_orig),
                              'stlpath': str(xm.rotated_stl[0])},
                             'rotated': [{'rotation': r._asdict(),
                                          'path': str(p)} for (r, p) in gcode_rotated]}

        for re, sp in zip(modelinfo[m.name]['rotated'], xm.rotated_stl[1:]):
            re['stlpath'] = str(sp)

    out['models'] = modelinfo

    with open(edir / "gcodes.json", "w") as f:
        json.dump(out, fp=f, indent='  ')

    return 0


def do_glitch(args):
    edir = Path(args.exptdir)
    if not edir.exists():
        print(f"ERROR: Experiment directory {args.exptdir} does not exist.",
              file=sys.stderr)
        return 1

    cfg = Config()
    paths = get_paths(args, cfg)
    ge = load_yaml(args.exptyaml)

    with open(edir / "gcodes.json", "r") as f:
        je = json.load(f)

    models = list(je['models'].keys())
    printerdim = XYZTuple(**je['slicer']['dimensions'])

    if args.models:
        include = set(args.models)
        wrong = include - set(models)
        if len(wrong):
            print(f"ERROR: Model(s) {wrong} do not exist", file=sys.stderr)
            return 1

        models = list(args.models)
        if len(include) != len(models):
            #TODO: identify duplicate model
            print(f"ERROR: Duplicate model names in list", file=sys.stderr)
            return 1

    logger.info(f"Running glitch on {models}")
    logger.info(f"Glitch output will be stored in {edir}")

    boxsize = None
    if args.boxsize:
        try:
            boxsize = parse_csnum(args.boxsize, float, ('length',
                                                        'width',
                                                        'height'),
                                  0)
        except ValueError:
            print(f"ERROR: {args.boxsize} is improperly formatted", file=sys.stderr)
            return 1

    out = {}
    for mn in models:
        m = ge.get_model(mn)
        m = m.override(args.sampling, boxsize, args.threshold, args.density)

        dirname = mn
        if m.has_override:
            dirname = dirname + "+" + m.override_suffix.replace(".", "_")
            logger.info(f"Invoking glitch for model '{m.name}' with overridden parameters {m.override_suffix}")
        else:
            logger.info(f"Invoking glitch for model '{m.name}'")

        gcode_orig = Path(je['models'][mn]['original']['path'])
        gcode_rotated = [(XYZTuple(**r['rotation']),
                          Path(r['path'])) for r in je['models'][mn]['rotated']]

        opdir = edir / (dirname + '.glitch')
        assert not opdir.exists(), opdir
        if not args.dryrun: opdir.mkdir()

        m.rotated_stl = [je['models'][mn]['original']['stlpath']]
        m.rotated_stl.extend([r['stlpath'] for r in je['models'][mn]['rotated']])

        m.load_heights()

        oabbox = m.get_oabbox(printerdim, gcode_orig, gcode_rotated,
                              oa_bbox=paths.oa_bbox)

        collect_files = m.invoke_glitch2(printerdim, gcode_orig,
                                         gcode_rotated,
                                         oabbox,
                                         output_dir = opdir,
                                         glitch = paths.glitch,
                                         dry_run=args.dryrun)

        if collect_files is None and not args.dryrun:
            return 1

        m.invoke_heatmap_merge(collect_files,
                               heatmap_merge = paths.heatmap_merge,
                               dry_run = args.dryrun)

    return 0

def do_config(args):
    cfg = Config()

    for k in ['pj3d', 'glitch', 'stlscale', 'stlrotate',
              ('oa_bbox', 'oabbox'), ('heatmap_merge', 'hmerge')]:
        if isinstance(k, tuple):
            key = k[0]
            arg = getattr(args, k[1])
        else:
            key = k
            arg = getattr(args, k)

        if arg is None:
            continue

        p = Path(arg)
        if p.exists():
            print(f"Setting path '{p.absolute()}' for '{key}'")
            cfg.set_path(key, str(p.absolute()))
        else:
            print(f"ERROR: Path '{p}' specified for '{key}' does not exist. Ignoring", file=sys.stderr)

    cfg.write_config()

def setup_logging(args):
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logging.getLogger().addHandler(sh)

    if args.logfile:
        fh = logging.FileHandler(args.logfile, mode='w')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logging.info(f"Logging to {args.logfile}")

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

    rm = sp.add_parser('rm', help="Remove a model")
    rm.add_argument("name", help="Name of the model to remove")

    gg = sp.add_parser('gcode', help="Generate GCode for an experiment")
    gg.add_argument("exptdir", help="Directory to store G-code for experiment")
    gg.add_argument("printer", help="Printer name")
    gg.add_argument("printsettings", help="Printer settings")
    gg.add_argument("models", nargs="*", help="Specify list of models to include in experiment")
    gg.add_argument("--pj3d", help="Path to pj3d binary", default=shutil.which('pj3d'))
    gg.add_argument("-l", dest="logfile", help="Specify a log file")
    gg.add_argument("-n", dest="dryrun", help="Dry-run, don't actually generate gcode", action="store_true")
    gg.add_argument("--stlscale", help="Path to stlscale binary", default=shutil.which('stlscale'))
    gg.add_argument("--stlrotate", help="Path to stlrotate binary", default=shutil.which('stlrotate'))

    gr = sp.add_parser('glitch', help="Run Glitch for an experiment")
    gr.add_argument("exptdir", help="Directory containing experiment (with gcode already generated)")
    gr.add_argument("models", nargs="*", help="Specify list of models to include in experiment")
    gr.add_argument("--glitch", help="Path to glitch", default=shutil.which('gcode_comp_Z.py'))
    gr.add_argument("-l", dest="logfile", help="Specify a log file")
    gr.add_argument("-n", dest="dryrun", help="Dry-run, don't actually generate gcode", action="store_true")
    gr.add_argument("-s", "--sampling", help="Sampling interval", type=float)
    gr.add_argument("-t", "--threshold", help="Threshold, in percentile", type=float)
    gr.add_argument("-d", "--density", help="Density for each box, in float", type=float)
    gr.add_argument("-b", "--boxsize", help="Box/cube size, for visualization") # Is this also used for HD?
    gr.add_argument("--oabbox", help="Path to oa_bbox.py", default=shutil.which('oa_bbox.py'))
    gr.add_argument("--hmerge", help="Path to heatmap_merge.py", default=shutil.which('heatmap_merge.py'))

    # gm = sp.add_parser('runone', help="Run Glitch on a single model")
    # gm.add_argument("name", help="Name of model")
    # gm.add_argument("printer", help="Printer name")
    # gm.add_argument("printsettings", help="Printer settings")
    # gm.add_argument("outputdir", help="Output directory, must not exist")
    # gm.add_argument("--pj3d", help="Path to pj3d binary", default=shutil.which('pj3d'))
    # gm.add_argument("--glitch", help="Path to glitch", default=shutil.which('gcode_comp_Z.py'))
    # gm.add_argument("-n", dest="dryrun", help="Dry-run, don't actually run glitch", action="store_true")
    # gm.add_argument("-l", dest="logfile", help="Log file name")

    cm = sp.add_parser('config', help="Configure Glitch Runner")
    cm.add_argument("--pj3d", help="Path to pj3d binary")
    cm.add_argument("--glitch", help="Path to glitch")
    cm.add_argument("--oabbox", help="Path to oa_bbox.py")
    cm.add_argument("--hmerge", help="Path to heatmap_merge.py")
    cm.add_argument("--stlscale", help="Path to stlscale binary")
    cm.add_argument("--stlrotate", help="Path to stlrotate binary")

    args = p.parse_args()

    if args.cmd == "create":
        sys.exit(do_create(args))
    elif args.cmd == "add":
        sys.exit(do_add(args))
    elif args.cmd == "runone":
        setup_logging(args)
        sys.exit(do_runone(args))
    elif args.cmd == "rm":
        sys.exit(do_rm(args))
    elif args.cmd == "gcode":
        setup_logging(args)
        sys.exit(do_gcode(args))
    elif args.cmd == "glitch":
        setup_logging(args)
        sys.exit(do_glitch(args))
    elif args.cmd == "config":
        sys.exit(do_config(args))
    else:
        raise NotImplementedError(args.cmd)
