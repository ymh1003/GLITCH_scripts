#!/usr/bin/env python3

import argparse
import json
import re
import sys
import csv

SLICER_COMPARISON_BEGIN_RE = re.compile(r"gcode_comp_Z\.py --prusa")
MESHREPAIR_COMPARISON_BEGIN_RE = re.compile(r"gcode_comp_Z\.py.*mesh(lab|mixer)")
INVARIANT_CHECKING_BEGIN_RE = re.compile(r"glitch_runner.py tricky")
MAKE_ERROR_RE = re.compile(r"make: \*\*\* .*Error.*")

# handle both time and time -p
TIME_OUTPUT_RE =re.compile(r"^(?P<key>real|user|sys)\s+(?P<value>[0-9ms.]+)$")

GCODE_NAME_RE = re.compile("--name (?P<name>[^\s]+)")
RUNNER_NAME_RE = re.compile("/(?P<name>[^/]+)\.yaml")

GCODE_TIME_RE = [re.compile("(?P<key>[^,]+) (?P<value>\d+) (?P<unit>ns)"),
                 (re.compile("(?P<key>Time for removing duplicates):\s+(?P<value>[\d\.]+)"), "s")]

class RunData:
    def __init__(self, title, logfile):
        self.title = title
        self.data = {}
        self.status = "SUCCESS"
        self.logfile = logfile
        self._make_name()

    def _make_name(self):
        if "gcode_comp_Z.py" in self.title:
            m = GCODE_NAME_RE.search(self.title)
        else:
            m = RUNNER_NAME_RE.search(self.title)

        if m:
            self.name = m.group('name')
        else:
            self.name = "<unknown>"

    def add(self, key, value):
        if key in self.data:
            if not isinstance(self.data[key], list):
                self.data[key] = [self.data[key]]
            self.data[key].append(value)
        else:
            self.data[key] = value

    def __str__(self):
        return self.name + " " + self.title + " " + self.status

    def flatten(self):
        out = dict(self.data)
        out['_status'] = self.status
        out['_name'] = self.name
        out['_title'] = self.title
        out['_logfile'] = self.logfile

        for k in self.data:
            if isinstance(self.data[k], list):
                del out[k]
                # note #0 may indicate spurious/duplicate data sometimes
                # seen in some log files...
                out.update(dict([(k + "#" + str(i), v) for i, v in enumerate(self.data[k])]))

        return out

    __repr__ = __str__

def parse_log(lf):
    current_run_data = None
    skipped = 0
    records = 0
    with open(lf, "r") as f:
        for l in f:
            for pattern in [SLICER_COMPARISON_BEGIN_RE,
                             MESHREPAIR_COMPARISON_BEGIN_RE,
                             INVARIANT_CHECKING_BEGIN_RE]:
                m = pattern.search(l)
                if m is not None:
                    if current_run_data is not None:
                        records += 1
                        yield current_run_data

                    current_run_data = RunData(l.strip(), lf)
                    break

            if current_run_data is None:
                skipped += 1
                continue

            m = MAKE_ERROR_RE.match(l)
            if m:
                current_run_data.status = f"ERROR ({l.strip()})"
                continue

            m = TIME_OUTPUT_RE.match(l)
            if m:
                current_run_data.add("_" + m.group('key'), m.group('value'))
                continue

            # maybe track position in original file for header
            # ordering in output
            for tr in GCODE_TIME_RE:
                if isinstance(tr, tuple):
                    tr, unit = tr
                else:
                    unit = None

                m = tr.match(l)
                if m:
                    unit = unit or m.group('unit')
                    k = f"{m.group('key')} [{unit}]"
                    v = m.group('value')
                    current_run_data.add(k, v)


    if current_run_data is not None:
        records += 1
        yield current_run_data

    print(f"{lf}: {skipped} lines skipped, {records} records processed.", file=sys.stderr)

class OutFile:
    def __init__(self, filename):
        self.data = []
        self._header_keys = set()
        self._header = []
        self.filename = filename

    def _process_header(self, k):
        if k not in self._header_keys:
            self._header_keys.add(k)
            self._header.append(k)

    def add(self, rd):
        rf = rd.flatten()
        self.data.append(rf)
        for r in rf:
            self._process_header(r)

    def write(self):
        with open(self.filename, "w") as f:
            o = csv.DictWriter(f, self._header)
            o.writeheader()
            o.writerows(self.data)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract data from artifact logs")
    p.add_argument("logfiles", nargs="+")
    p.add_argument("-o", dest="output", help="Output file")

    args = p.parse_args()

    out = OutFile(args.output if args.output else "/dev/stdout")
    for lf in args.logfiles:
        for rd in parse_log(lf):
            out.add(rd)

    out.write()

