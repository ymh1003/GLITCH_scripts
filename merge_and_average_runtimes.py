#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import sys

NAME_CHANGES = {'12975': 'Amy',
                'ercf-paper-exp': 'BearingInsert',
                '13339': 'Fuselage_A',
                'dusty': 'Fuselage_B',
                '5582': 'HexagonalCap',
                'przedluzka': 'Ring',
                '15653': 'Turbo',
                'adapter'	: 'Adapter',
                'air_nozzle'	: 'AirNozzle',
                'airplane'	: 'Airplane',
                'arm'	: 'Arm',
                'bagon'	: 'Bagon',
                'batman'	: 'Batman',
                'beard'	: 'Beard',
                'bolt'	: 'Bolt',
                'borboleta'	: 'Borboleta',
                'bottlecap'	: 'Bottlecap',
                'car'	: 'Car',
                'carcasa'	: 'Carcasa',
                'circularHole'	: 'CircularHole',
                'clip'	: 'Clip',
                'cloud'	: 'Cloud',
                'cresGaruru'	: 'CresGaruru',
                'doubleCube'	: 'DoubleCube',
                'dragon'	: 'Dragon',
                'drive_frame_upper'	: 'DriveFrameUpper',
                'drum'	: 'Drum',
                'faucetHead'	: 'FaucetHead',
                'ford'	: 'Ford',
                'frame'	: 'Frame',
                'ghostMask'	: 'GhostMask',
                'gun'	: 'Gun',
                'holyGrail'	: 'HolyGrail',
                'horse'	: 'Horse',
                'm2_nut'	: 'M2Nut',
                'man'	: 'Man',
                'omegaBoard'	: 'OmegaBoard',
                'radenci'	: 'Radenci',
                'rear_bed_mount_left'	: 'ReadBedMountRight',
                'rear_bed_mount_right'	: 'RearBedMountLeft',
                'samurai'	: 'Samurai',
                'ship'	: 'Ship',
                'slices'	: 'Slices',
                'spool_holder'	: 'SpoolHolder',
                'sterm'	: 'Sterm',
                'sword'	: 'Sword',
                't8_nut_block'	: 'T8NutBlock',
                'tabby'	: 'Tabby',
                'tapa'	: 'Tapa',
                'tardis'	: 'Tardis',
                'trash'	: 'Trash',
                'tray'	: 'Tray',
                'truss'	: 'Truss',
                'vents'	: 'Vents',
                'vertexProblem'	: 'VertexProblem',
                'warrior'	: 'Warrior',
                }

def load_expt_run(csvfiles):
    # multiple csv files can belong to the same experimental run in
    # case the same benchmark is run multiple times, keep only the
    # last one

    out = pd.DataFrame()
    for f in csvfiles:
        df = pd.read_csv(f)

        df = df[["_real", "_status", "_name"]]
        # drop all <unknown> from name
        # drop all ERROR status
        succ = df[(df["_status"] == "SUCCESS") & (df["_name"] != "<unknown>")]
        out = pd.concat([out, succ])

    ret = out.drop_duplicates(['_name'], keep='last')
    print(len(out) - len(ret), "row(s) dropped as duplicate")
    return ret

def do_avg(data):
    out = pd.concat(data)

    d = out.groupby('_name')["_real"].describe()
    d["ci95"] = 1.96*(d["std"] / np.sqrt(d["count"]))

    d.index = [x if x not in NAME_CHANGES else NAME_CHANGES[x] for x in d.index]
    return d


def produce_fig_csv(avg_data, printdata):
    pt = pd.read_csv(printdata, index_col=['Benchmark name'])

    ad = avg_data.join(pt)
    out = pd.DataFrame({'Analysis time': ad['mean'],
                        'Print time': ad['Print time'],
                        })
    out['Ratio'] = out['Analysis time'] / out['Print time']
    out.index.name = 'Benchmark name'
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge and average CSV files to produce runtime data")

    p.add_argument("-m", "--merge", action="append", help="Specify as CSV1,CSV2 to merge, with the merged version being treated as a single CSV file for averaging", default=[])
    p.add_argument("csvfile", nargs="*", help="CSVFile to average (when multiple specified")
    p.add_argument("-p", dest="print_time", help="CSV containing print time data", default="print_times.csv")
    p.add_argument("-o", dest="output", help="CSV for final output", default="figure-experiment.csv")

    args = p.parse_args()

    if len(args.merge) == 0 and len(args.csvfile) == 0:
        print("ERROR: You need to specify atleast 1 CSV file")
        sys.exit(1)

    data = []
    for m in args.merge:
        data.append(load_expt_run(m.split(",")))

    for f in args.csvfile:
        data.append(load_expt_run([f]))

    ad = do_avg(data)
    fd = produce_fig_csv(ad, args.print_time)
    fd.to_csv(args.output)
