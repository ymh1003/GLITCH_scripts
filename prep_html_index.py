#!/usr/bin/env python3

import argparse
import glob
from pathlib import Path
import zipfile
import urllib

TMPL = """
<html>
<body>
{results}
</body>
</html>
"""

IND_RESULT = """
   <p>{path}
      <br/>
      <object type="application/pdf" data="{pdf}" style="height:4in; width:5in"></object>
      <img style="height:4in;" src="{png}" />
   </p>"""

def gen_html(results, orderfile):
    k = list(results.keys())

    if orderfile:
        with open(orderfile, "r") as of:
            order = {}
            n = 0
            for l in of:
                ls = l.strip()
                if not ls or ls[0] == "#": continue
                assert l not in k, f"Duplicate {l}"
                order[ls] = n
                n = n + 1

        k.sort(key=lambda x: order[str(Path(x).parent)])

    ind = []
    for i in k:
        ind.append(IND_RESULT.format(path = i,
                                     pdf = urllib.parse.quote(results[i]['pdf'][0]), png = urllib.parse.quote(results[i]['png'][0])))

    return TMPL.format(results="\n".join(ind))

def get_files(directory, force_all = False):
    out = {}
    for f in glob.glob(args.directory + "/**", recursive=True):
        if not (f.endswith('.png') or f.endswith('.pdf')): continue

        if f.endswith('.pdf') and '_orig_' in f:
            # skip _orig_dist.pdf
            continue

        pf = Path(f)
        pk = str(pf.parent)
        if pk not in out:
            out[pk] = {'png': [],
                       'pdf': []}

        out[pk][f[-3:]].append(f)

    keep = set()
    for k in out:
        if len(out[k]['png']):
            assert(len(out[k]['png']) == 1)
            if len(out[k]['pdf']) > 2: # overall, and avg
                # drop avg if overall is also present [only in invariant-checking]
                out[k]['pdf'] = [x for x in out[k]['pdf'] if '_avg_dist' not in x]

            keep.add(k)
        elif force_all:
            # for iv
            out[k]['png'].append('none.png')
            keep.add(k)

    return dict([(k, v) for k, v in out.items() if k in keep])

def write_zip(output, htmlfile, results):
    with zipfile.ZipFile(output, "w") as z:
        z.write(htmlfile)
        for k in results:
            f = results[k]
            if f['png'][0] != 'none.png':
                z.write(f['png'][0])
            z.write(f['pdf'][0])

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collate results as an HTML file")
    p.add_argument("directory")
    p.add_argument("outputhtml")
    p.add_argument("outputzip")
    p.add_argument("--order", help="Order of results, a file containing directories")
    p.add_argument("--all", help="Include all directories, not just those containing PNGs", action="store_true")

    args = p.parse_args()

    f = get_files(args.directory, args.all)
    h = gen_html(f, args.order)

    with open(args.outputhtml, "w") as html:
        html.write(h)

    write_zip(args.outputzip, args.outputhtml, f)
