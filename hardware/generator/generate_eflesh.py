#!/usr/bin/env python3
"""
generate_eflesh.py - Generate an eFlesh microstructure from an input STL.

Usage:
    python generate_eflesh.py Fingertip_Shell_V1.stl
    python generate_eflesh.py input.stl --E 0.1 --cell-size 8
    python generate_eflesh.py input.stl --magnets '[[4.76,1.59,[5,0,-5]],[4.76,1.59,[5,0,5]]]'
"""
import argparse
import copy
import json
import os
import re
import subprocess
import sys
import tempfile

import meshio
import numpy as np


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def win_to_wsl(path):
    """Convert an absolute Windows path to a WSL /mnt/<drive>/... path."""
    path = str(path).replace('\\', '/')
    m = re.match(r'^([A-Za-z]):(.*)', path)
    if m:
        return f'/mnt/{m.group(1).lower()}{m.group(2)}'
    return path


def abs_wsl(path):
    return win_to_wsl(os.path.abspath(path))


# ---------------------------------------------------------------------------
# Step 1: replicate cut-cell notebook — build data.json
# ---------------------------------------------------------------------------

def change_file_extension(filename, new_ext):
    base, _ = os.path.splitext(filename)
    return base + '.' + new_ext


def generate_data_json(input_stl, cell_size, E, nu,
                       microstructure_repo_path, matopt_repo_path,
                       only_cube_cells=False):
    """Read STL, compute grid, evaluate material params, write data.json."""
    input_stl = os.path.abspath(input_stl)
    base      = os.path.splitext(input_stl)[0]
    input_obj = base + '.obj'
    out_path  = f'{base}_{E}_{cell_size}.obj'

    m = meshio.read(input_stl)
    m.write(input_obj)

    v         = m.points.astype(float)
    bbox_min  = np.amin(v, axis=0)
    bbox_max  = np.amax(v, axis=0)
    corner0   = list(map(int, np.ceil(bbox_min / cell_size) - 1))
    corner1   = list(map(int, np.floor(bbox_max / cell_size)))
    print(f'  Bounding box: {bbox_min.tolist()} -> {bbox_max.tolist()}')
    print(f'  Grid corners: {corner0} -> {corner1}')

    sys.path.insert(0, os.path.join(matopt_repo_path, 'tools/material2geometry'))
    from material2geometry import Material2Geometry

    coeffs   = os.path.join(matopt_repo_path,
                            'tools/material2geometry/0646_geo_1_coeffs.txt')
    mat2geo  = Material2Geometry(in_path=coeffs)

    wire = os.path.join(microstructure_repo_path,
                        'data/patterns/3D/reference_wires/pattern0646.wire')
    patterns = []
    entry    = {'params': [], 'symmetry': 'Cubic', 'pattern': wire, 'index': [0, 0, 0]}

    for i in range(corner0[0], corner1[0] + 1):
        for j in range(corner0[1], corner1[1] + 1):
            for k in range(corner0[2], corner1[2] + 1):
                entry['params'] = mat2geo.evaluate(nu, E)
                entry['index']  = [i, j, k]
                patterns.append(copy.deepcopy(entry))

    print(f'  Generated {len(patterns)} cell entries')

    data_json = os.path.join(microstructure_repo_path, 'data.json')
    with open(data_json, 'w') as fp:
        json.dump(patterns, fp)

    return input_obj, out_path, data_json


# ---------------------------------------------------------------------------
# Step 2: fix Windows paths in data.json → WSL paths
# ---------------------------------------------------------------------------

def fix_paths_in_json(data_json_path):
    """Replace Windows C:\... paths inside data.json with /mnt/c/... paths."""
    with open(data_json_path, 'r') as f:
        patterns = json.load(f)

    for entry in patterns:
        if 'pattern' in entry:
            entry['pattern'] = win_to_wsl(entry['pattern'])

    with open(data_json_path, 'w') as f:
        json.dump(patterns, f)


# ---------------------------------------------------------------------------
# Step 3: call cut_cells_cli via WSL
# ---------------------------------------------------------------------------

def run_cut_cells(cut_cells_cli, data_json_path, input_obj,
                  out_path, cell_size, only_cube_cells=False):
    cmd = [
        'wsl',
        win_to_wsl(cut_cells_cli),
        '-p', abs_wsl(data_json_path),
    ]
    if not only_cube_cells:
        cmd += ['--surface', abs_wsl(input_obj)]
    cmd += ['--gridSize', str(cell_size), '-o', abs_wsl(out_path), '-r', '50']

    print(f'  {" ".join(cmd)}')
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Step 4: call Blender to run create_pouch logic
# ---------------------------------------------------------------------------

def run_blender(blender_exe, create_pouch_script, input_obj, output_obj, magnets):
    """Write a temp Blender driver script that imports create_pouch and runs it."""
    script_dir = os.path.dirname(os.path.abspath(create_pouch_script))

    driver = f"""\
import sys, math
sys.path.insert(0, {repr(script_dir)})
import create_pouch
import bpy

input_path  = {repr(os.path.abspath(input_obj))}
output_path = {repr(os.path.abspath(output_obj))}
list_of_magnets = {json.dumps(magnets)}

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.wm.obj_import(filepath=input_path)
bpy.context.view_layer.objects.active.name = "micro"

for index, magnet in enumerate(list_of_magnets):
    rot = magnet[3] if len(magnet) > 3 else [0, 90, 0]
    ori = tuple(math.radians(d) for d in rot)
    convex_hull, pouch_obj = create_pouch.create_pouch(
        magnet[0], magnet[1], magnet[2], "pouch" + str(index), orientation=ori
    )
    create_pouch.boolean("micro", convex_hull, "DIFFERENCE")
    create_pouch.boolean("micro", pouch_obj, "UNION")

create_pouch.export_mesh(output_path)
print("Blender: exported to", output_path)
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                     encoding='utf-8') as tmp:
        tmp.write(driver)
        tmp_path = tmp.name

    try:
        cmd = [blender_exe, '--background', '--python', tmp_path]
        print(f'  {" ".join(cmd)}')
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_MAGNETS = [
    [4.76, 1.59, [5, 0, -5]],
    [4.76, 1.59, [5, 0,  5]],
]

DEFAULT_BLENDER = r'C:\Program Files\Blender Foundation\Blender 4.2\blender.exe'


def main():
    parser = argparse.ArgumentParser(
        description='Generate an eFlesh microstructure from an input STL.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('input_stl',
                        help='Input STL file')
    parser.add_argument('--cell-size', type=float, default=6,
                        help='Cell size in mm')
    parser.add_argument('--E',         type=float, default=0.05,
                        help="Young's modulus")
    parser.add_argument('--nu',        type=float, default=0.3,
                        help='Poisson ratio')
    parser.add_argument('--magnets',   type=json.loads,
                        default=DEFAULT_MAGNETS,
                        metavar='JSON',
                        help='Array of [diameter, thickness, [x,y,z]] per magnet')
    parser.add_argument('--blender',   default=DEFAULT_BLENDER,
                        help='Path to blender.exe')
    args = parser.parse_args()

    script_dir       = os.path.dirname(os.path.abspath(__file__))
    matopt_repo_path = os.path.join(os.path.dirname(script_dir), 'matopt')
    cut_cells_cli    = os.path.join(script_dir,
                                    'build', 'isosurface_inflator', 'cut_cells_cli')
    create_pouch_py  = os.path.join(script_dir, 'create_pouch.py')

    # ------------------------------------------------------------------
    print('[1/4] Generating data.json ...')
    input_obj, out_path, data_json = generate_data_json(
        args.input_stl, args.cell_size, args.E, args.nu,
        script_dir, matopt_repo_path,
    )

    # ------------------------------------------------------------------
    print('[2/4] Fixing Windows paths in data.json for WSL ...')
    fix_paths_in_json(data_json)

    # ------------------------------------------------------------------
    print('[3/4] Running cut_cells_cli via WSL ...')
    run_cut_cells(cut_cells_cli, data_json, input_obj, out_path, args.cell_size)

    # ------------------------------------------------------------------
    final_output = os.path.splitext(out_path)[0] + '_with_pouches.obj'
    print('[4/4] Running Blender to add pouches ...')
    run_blender(args.blender, create_pouch_py, out_path, final_output, args.magnets)

    print(f'\nDone.  Final file saved to: {final_output}')


if __name__ == '__main__':
    main()
