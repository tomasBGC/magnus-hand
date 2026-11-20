# hardware

`generator/` generates the soft lattice geometry of a fingertip from an input STL of its shell and cuts
the magnet pouches into it. The files are local edits and additions to the mesh generator of eFlesh (see
`NOTICE.md`); the upstream code, its third-party libraries and its build output are not included.
`BOM.md` is the bill of materials (total $256.29).

| File | What it is |
|---|---|
| `BOM.md` | Bill of materials and per-fingertip cost, print time and filament use. |
| `generator/generate_eflesh.py` | Command-line generator: input STL, cell size, material parameters and magnet positions in; lattice with magnet pouches out. Runs the built inflator tools and Blender. |
| `generator/generate_eflesh_gui.py` | GUI front end for the generator. |
| `generator/run_eflesh.bat` | Windows launcher for the GUI. |
| `generator/build.sh` | Builds the inflator tools the generator needs. |
| `generator/create_pouch.py` | Blender script that creates the magnet pouches and combines them with the lattice. |
| `generator/cut-cell.ipynb` | Cut-cell notebook. |
| `generator/cmake/` | The two CMake files used by the build (`DownloadProject.CMakeLists.cmake.in`, `MicroDependencies.cmake`). |
| `print/` | Print files for one fingertip (TPU lattice STL, sliced Bambu Studio project, rigid PLA mount STL, generator inputs), with print settings, assembly order and checksums; see `print/README.md`. |

The print files in `print/` total 47.0 MB; the largest is the TPU lattice at 42.8 MB. The hand base, shells and
other AmazingHand parts are not included; get them from
https://github.com/pollen-robotics/AmazingHand.
