# Notice and attribution

This repository builds on the open-source projects below. Licence text or
licence status is given for each.

## eFlesh (upstream) — MIT

The fingertip geometry generator in `hardware/generator/` consists of local
modifications and additions to the mesh-generation tree
`microstructure/microstructure_inflators` of eFlesh
(https://github.com/notvenky/eFlesh). The modified or added files are:
`build.sh`, `create_pouch.py`, `cut-cell.ipynb`, `cmake/DownloadProject.CMakeLists.cmake.in`,
`cmake/MicroDependencies.cmake` (modified) and `generate_eflesh.py`,
`generate_eflesh_gui.py`, `run_eflesh.bat` (added). The unmodified upstream
code, its third-party libraries and its datasets are not copied here; obtain
them from the upstream repository (upstream commit checked out when this
copy was made: `4237e90cd7fef4f121af5f77dd6ff042f70ad122`).

eFlesh is released under the MIT License. Its licence text is reproduced
here because modified upstream files are included:

```
MIT License

Copyright (c) 2025 Venkatesh Pattabiraman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The eFlesh sensing concept and the upstream firmware/software it provides
are the work of its authors; this repository's magnetometer-array firmware
and PC software are separate, locally developed code built for the AmazingHand.

## Pollen Robotics AmazingHand — provenance of `software/AmazingHand/hand_controller.py`

`software/AmazingHand/hand_controller.py` is a new implementation (it relays
servo commands over the existing eFlesh USB serial link) but is **derived from
Pollen Robotics' AmazingHand code**: its command logic mirrors
`PythonExample/AmazingHand_FingerTest.py`, and the Feetech SCS0009 register
addresses and raw-value/angle formulas come from the project's `rustypot`
`Scs0009PyController` (`src/servo/feetech/scs0009.rs`). The file's docstring
and `software/AmazingHand/README.md` say so. No upstream file is copied
verbatim.

Source: https://github.com/pollen-robotics/AmazingHand

Licence: **Apache License, Version 2.0** for the software. This was confirmed
by fetching the upstream `LICENSE` file itself:
https://raw.githubusercontent.com/pollen-robotics/AmazingHand/main/LICENSE
(page: https://github.com/pollen-robotics/AmazingHand/blob/main/LICENSE). It is
the standard Apache 2.0 text; its copyright line is still the unfilled
template placeholder, so no named copyright holder is given upstream. The
repository README states "Project is licensed under Apache 2.0" for the
software and Creative Commons Attribution 4.0 International (CC BY 4.0) for the
mechanical design; no hardware or CAD files are redistributed here. The upstream
`PythonExample/` folder
(https://github.com/pollen-robotics/AmazingHand/tree/main/PythonExample) has no
separate LICENSE or file header, so the root Apache-2.0 licence applies to it.

The local copy of this project's `AmazingHand` folder contained no LICENSE file
and no licence headers; `hand_controller.py` now carries a header stating that
it is derived from the upstream project (Apache-2.0) and has been modified, as
Apache-2.0 section 4 asks for modified files. The upstream licence text is not
reproduced in this repository; see the URL above.

## MLX90393 library — MIT (per file header)

`firmware/MagneticSensors/MCU/lib/MLX90393/` contains an MLX90393 magnetometer
driver whose file headers read "Copyright 2016 Theodore C. Yapo" and
"released under MIT License (see file)". The licence file referred to is not
present in this copy; consult the original library for the full text.

## Other

Python dependencies (see `requirements.txt`) are installed from PyPI under
their own licences. The robot hand and its servos are third-party hardware and
are not redistributed here.
