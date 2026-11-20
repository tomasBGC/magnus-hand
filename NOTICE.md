# Notice and third-party components

This repository's own code and analysis are MIT-licensed (`LICENSE`); its hardware designs,
documentation, figures, data and trained models are CC BY 4.0 (`LICENSE-HARDWARE-DOCS-DATA.md`).
The third-party components below are the only ones present in the repository. They keep their
own licences and are **not** relicensed.

| Component | Where in this repository | Licence | Source |
|---|---|---|---|
| eFlesh mesh-generator files (modified or added) | `hardware/generator/` | MIT, Copyright (c) 2025 Venkatesh Pattabiraman | https://github.com/notvenky/eFlesh |
| eFlesh magnetometer-array firmware, on which the Teensy firmware is based | `firmware/MagneticSensors/MCU/` | MIT, same as above | https://github.com/notvenky/eFlesh (`arduino/5X_eflesh_stream`) |
| DownloadProject CMake template | `hardware/generator/cmake/DownloadProject.CMakeLists.cmake.in` | MIT, Copyright (c) 2015 Crascit | https://github.com/Crascit/DownloadProject |
| MLX90393 Arduino driver | `firmware/MagneticSensors/MCU/lib/MLX90393/` | MIT, Copyright (c) 2016-2018 Theodore C. Yapo | https://github.com/tesshellebrekers/arduino-MLX90393 |
| Pollen Robotics AmazingHand control code (derived) | `software/AmazingHand/hand_controller.py` | Apache License 2.0 | https://github.com/pollen-robotics/AmazingHand |
| Pollen Robotics AmazingHand mechanical design (the rigid fingertip mount is derived from it) | `hardware/print/Distal_V3.stl` | CC BY 4.0 | https://github.com/pollen-robotics/AmazingHand |

No file in the repository carries a share-alike or non-commercial licence term, and none
conflicts with the MIT and CC BY 4.0 licences above.

## eFlesh (MIT)

The fingertip geometry generator in `hardware/generator/` consists of local modifications and
additions to the mesh-generation tree `microstructure/microstructure_inflators` of eFlesh. The
modified or added files are `build.sh`, `create_pouch.py`, `cut-cell.ipynb`,
`cmake/DownloadProject.CMakeLists.cmake.in`, `cmake/MicroDependencies.cmake` (modified) and
`generate_eflesh.py`, `generate_eflesh_gui.py`, `run_eflesh.bat` (added). The unmodified upstream
code, its third-party libraries and its datasets are not copied here; obtain them from the upstream
repository (upstream commit checked out when this copy was made:
`4237e90cd7fef4f121af5f77dd6ff042f70ad122`). The `microstructure_inflators` folder in the upstream
repository has no licence file of its own, so the repository-level MIT licence below is the only
licence stated for it.

The magnetometer-array firmware in `firmware/MagneticSensors/MCU/` is credited in
`software/MagneticSensors/README.md` as based on the original `5X_eflesh_stream.ino` by
Venkatesh P (eFlesh). The eFlesh licence text, reproduced because modified or derived upstream
material is included:

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

## DownloadProject (MIT)

`hardware/generator/cmake/DownloadProject.CMakeLists.cmake.in` is the Crascit DownloadProject
template; its header reads "Distributed under the OSI-approved MIT License". Licence text from the
upstream repository:

```
The MIT License (MIT)

Copyright (c) 2015 Crascit

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

## MLX90393 driver (MIT)

`firmware/MagneticSensors/MCU/lib/MLX90393/` contains the MLX90393 magnetometer driver. Its file
headers read "Copyright 2016 Theodore C. Yapo" and "released under MIT License". Licence text from
the upstream repository:

```
The MIT License (MIT)

Copyright (c) 2016-2018 Theodore C. Yapo

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

## Pollen Robotics AmazingHand (Apache-2.0)

`software/AmazingHand/hand_controller.py` is a new implementation (it relays servo commands over
the existing eFlesh USB serial link) that is **derived from Pollen Robotics' AmazingHand code**: its
command logic mirrors `PythonExample/AmazingHand_FingerTest.py`, and the Feetech SCS0009 register
addresses and raw-value/angle formulas come from the project's `rustypot` `Scs0009PyController`
(`src/servo/feetech/scs0009.rs`). No upstream file is copied verbatim. The file carries a header
stating that it is derived from the upstream project and has been modified, as section 4 of the
Apache License asks for modified files. It remains under the Apache License 2.0; the licence text is
in `LICENSE-APACHE-2.0.txt`.

Source: https://github.com/pollen-robotics/AmazingHand. The upstream README states that the project
is licensed under Apache 2.0 and that its mechanical design is licensed under Creative Commons
Attribution 4.0 International (CC BY 4.0). The upstream licence file is the standard Apache 2.0
text and its copyright line is an unfilled template, so no named upstream copyright holder is given.
The rigid fingertip mount in `hardware/print/Distal_V3.stl` is derived from the AmazingHand
mechanical design, which is licensed CC BY 4.0 by Pollen Robotics; the derived part is released under
the same CC BY 4.0 licence (`LICENSE-HARDWARE-DOCS-DATA.md`), with credit to Pollen Robotics and a link
to https://creativecommons.org/licenses/by/4.0/. No other AmazingHand mechanical or CAD files are
included in this repository; the hand shown in `paper/fig_hand.png` is built from that design, and
the hand base, shells and other parts are available from the upstream repository.

## Credits without included files

`software/MagneticSensors/README.md` also credits the ReSkin and AnySkin projects
(https://any-skin.github.io). No files from them are included in this repository.

## Other

Python dependencies (see `requirements.txt`) are installed from PyPI under their own licences and
are not redistributed here. The robot hand, servos, magnetometer boards, load cell and
microcontrollers are third-party hardware and are not redistributed here.
