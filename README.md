# Magnus Hand: Low-Cost 3D-Printed Magnetic Tactile Fingertips for an Open-Source Dexterous Hand

**Author:** Tomás Costa, Institute for Manufacturing (Computer-Aided Manufacturing Group), University of Cambridge; MPhil in Industrial Systems, Manufacture, and Management (ISMM - C60)

## Summary

This repository holds the code, firmware, trained models, data and analysis for 3D-printed magnetic
tactile fingertips fitted to Pollen Robotics' open-source AmazingHand. Each fingertip is a soft printed
lattice (TPU 95A) with embedded magnets above a board of five magnetometers; pressing the lattice moves
the magnets and the magnetometers read the change. A learned model turns the raw readings into contact,
force and contact direction, and a separate load cell provides the reference force. The repository
contains the generator for the fingertip geometry, the microcontroller firmware, the PC software for
recording and live use, the models, the datasets and the scripts behind every reported number.

## Key results

All values come from `analysis/RESULTS_LEDGER.md`. Every row there is offline-only: tested on held-out
data from the same grasp dataset (`data/load_cell_dataset.csv`), seed 42, one split, no confidence
intervals, and not validated on hardware. "Standard split" is a rep-grouped 80/20 holdout;
"leave-one-scenario-out" (LOSO) holds out each of the 7 finger-combination scenarios in turn.

| Result | Value |
|---|---|
| Contact accuracy of the deployed classifier, standard split | Index 0.990, Middle 0.931, Ring 0.977, Thumb 0.529; all fingers 0.874 (n = 3597) |
| Force error of the grasp-trained model, standard split, all rows | MAE 0.2096 N, RMSE 0.3116 N, r = 0.9156 (constant-mean baseline MAE 0.6782 N) |
| Force error, leave-one-scenario-out, all rows | MAE 0.6569 N (baseline 0.7201 N), r = 0.3456 |
| Force error, leave-one-scenario-out, touch rows only | MAE 0.5795 N (baseline 0.5728 N): no better than the baseline |
| Light / medium / firm level accuracy, standard split | 0.7472 |
| Light vs firm at a 1.0 N threshold, standard split | accuracy 0.9738 (majority-class accuracy 0.8091) |
| Light vs firm at a 1.0 N threshold, leave-one-scenario-out | accuracy 0.7219 (majority-class accuracy 0.8082) |
| Closed-loop grip recordings (live hand, two single recordings) | 21 slip-event ticks (21 July 2026) and 81 (8 September 2026); descriptive counts only |

Caveats C1 to C8 in the ledger apply to these numbers; for example, the Thumb standard-split holdout has
no no-touch rows, and the deployed contact classifier was itself trained on this dataset's training split.

## Repository map

| Folder | Contents |
|---|---|
| `analysis/` | Scripts that compute each reported result, their CSV outputs, the figure scripts and `RESULTS_LEDGER.md`. |
| `data/` | The datasets used, the closed-loop grip recordings and a README describing every file. |
| `firmware/` | PlatformIO firmware for the Teensy 4.0 magnetometer array and the load-cell microcontroller. |
| `hardware/` | The fingertip lattice generator, the bill of materials and the print files for one fingertip. |
| `models/` | The four trained models, with `SHA256SUMS.txt` and a README saying which results each explains. |
| `paper/` | The final figures. |
| `software/` | PC-side Python: sensor and load-cell interfaces, recording tools, hand control, and `model_v1/` (model training, evaluation, live interfaces). |

## Getting started

The analysis runs on the committed data and models. These commands were run on Windows 10 with
Python 3.14.5 (scikit-learn 1.9.0, as pinned in `requirements.txt`), from the repository root:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python analysis\a_contact_accuracy.py
.venv\Scripts\python analysis\b_force_error.py
.venv\Scripts\python analysis\train_force_candidate.py
```

`a_contact_accuracy.py` reproduces the per-finger contact accuracies (about 6 s), `b_force_error.py`
the force errors (about 1.5 min) and `train_force_candidate.py` retrains the grasp-trained force model
(about 20 s, holdout MAE 0.210 N). Each script writes into `analysis/results/` or
`models/grasp_trained_force/`; set the environment variable `EFLESH_OUT_DIR` to another folder first to
leave the committed files untouched (run this way, the outputs of `a_contact_accuracy.py` and
`b_force_error.py` were byte-identical to the committed CSVs). To check the models against their
checksums, run `sha256sum -c SHA256SUMS.txt` from inside `models/`.

Pickle files run code when loaded; load the files in `models/` only if you trust this repository.
The firmware and the live hardware software were not built or run as part of preparing this
repository. The recording and training scripts under `software/` keep the paths of the original working
folder (`model_v1/`, `recordings/`), so they need their paths adjusted before they run here; see
`data/README.md` for which script reads which file.

## Data

See `data/README.md` for every file, how it was produced, which scripts read it, what is not included
(raw recordings and the three large `*_backregion.csv` files) and the SHA-256 checksums of the latter.

## Hardware

- **Fingertip:** a deformable lattice printed in TPU 95A with a hard PLA mount, five magnets in a star
  formation and a board of five magnetometers (`paper/fig_fingertip.png`). The lattice is generated by
  `hardware/generator/` from a fingertip-shell STL.
- **Sensing electronics:** four boards of five MLX90393 magnetometers (20 sensors) read through a
  4-channel I2C multiplexer (STEMMA QT, addressed in firmware as a TCA9546A at 0x70) by a Teensy 4.0 (`firmware/MagneticSensors/`).
- **Reference force:** an HX711 amplifier with a strain-gauge load cell on an ATmega328P board
  (`firmware/LoadScale/`).
- **Hand:** the AmazingHand with Feetech SCS0009 servos, controlled through `software/AmazingHand/`.
- **Cost:** the full system (AmazingHand, four sensing fingertips and electronics) costs $256.29 in
  total; see [`hardware/BOM.md`](hardware/BOM.md) for the itemised bill (prices sourced in GBP in August
  2026 and converted to USD), the component list, and the per-fingertip consumable cost ($3.04), print time
  and filament use.
- **Print files:** [`hardware/print/`](hardware/print/README.md) holds the files for one fingertip: the TPU
  lattice (STL and a sliced Bambu Studio project), the rigid PLA magnetometer mount and adaptor (STL), and
  the generator inputs, with print settings, assembly order and checksums. They are licensed CC BY 4.0, and
  the fingertip mount is derived from the AmazingHand mechanical design (Pollen Robotics, CC BY 4.0). The
  hand base, shells and other parts are not included; get them from the upstream AmazingHand repository. The
  lattice can be regenerated with `hardware/generator/`.

## Relation to prior work

- **eFlesh** (Venkatesh Pattabiraman, MIT; https://github.com/notvenky/eFlesh) provides the magnetic
  lattice-skin concept, the mesh generator that `hardware/generator/` modifies and extends, and the
  original magnetometer streaming firmware on which `firmware/MagneticSensors/` is based.
- **AmazingHand** (Pollen Robotics; https://github.com/pollen-robotics/AmazingHand) is the open-source
  hand the fingertips are fitted to. `software/AmazingHand/hand_controller.py` is derived from its
  servo control code (Apache-2.0). Its mechanical design is not redistributed here.
- **New in this repository:** the fingertip generator scripts and GUI; the 20-sensor firmware and PC
  software; the load-cell reference and the combined recording tools; the models for contact, force and
  contact direction, including the true-zero contact fix and the crosstalk models; the grasp dataset
  paired with load-cell force; the analysis and results ledger; and the closed-loop grip recordings.

See `NOTICE.md` for every third-party component and its licence.

## How to cite

If you use this repository, please cite it using `CITATION.cff` (GitHub's "Cite this repository" reads
it), or as:

```
Costa, T. (2026). Magnus Hand: Low-Cost 3D-Printed Magnetic Tactile Fingertips for an Open-Source
Dexterous Hand. Software and data repository.
```

## Authorship and provenance

The research, hardware design, experiments and analysis, building on eFlesh and AmazingHand (see Relation
to prior work), are the work of Tomás Costa, carried out as part of a dissertation for the MPhil in
Industrial Systems, Manufacture, and Management (ISMM - C60) and follow-on research at the University of
Cambridge. The repository was
organised and documented with the help of Claude Code using the Claude Sonnet 5.5 model, under the
author's direction and review. The author is responsible for all content.

## Licence

- Software and analysis code: MIT (`LICENSE`).
- Hardware designs, documentation, figures, data and trained models: CC BY 4.0
  (`LICENSE-HARDWARE-DOCS-DATA.md`).
- Third-party files keep their own licences (`NOTICE.md`); `software/AmazingHand/hand_controller.py`
  is Apache-2.0 (`LICENSE-APACHE-2.0.txt`).

## Acknowledgements

Supervisors: Sebastian Pattinson and Liyou Zhou. Collaborator: Hamza Khattak.
