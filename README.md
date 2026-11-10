# Magnetic tactile fingertips for a robot hand

Code, firmware, models and data for 3D-printed, magnetic ("eFlesh"-style)
tactile fingertips fitted to an open-source robot hand (Pollen Robotics'
AmazingHand). Each fingertip contains a small array of magnetometers that
sense how a soft printed lattice deforms. A learned model turns the raw
magnetic readings into contact, force and contact direction, and a second
sensor (a load cell) provides the reference force.

This repository is a cleaned copy: only what is needed to understand and
reproduce the work is included. See `NOTICE.md` for credits and licences of
the upstream projects this builds on.

## What is in each folder

| Folder | What it contains |
|---|---|
| `hardware/generator/` | The scripts used to generate the fingertip lattice geometry. These are my local edits and additions to the upstream eFlesh mesh generator (`microstructure_inflators`): the build script, the fingertip generator (`generate_eflesh.py`, plus a GUI), the pouch/cut-cell helpers and two modified CMake files. The upstream code itself, its third-party libraries and its build output are **not** included; see `NOTICE.md` for where to get them. |
| `firmware/` | Microcontroller code (PlatformIO). `MagneticSensors/MCU` is the Teensy 4.0 firmware that reads the magnetometer array and streams frames to the PC (this is the deployed firmware). `LoadScale/MCU` is the small firmware that reads the load cell. |
| `software/` | The PC-side Python code. `MagneticSensors/` and `LoadScale/` talk to the two sensors; the top-level `unified_*.py` scripts record both together; `AmazingHand/hand_controller.py` drives the finger servos; `model_v1/` holds model training, evaluation and the live sensing / hand-control interfaces, plus the notes and reports that document how the models were built. |
| `models/` | Four trained models as pickle files, with `SHA256SUMS.txt` and a `README.md` saying which paper number each explains: `before_true_zero/` (older contact classifier), `after_true_zero/` (newer contact classifier; the two are compared in the before/after results), `deployed/` (the model used live) and `grasp_trained_force/` (the force model trained on the load-cell grasp data). |
| `data/` | The datasets used: the calibration-rig recordings, the grasp data paired with load-cell force, and the smaller derived training sets. Raw recordings and very large derived files are not included; see `data/README.md`. |
| `analysis/` | The paper analysis: scripts that compute each result, the `results/` CSV outputs, the figure scripts (`fig*.py`) and `RESULTS_LEDGER.md`, which records every number reported. |
| `paper/` | The final figures only. |

Also at the top level: `requirements.txt` (pinned Python dependencies),
`.gitignore` and `NOTICE.md`.

## Things to know

* **Pickle files run code when loaded.** Only load the files in `models/`
  if you trust this repository, and check them against `models/SHA256SUMS.txt`.
* **Paths.** The scripts in `analysis/` find `data/` and `models/` from their
  own location, so they run from the repository root (or anywhere), for
  example `python analysis/a_contact_accuracy.py`. Setting the environment
  variable `EFLESH_OUT_DIR` makes them write results, figures and the force
  model to that folder instead of overwriting the committed ones. The training
  and recording scripts under `software/` were left as written for the original
  folder layout (they expect `model_v1/`, `recordings/` and the datasets next to
  each other), so they need their paths adjusted before they will run here;
  `data/README.md` lists which script reads which file.
* **Not everything under `software/model_v1/` is reproducible from this
  repository.** Intermediate datasets and raw recordings were left out because
  of size and because some were superseded. Reports there describe those
  steps for context.
* The hardware (the hand itself) is the AmazingHand design by Pollen Robotics
  and is not redistributed here.
