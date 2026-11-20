# Closed-loop grip recordings

Two live recordings from the Force-Target Grip controller (`software/model_v1/hand_control_gui.py`) on the real hand. They are separate from `load_cell_dataset.csv` and are **not** used by the paper analysis scripts. Numbers derived from them are in the last section of `analysis/RESULTS_LEDGER.md`.

## Files

| File | Recording | Contents |
|---|---|---|
| `slip_detection_live_verification.json` | 2026-07-21, 19:26 UTC, 43.4 s | 524 ticks (293 grip phase, 231 hold phase). Each tick has `force`, `raw_force`, `contact`, `slip_active` and `slip_event` for Index, Middle, Ring and Thumb. Top level: `grip_state`, `abort_reason`, `n_slip_events` (21). |
| `eflesh_test_20260908T121329Z grasp test_meta.json` | 2026-09-08, 12:12:46 to 12:13:26 UTC | Live thresholds, `n_ticks` (323), `n_slip_events` (81) and the list of slip events (`finger`, `force_at_event`, `message`). |
| `eflesh_test_20260908T121329Z grasp test.csv` | same recording | One row per tick: per-finger position (%), `force`, `raw_force`, `contact`, `slip_active`, `slip_event` and `slip_message`. |

The screen video of the 8 September recording (`.avi`, 46 MB) is not included.

## Provenance and the one edit

Copied from the backup of the original project folder (`Sensor System`). The SHA-256 of each **original** file, recorded before editing:

| File | SHA-256 of the original |
|---|---|
| `slip_detection_live_verification.json` | `74be90cd67c30079685388cc7038beca16137979ccad4d5493ee9e77bf7388da` |
| `eflesh_test_20260908T121329Z grasp test_meta.json` | `f508bb6b2555fbf5355f8b79cf4f0c3db244719f57341e1d7481e797ff06e873` |
| `eflesh_test_20260908T121329Z grasp test.csv` | `133c342076aaba2b5c568ea7f047caec8bea42cfd7c0e503cb82afbd0acdb0e5` |

In the two JSON copies, **one value was changed and nothing else**: the `model_path` field (a local file path of the model that was loaded) was replaced with `"<local path removed>"`. The CSV is byte-identical to its original. Because of this edit, the SHA-256 of the two JSON copies here does not match the values above; the CSV's does. The SHA-256 of the files as they are in this repository:

| File | SHA-256 in this repository |
|---|---|
| `slip_detection_live_verification.json` | `74c11c02912c9de50895f6047d068ba89e3a6a274331423b6cd009ad87c30c70` |
| `eflesh_test_20260908T121329Z grasp test_meta.json` | `90a8d248446281377faff03e8a06dfa918d9b33c5f4c0b64470cb976c4ce5468` |
| `eflesh_test_20260908T121329Z grasp test.csv` | `133c342076aaba2b5c568ea7f047caec8bea42cfd7c0e503cb82afbd0acdb0e5` |

## Objects

- **21 July test:** an empty water bottle, per the operator's account (not recorded in the file).
- **8 September recording:** no object is recorded in any of the files.

## Notes

- `n_slip_events` in the 21 July file counts slip-event ticks across all four fingers (Thumb 13, Ring 7, Index 1, Middle 0), not Thumb alone.
- `force` is the confirmed-contact-gated force the live system displays (0.0 when contact is not confirmed); `raw_force` is the model's ungated output.
