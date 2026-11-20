# print

Print files for one magnetic fingertip. Checksums are in `SHA256SUMS.txt`. The hand base, shells and other parts of the
AmazingHand are not included; get them from the upstream repository,
https://github.com/pollen-robotics/AmazingHand.

## Files

| File | What it is |
|---|---|
| `Fingertip_V3_0.002_9.0_with_pouches.stl` | The soft TPU lattice of the fingertip with the five magnet pouches built in (42.8 MB). Print this in TPU. |
| `Fingertip_V3_0.002_9.0_with_pouches.gcode.3mf` | Bambu Studio project, already sliced, for the TPU lattice. It holds the print settings listed below. |
| `Distal_V3.stl` | The rigid PLA part: the magnetometer slot (with screw slots for the PCB) and the skeletal adaptor joint that fixes the fingertip to the AmazingHand finger. |
| `Fingertip_V3.stl` | The fingertip shell, an input to the generator in `hardware/generator/`. |
| `magnet_pouches.stl` | The magnet pouch geometry, an input to the generator. |

No sliced project for the PLA part is included.

## Print settings

**TPU lattice** (from the `.gcode.3mf`, except the orientation, which is from the dissertation):

| Setting | Value |
|---|---|
| Printer | Bambu Lab P1S, 0.4 mm nozzle |
| Material | TPU 95A (Bambu Lab TPU 95A profile) |
| Layer height | 0.12 mm (0.2 mm first layer) |
| Supports | Tree supports (manual) |
| Filament | 6.48 g, including supports |
| Predicted print time | 5,287 s (about 1 h 28 min) |
| Orientation | Printed inverted |
| Pause | A pause (`M400 U1`) at a layer height of 17.12 mm, at the pouch layer, to insert the magnets |

**Rigid PLA part** (dissertation values; no sliced project is included):

| Setting | Value |
|---|---|
| Material | PLA Basic |
| Layer height | 0.12 mm |
| Infill | 100% |
| Orientation | Printed vertically inverted, with normal supports |
| Filament | 11.71 g |
| Print time | about 1 h 10 min |

The bill of materials and per-fingertip cost are in `../BOM.md`.

## Magnet pouch size

Magnets: N52 neodymium, 5 mm diameter and 2 mm thick, five per fingertip. The pouches in the shipped
lattice STL are sized for these magnets. The defaults in `../generator/` are only starting values; the
generator's pouch table sets the actual diameter and thickness.

## Assembly order

1. Print the TPU lattice. At the pause, insert five N52 magnets, the four outer magnets with the same polarity and the centre
   magnet reversed relative to them, then resume the print.
2. Print the rigid PLA part.
3. Bond the two sections together with cyanoacrylate.
4. Screw the magnetometer PCB into its slot.
5. Screw the fingertip onto the AmazingHand finger.

## Licence

These parts are licensed under CC BY 4.0 (`../../LICENSE-HARDWARE-DOCS-DATA.md`). The fingertip mount is
derived from the AmazingHand mechanical design (Pollen Robotics, CC BY 4.0). See `../../NOTICE.md`.
