# firmware

Microcontroller firmware, built with PlatformIO (each `platformio.ini` selects the board). It was not
built as part of preparing this repository. The PC-side code that talks to it is in `software/`.

| Folder | What it is |
|---|---|
| `MagneticSensors/MCU/` | Teensy 4.0 firmware that reads 20 MLX90393 magnetometers (four boards of five) through a 4-channel I2C multiplexer (STEMMA QT, addressed in firmware as a TCA9546A at 0x70) and streams frames over USB serial. Contains `src/main.cpp` and the libraries `eflesh/` (sensor array), `handshake/` (auto-connect protocol), `tca/` (multiplexer driver) and `MLX90393/` (magnetometer driver, third-party; see `NOTICE.md`). |
| `LoadScale/MCU/` | ATmega328P firmware that reads the HX711 load-cell amplifier and streams force values. Contains `src/main.cpp`. |
