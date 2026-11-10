/*
  TCA9546A I2C Multiplexer Driver
  ================================
  Controls the TCA9546A 4-channel I2C switch used to multiplex
  4 eFlesh sensor boards onto a single I2C bus.

  - tcaSelect(channel): enables one channel (0–3), verifies the latch
  - tcaDisable(): disconnects all channels

  Default address 0x70 (all A0/A1/A2 pins to GND).
*/

#ifndef EFLESH_TCA_H
#define EFLESH_TCA_H

#include <Arduino.h>
#include <Wire.h>

#define TCA9546A_ADDR  0x70

/// Select a single mux channel (0–3).  Returns true on success.
bool tcaSelect(uint8_t channel);

/// Disable all mux channels (bus isolated).
void tcaDisable();

#endif // EFLESH_TCA_H
