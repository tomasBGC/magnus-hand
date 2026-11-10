/*
  TCA9546A I2C Multiplexer Driver — Implementation
*/

#include "tca.h"

bool tcaSelect(uint8_t channel) {
  const uint8_t desiredMask = (1 << channel);

  for (uint8_t attempt = 0; attempt < 3; ++attempt) {
    Wire.beginTransmission(TCA9546A_ADDR);
    Wire.write(desiredMask);
    if (Wire.endTransmission() != 0) {
      delayMicroseconds(50);
      continue;
    }

    // Read back control register to verify channel latch.
    if (Wire.requestFrom((uint8_t)TCA9546A_ADDR, (uint8_t)1) == 1) {
      const uint8_t reg = Wire.read();
      if (reg == desiredMask) {
        return true;
      }
    }

    delayMicroseconds(50);
  }

  return false;
}

void tcaDisable() {
  Wire.beginTransmission(TCA9546A_ADDR);
  Wire.write(0x00);
  Wire.endTransmission();
}
