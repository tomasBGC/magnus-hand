/*
  eFlesh Sensor Array Abstraction
  ===============================
  Data model, discovery, read pipeline, and serial framing for the
  4×5 multiplexed MLX90393 magnetometer array.

  Public API:
    initGroup()      — discover & configure sensors on one mux channel
    triggerGroup()   — start single-shot conversions on one group
    collectGroup()   — read back completed conversions on one group
    readGroup()      — convenience: trigger + wait + collect
    zeroGroup()      — fill group data with zeros
    sendFrame()      — emit 324-byte binary frame over Serial

  Dependencies: tca (TCA9546A mux), MLX90393 (sensor driver)
*/

#ifndef EFLESH_ARRAY_H
#define EFLESH_ARRAY_H

#include <Arduino.h>
#include <Wire.h>
#include <MLX90393.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const uint8_t  NUM_GROUPS           = 4;
static const uint8_t  NUM_SENSORS          = 5;

/// Single-shot conversion time at DIG_FILT=4, OSR=0, RES=2.
/// ~4.15 ms per Table 18 + 350 µs margin for I2C jitter across 20 sensors.
static const uint16_t SINGLE_SHOT_CONV_US  = 4500;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// One sensor's calibrated reading (all fields float).
struct SensorFrame {
  float t, x, y, z;
};

/// One mux channel = one eFlesh board with up to 5 MLX90393 sensors.
struct SensorGroup {
  MLX90393    mlx[NUM_SENSORS];
  SensorFrame data[NUM_SENSORS];
  uint8_t     addresses[NUM_SENSORS];
  bool        ready[NUM_SENSORS];
  uint8_t     triggeredMask;   // bitmask of successfully triggered sensors
  uint8_t     count;           // how many sensors were found
  bool        active;          // true if at least one sensor is ready
};

/// Global array — one entry per mux channel.
extern SensorGroup groups[NUM_GROUPS];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Discover and initialise sensors on the currently selected mux channel.
/// Prints status to Serial.
void initGroup(uint8_t groupIdx);

/// Convenience: trigger + wait + collect for a single group.
void readGroup(uint8_t groupIdx);

/// Start single-shot conversions on all ready sensors in the group.
/// Sets triggeredMask for later collectGroup().
void triggerGroup(uint8_t groupIdx);

/// Read back conversions that were started by triggerGroup().
void collectGroup(uint8_t groupIdx);

/// Fill the group's data with zeros (used for inactive / errored channels).
void zeroGroup(uint8_t groupIdx);

/// Emit a 324-byte binary frame over Serial.
/// [0xAA] [0x55] [seq LE16] [320 bytes of float txyz per sensor]
void sendFrame(uint16_t seq);

#endif // EFLESH_ARRAY_H
