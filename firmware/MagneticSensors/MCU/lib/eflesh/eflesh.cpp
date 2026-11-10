/*
  eFlesh Sensor Array Abstraction — Implementation
*/

#include "eflesh.h"
#include "tca.h"

// ============================================================================
// Internal helpers — address discovery & ordering
// ============================================================================

/// Known MLX90393 address patterns.
static const uint8_t ADDR_SET_CONSEC[NUM_SENSORS] = {0x0C, 0x0D, 0x0E, 0x0F, 0x10};
static const uint8_t ADDR_SET_WHITE[NUM_SENSORS]  = {0x0C, 0x10, 0x11, 0x12, 0x13};

static void scanI2C(uint8_t* found, uint8_t& count) {
  count = 0;
  for (uint8_t addr = 0x08; addr <= 0x77; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      found[count++] = addr;
      if (count >= 16) break;
    }
  }
}

static bool isMlx90393Address(uint8_t addr) {
  return (addr >= 0x0C && addr <= 0x13);
}

static bool hasExactSet(const uint8_t* found, uint8_t count,
                         const uint8_t* pattern) {
  uint8_t hits = 0;
  for (uint8_t p = 0; p < NUM_SENSORS; ++p) {
    for (uint8_t i = 0; i < count; ++i) {
      if (found[i] == pattern[p]) { hits++; break; }
    }
  }
  return (hits == NUM_SENSORS);
}

static void sortAscending(uint8_t* arr, uint8_t n) {
  for (uint8_t i = 0; i < n; ++i)
    for (uint8_t j = i + 1; j < n; ++j)
      if (arr[j] < arr[i]) {
        uint8_t t = arr[i]; arr[i] = arr[j]; arr[j] = t;
      }
}

/// Match discovered I2C addresses to a known MLX90393 pattern.
/// Fills *ordered* with up to NUM_SENSORS addresses, sets *orderedCount*.
static void chooseAddresses(const uint8_t* found, uint8_t count,
                            uint8_t* ordered, uint8_t& orderedCount) {
  uint8_t mlxFound[NUM_SENSORS * 2] = {0};
  uint8_t mlxCount = 0;

  for (uint8_t i = 0; i < count; ++i) {
    if (isMlx90393Address(found[i])) {
      mlxFound[mlxCount++] = found[i];
      if (mlxCount >= (NUM_SENSORS * 2)) break;
    }
  }

  orderedCount = 0;

  if (mlxCount >= NUM_SENSORS) {
    if (hasExactSet(mlxFound, mlxCount, ADDR_SET_CONSEC)) {
      for (uint8_t i = 0; i < NUM_SENSORS; ++i)
        ordered[i] = ADDR_SET_CONSEC[i];
      orderedCount = NUM_SENSORS;
      return;
    }
    if (hasExactSet(mlxFound, mlxCount, ADDR_SET_WHITE)) {
      ordered[0] = 0x0C;
      ordered[1] = 0x11;
      ordered[2] = 0x12;
      ordered[3] = 0x13;
      ordered[4] = 0x10;
      orderedCount = NUM_SENSORS;
      return;
    }
  }

  // Fallback: sort ascending
  uint8_t tmp[NUM_SENSORS * 2];
  uint8_t n = mlxCount;
  for (uint8_t i = 0; i < n; ++i) tmp[i] = mlxFound[i];
  sortAscending(tmp, n);

  orderedCount = (n >= NUM_SENSORS) ? NUM_SENSORS : n;
  for (uint8_t i = 0; i < orderedCount; ++i)
    ordered[i] = tmp[i];
}

// ============================================================================
// Global state
// ============================================================================

SensorGroup groups[NUM_GROUPS];

// ============================================================================
// Group initialisation (mux channel must already be selected)
// ============================================================================

void initGroup(uint8_t groupIdx) {
  SensorGroup& g = groups[groupIdx];
  g.count = 0;
  g.active = false;
  for (uint8_t i = 0; i < NUM_SENSORS; ++i) {
    g.ready[i] = false;
  }

  delayMicroseconds(200);   // settle after mux switch

  // --- I2C scan ---
  uint8_t found[16] = {0};
  uint8_t foundCount = 0;
  scanI2C(found, foundCount);

  Serial.print(F("  Group "));
  Serial.print(groupIdx);
  Serial.print(F(" I2C scan: found "));
  Serial.print(foundCount);
  Serial.print(F(" device(s) → "));
  for (uint8_t i = 0; i < foundCount; ++i) {
    Serial.print(F("0x"));
    Serial.print(found[i], HEX);
    Serial.print(F(" "));
  }
  Serial.println();

  // --- Match addresses ---
  uint8_t ordered[NUM_SENSORS] = {0};
  uint8_t orderedCount = 0;
  chooseAddresses(found, foundCount, ordered, orderedCount);

  if (orderedCount == 0) {
    Serial.print(F("  Group "));
    Serial.print(groupIdx);
    Serial.println(F(" → NO SENSORS FOUND, skipping."));
    return;
  }

  g.count = orderedCount;
  uint8_t readyCount = 0;

  // --- Initialise each sensor ---
  for (uint8_t i = 0; i < orderedCount; ++i) {
    g.addresses[i] = ordered[i];
    byte status = g.mlx[i].begin(ordered[i], -1, Wire);

    Serial.print(F("    Sensor["));
    Serial.print(i);
    Serial.print(F("] @ 0x"));
    Serial.print(ordered[i], HEX);
    Serial.print(F(" init="));
    Serial.print(status, HEX);

    if (status == MLX90393::STATUS_OK) {
      g.mlx[i].setGainSel(0x1);
      g.mlx[i].setResolution(0x2, 0x2, 0x2);
      g.mlx[i].setDigitalFiltering(0x4);

      // --- Chip-specific overflow-guard override: Ring (group 2), sensor
      // slot 3 (S4, 0-indexed) X-axis only. See model_v1/
      // HARDWARE_NOTE_S3Z_OVERFLOW.md's "Root cause identified" section
      // (2026-08-31): this ONE chip's X-axis quiescent raw ADC value sits
      // close enough to the +-32768-count digital rail at the otherwise-
      // uniform RES=2 setting to wrap on ordinary ambient noise, glitching
      // 100% of ticks. RES is a purely digital decimation/scaling choice
      // (unlike GAIN_SEL, which is an analog front-end stage shared across
      // a chip's 3 axes) -- lowering it rescales only THIS axis's raw
      // output, so it cannot affect any other chip or this chip's own Y/Z
      // (independently settable via the same RES_XYZ register). Dropping
      // RES 2->0 (a 4x reduction, (1<<2) -> (1<<0)) shrinks both signal and
      // DC offset by the same factor, pulling the offset comfortably clear
      // of the rail at the cost of coarser resolution on this one axis.
      if (groupIdx == 2 && i == 3) {
        g.mlx[i].setResolution(0x0, 0x2, 0x2);
        Serial.print(F("      [Ring S4 X-axis: RES 2->0, overflow-guard fix applied] "));
      }

      g.ready[i] = true;
      readyCount++;
      Serial.println(F(" OK (single-shot)"));
    } else {
      Serial.println(F(" FAIL"));
    }
  }

  // Zero-fill unused slots
  for (uint8_t i = orderedCount; i < NUM_SENSORS; ++i) {
    g.data[i] = {0.0f, 0.0f, 0.0f, 0.0f};
  }

  g.count = readyCount;
  if (readyCount > 0) {
    g.active = true;
  }
}

// ============================================================================
// Read pipeline
// ============================================================================

void readGroup(uint8_t groupIdx) {
  SensorGroup& g = groups[groupIdx];
  if (!g.active) {
    zeroGroup(groupIdx);
    return;
  }
  triggerGroup(groupIdx);
  delayMicroseconds(SINGLE_SHOT_CONV_US);
  collectGroup(groupIdx);
}

void triggerGroup(uint8_t groupIdx) {
  SensorGroup& g = groups[groupIdx];
  g.triggeredMask = 0;
  if (!g.active) return;

  const uint8_t zyxt = (MLX90393::X_FLAG | MLX90393::Y_FLAG |
                        MLX90393::Z_FLAG | MLX90393::T_FLAG);

  for (uint8_t i = 0; i < NUM_SENSORS; ++i) {
    if (!g.ready[i]) continue;
    byte st = g.mlx[i].startMeasurement(zyxt);
    if (g.mlx[i].isOK(st)) {
      g.triggeredMask |= (1 << i);
    }
  }
}

void collectGroup(uint8_t groupIdx) {
  SensorGroup& g = groups[groupIdx];
  if (!g.active) {
    zeroGroup(groupIdx);
    return;
  }

  const uint8_t zyxt = (MLX90393::X_FLAG | MLX90393::Y_FLAG |
                        MLX90393::Z_FLAG | MLX90393::T_FLAG);

  for (uint8_t i = 0; i < NUM_SENSORS; ++i) {
    if (!(g.triggeredMask & (1 << i))) {
      g.data[i] = {0.0f, 0.0f, 0.0f, 0.0f};
      continue;
    }
    MLX90393::txyzRaw raw;
    byte st = g.mlx[i].readMeasurement(zyxt, raw);
    if (g.mlx[i].isOK(st)) {
      MLX90393::txyz conv = g.mlx[i].convertRaw(raw);
      g.data[i].t = conv.t;
      g.data[i].x = conv.x;
      g.data[i].y = conv.y;
      g.data[i].z = conv.z;
    } else {
      g.data[i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }
  }
}

void zeroGroup(uint8_t groupIdx) {
  for (uint8_t i = 0; i < NUM_SENSORS; ++i) {
    groups[groupIdx].data[i] = {0.0f, 0.0f, 0.0f, 0.0f};
  }
}

// ============================================================================
// Serial frame output
// ============================================================================

void sendFrame(uint16_t seq) {
  Serial.write(0xAA);
  Serial.write(0x55);
  Serial.write((uint8_t)(seq & 0xFF));
  Serial.write((uint8_t)((seq >> 8) & 0xFF));

  for (uint8_t g = 0; g < NUM_GROUPS; ++g) {
    for (uint8_t s = 0; s < NUM_SENSORS; ++s) {
      SensorFrame& d = groups[g].data[s];
      Serial.write((uint8_t*)&d, sizeof(SensorFrame));
    }
  }

  Serial.send_now();
}
