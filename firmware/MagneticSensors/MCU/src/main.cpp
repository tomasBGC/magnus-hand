/*
  eFlesh 4×5 Multiplexed Sensor Array
  Teensy 4.0 + TCA9546A I2C Multiplexer

  Reads 4 groups of 5 MLX90393 magnetometers (20 total) through a TCA9546A
  multiplexer and streams calibrated data over USB serial.

  Operating mode: single-shot, pipelined across all 4 mux channels.
  All 20 sensors are triggered, then USB send + conversion wait overlap,
  then all 20 results are read back.  ~106 fps at DIG_FILT=4, OSR=0, 1 MHz I2C.

  Burst mode is NOT used — it is incompatible with the I2C mux (sensors
  keep converting while electrically isolated during channel switching,
  producing garbage).

  Hardware:
    - Teensy 4.0
    - TCA9546A I2C mux (default addr 0x70, all A pins to GND)
      - SD0/SC0 → eFlesh board 0 (sensors 0x0C–0x10)
      - SD1/SC1 → eFlesh board 1
      - SD2/SC2 → eFlesh board 2
      - SD3/SC3 → eFlesh board 3

  Serial Protocol (binary, 324 bytes/frame):
    [0xAA] [0x55] [seq_lo] [seq_hi]
    [group0: 5×{t,x,y,z}=80B] ... [group3: 80B]

  Based on original 5X_eflesh_stream.ino and arduino-MLX90393 library.
*/

#include <Wire.h>
#include <MLX90393.h>
#include "handshake.h"
#include "tca.h"
#include "eflesh.h"

// ============================================================================
// Diagnostic toggles (compile-time)
// ============================================================================

/// Set true to read only one mux group each frame (isolates switching issues).
static const bool    DIAG_SINGLE_GROUP_ENABLED = false;
static const uint8_t DIAG_SINGLE_GROUP_INDEX   = 1;

// ============================================================================
// Servo command relay (Feetech SCS0009 via Serial1 / Waveshare adapter)
// ============================================================================
// Shares the same USB serial link the host (EfleshController) uses for
// sensor streaming. Frames are a distinct message type from sensor data:
//   0xFA 0xCE  id  addr  len  data[len]  checksum
// checksum = (id + addr + len + sum(data)) & 0xFF  (frame integrity only,
// distinct from the Feetech packet checksum computed below). Relayed out
// Serial1 as a Feetech/Dynamixel protocol-1.0 WRITE packet:
//   FF FF id payloadLen 0x03 addr data... checksum
enum ServoCmdState { CMD_WAIT_SYNC1, CMD_WAIT_SYNC2, CMD_WAIT_ID, CMD_WAIT_ADDR, CMD_WAIT_LEN, CMD_WAIT_DATA, CMD_WAIT_CHECKSUM };
static ServoCmdState servoCmdState = CMD_WAIT_SYNC1;
static uint8_t servoCmdId = 0, servoCmdAddr = 0, servoCmdLen = 0, servoCmdDataIdx = 0;
static uint8_t servoCmdData[2];

void sendFeetechWrite(uint8_t id, uint8_t addr, const uint8_t* data, uint8_t dataLen) {
  const uint8_t instr = 0x03;  // WRITE
  uint8_t payloadLen = (uint8_t)(dataLen + 3);

  uint8_t sum = (uint8_t)(id + payloadLen + instr + addr);
  for (uint8_t i = 0; i < dataLen; ++i) sum = (uint8_t)(sum + data[i]);
  uint8_t checksum = (uint8_t)~sum;

  Serial1.write((uint8_t)0xFF);
  Serial1.write((uint8_t)0xFF);
  Serial1.write(id);
  Serial1.write(payloadLen);
  Serial1.write(instr);
  Serial1.write(addr);
  Serial1.write(data, dataLen);
  Serial1.write(checksum);
}

// Feeds one incoming USB byte through the servo-command state machine.
// Returns true if the byte belongs to a servo command frame (and so must
// NOT be passed to the ASCII reconnection-handshake scanner below); false
// if the parser is idle and the byte is not a servo-frame sync byte, in
// which case the caller should handle it as a normal handshake byte.
bool feedServoCommandByte(uint8_t b) {
  switch (servoCmdState) {
    case CMD_WAIT_SYNC1:
      if (b == 0xFA) { servoCmdState = CMD_WAIT_SYNC2; return true; }
      return false;
    case CMD_WAIT_SYNC2:
      servoCmdState = (b == 0xCE) ? CMD_WAIT_ID : CMD_WAIT_SYNC1;
      return true;
    case CMD_WAIT_ID:
      servoCmdId = b;
      servoCmdState = CMD_WAIT_ADDR;
      return true;
    case CMD_WAIT_ADDR:
      servoCmdAddr = b;
      servoCmdState = CMD_WAIT_LEN;
      return true;
    case CMD_WAIT_LEN:
      servoCmdLen = b;
      if (servoCmdLen == 0 || servoCmdLen > 2) {
        servoCmdState = CMD_WAIT_SYNC1;  // malformed, resync
        return true;
      }
      servoCmdDataIdx = 0;
      servoCmdState = CMD_WAIT_DATA;
      return true;
    case CMD_WAIT_DATA:
      servoCmdData[servoCmdDataIdx++] = b;
      if (servoCmdDataIdx >= servoCmdLen) servoCmdState = CMD_WAIT_CHECKSUM;
      return true;
    case CMD_WAIT_CHECKSUM: {
      uint8_t sum = (uint8_t)(servoCmdId + servoCmdAddr + servoCmdLen);
      for (uint8_t i = 0; i < servoCmdLen; ++i) sum = (uint8_t)(sum + servoCmdData[i]);
      if (b == sum) {
        sendFeetechWrite(servoCmdId, servoCmdAddr, servoCmdData, servoCmdLen);
      }
      servoCmdState = CMD_WAIT_SYNC1;
      return true;
    }
  }
  return true;  // unreachable
}

// ============================================================================
// Setup
// ============================================================================

void setup() {
  Serial.begin(2000000);
  while (!Serial && millis() < 3000) { /* wait for USB */ }

  Serial1.begin(1000000);  // SCS0009 bus baud (matches AmazingHand firmware/Python examples)

  // --- Handshake ---
  Serial.println(F("=== eFlesh 4x5 Multiplexed Sensor Array ==="));
  Serial.println(F("Waiting for host handshake..."));
  if (!eflesh_handshake(Serial)) {
    Serial.println(F("[WARN] Handshake timed out, proceeding without host."));
  }
  Serial.println(F("Host connected.\n"));

  // --- I2C init @ 1 MHz ---
  Wire.begin();
  Wire.setClock(1000000);
  delay(10);

  // --- Verify TCA9546A ---
  tcaDisable();
  delayMicroseconds(100);
  Wire.beginTransmission(TCA9546A_ADDR);
  if (Wire.endTransmission() == 0) {
    Serial.print(F("TCA9546A found at 0x"));
    Serial.println(TCA9546A_ADDR, HEX);
  } else {
    Serial.println(F("[WARN] TCA9546A not responding — check wiring!"));
  }
  Serial.println();

  // --- Initialise each group ---
  for (uint8_t ch = 0; ch < NUM_GROUPS; ++ch) {
    Serial.print(F("--- Initializing Group "));
    Serial.print(ch);
    Serial.println(F(" (mux channel) ---"));

    if (!tcaSelect(ch)) {
      Serial.print(F("[WARN] Mux select failed for group "));
      Serial.println(ch);
      continue;
    }
    initGroup(ch);
    Serial.println();
  }

  // --- Print summary ---
  Serial.println(F("=== Initialization complete ==="));
  uint8_t activeGroups = 0;
  for (uint8_t ch = 0; ch < NUM_GROUPS; ++ch) {
    if (groups[ch].active) {
      activeGroups++;
      Serial.print(F("Group "));
      Serial.print(ch);
      Serial.print(F(": "));
      Serial.print(groups[ch].count);
      Serial.print(F(" sensors at "));
      for (uint8_t s = 0; s < groups[ch].count; ++s) {
        Serial.print(F("0x"));
        Serial.print(groups[ch].addresses[s], HEX);
        Serial.print(F(" "));
      }
      Serial.println();
    }
  }
  Serial.print(F("Total: "));
  Serial.print(activeGroups);
  Serial.println(F(" active group(s)"));

  if (DIAG_SINGLE_GROUP_ENABLED) {
    Serial.print(F("[DIAG] Single-group mode.  Reading only Group "));
    Serial.println(DIAG_SINGLE_GROUP_INDEX);
  } else {
    Serial.println(F("[DIAG] Reading all groups (pipelined single-shot)."));
  }

  Serial.println(F("\n=== Starting data stream (binary) ===\n"));
  delay(100);
}

// ============================================================================
// Main loop
// ============================================================================

void loop() {
  static uint16_t seq = 0;

  // --- Reconnection handshake (non-blocking) + servo command relay ---
  while (Serial.available()) {
    char c = Serial.read();

    // Servo command frames are binary and distinct from the ASCII
    // handshake below; bytes belonging to one are consumed here and must
    // not reach the handshake scanner (they can contain arbitrary byte
    // values, including ones that look like printable ASCII).
    if (feedServoCommandByte((uint8_t)c)) continue;

    static char hskBuf[12];
    static uint8_t hskIdx = 0;

    if (c == '\n' || c == '\r') {
      if (hskIdx > 0) {
        hskBuf[hskIdx] = '\0';
        if (strcmp(hskBuf, "EFLESH_ACK") == 0) {
          Serial.println(F("EFLESH_RDY"));
          Serial.send_now();
        }
        hskIdx = 0;
        memset(hskBuf, 0, sizeof(hskBuf));
      }
    } else if (c >= 0x20 && c <= 0x7E && hskIdx < 11) {
      hskBuf[hskIdx++] = c;
    } else if (hskIdx >= 11) {
      hskIdx = 0;
      memset(hskBuf, 0, sizeof(hskBuf));
    }
  }

  // -----------------------------------------------------------------------
  // Pipelined single-shot (default)
  // -----------------------------------------------------------------------

  if (!DIAG_SINGLE_GROUP_ENABLED) {
    // Phase 1 — trigger all groups
    for (uint8_t ch = 0; ch < NUM_GROUPS; ++ch) {
      if (!tcaSelect(ch)) { zeroGroup(ch); continue; }
      delayMicroseconds(100);
      triggerGroup(ch);
    }

    // Phase 2 — send previous frame during conversion
    sendFrame(seq++);

    // Phase 3 — wait for conversions (~4.5 ms)
    delayMicroseconds(SINGLE_SHOT_CONV_US);

    // Phase 4 — collect results
    for (uint8_t ch = 0; ch < NUM_GROUPS; ++ch) {
      if (!tcaSelect(ch)) { zeroGroup(ch); continue; }
      delayMicroseconds(100);
      collectGroup(ch);
    }

    return;
  }

  // -----------------------------------------------------------------------
  // Single-group diagnostic
  // -----------------------------------------------------------------------

  for (uint8_t ch = 0; ch < NUM_GROUPS; ++ch) {
    if (ch != DIAG_SINGLE_GROUP_INDEX) { zeroGroup(ch); continue; }

    if (!tcaSelect(ch)) { zeroGroup(ch); continue; }
    delayMicroseconds(100);
    readGroup(ch);
  }

  sendFrame(seq++);
}
