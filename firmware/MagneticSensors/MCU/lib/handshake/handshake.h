/*
  Handshake Protocol for eFlesh Sensor Array
  ==========================================
  Ensures the PC-side application connects to the correct device by
  exchanging short ASCII keys over USB serial before data streaming.

  Protocol (MCU side):
    1. MCU broadcasts "EFLESH_HSK\n" every HANDSHAKE_INTERVAL_MS
    2. On receiving "EFLESH_ACK\n" from PC, MCU responds "EFLESH_RDY\n"
    3. MCU returns true — caller proceeds to normal text header + binary stream

  Usage in main.cpp:
    #include "handshake.h"
    void setup() {
        Serial.begin(2000000);
        while (!Serial && millis() < 3000) {}
        if (eflesh_handshake(Serial)) {
            // proceed to init sensors, stream data
        }
    }
*/

#ifndef EFLESH_HANDSHAKE_H
#define EFLESH_HANDSHAKE_H

#include <Arduino.h>
#include <Stream.h>

// ----- Configuration -----

/// Interval (ms) between handshake broadcasts.
#define HANDSHAKE_INTERVAL_MS  150

/// Timeout (ms) for the entire handshake.  0 = wait forever.
#define HANDSHAKE_TIMEOUT_MS   0

/// Maximum characters to buffer while looking for the ACK key.
#define HANDSHAKE_BUF_SIZE     64

// ----- Protocol strings -----

/// MCU → PC : "I am an eFlesh device"
#define HANDSHAKE_KEY_HSK      "EFLESH_HSK"

/// PC → MCU : "I acknowledge you"
#define HANDSHAKE_KEY_ACK      "EFLESH_ACK"

/// MCU → PC : "Handshake complete, data follows"
#define HANDSHAKE_KEY_RDY      "EFLESH_RDY"

// ============================================================================
/// Perform the eFlesh handshake on the given stream (usually Serial).
///
/// @param  stream   The serial stream (Serial, Serial1, etc.)
/// @return true if handshake succeeded, false on timeout.
// ============================================================================
bool eflesh_handshake(Stream &stream);

#endif // EFLESH_HANDSHAKE_H
