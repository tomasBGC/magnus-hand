/*
  Handshake Protocol Implementation
  See handshake.h for protocol documentation.
*/

#include "handshake.h"

bool eflesh_handshake(Stream &stream) {
    char buf[HANDSHAKE_BUF_SIZE];
    uint8_t bufIdx = 0;
    memset(buf, 0, sizeof(buf));
    const size_t ackLen = strlen(HANDSHAKE_KEY_ACK);

    unsigned long lastBroadcast = 0;
    unsigned long start          = millis();

    while (true) {
        // --- Check for timeout ---
#if HANDSHAKE_TIMEOUT_MS > 0
        if (millis() - start > HANDSHAKE_TIMEOUT_MS) {
            stream.println(F("[HSK] Timeout — no ACK received."));
            return false;
        }
#endif

        // --- Broadcast handshake key periodically ---
        if (millis() - lastBroadcast >= HANDSHAKE_INTERVAL_MS) {
            stream.println(F(HANDSHAKE_KEY_HSK));
            lastBroadcast = millis();
        }

        // --- Read incoming bytes looking for ACK ---
        while (stream.available()) {
            char c = (char)stream.read();

            // Keep delimiters from poisoning partial token matching.
            if (c == '\n' || c == '\r') {
                continue;
            }

            if (c >= 0x20 && c <= 0x7E && bufIdx < HANDSHAKE_BUF_SIZE - 1) {
                buf[bufIdx++] = c;
                buf[bufIdx] = '\0';
            } else if (c >= 0x20 && c <= 0x7E) {
                // Overflow: keep a sliding window so token matching still works.
                memmove(buf, buf + 1, HANDSHAKE_BUF_SIZE - 2);
                buf[HANDSHAKE_BUF_SIZE - 2] = c;
                buf[HANDSHAKE_BUF_SIZE - 1] = '\0';
                bufIdx = HANDSHAKE_BUF_SIZE - 1;
            } else if (bufIdx >= HANDSHAKE_BUF_SIZE - 1) {
                // Overflow — discard
                bufIdx = 0;
                memset(buf, 0, sizeof(buf));
            }

            // Accept ACK token anywhere in the incoming stream.
            if (strstr(buf, HANDSHAKE_KEY_ACK) != nullptr) {
                // Send RDY twice to reduce chance of host-side miss during reconnect.
                stream.println(F(HANDSHAKE_KEY_RDY));
                stream.println(F(HANDSHAKE_KEY_RDY));
                stream.flush();
                return true;
            }

            // Keep only a short suffix needed for future substring matches.
            if (bufIdx > ackLen) {
                const size_t keep = (ackLen > 0) ? (ackLen - 1) : 0;
                if (keep > 0) {
                    memmove(buf, buf + (bufIdx - keep), keep);
                }
                bufIdx = (uint8_t)keep;
                buf[bufIdx] = '\0';
            }
        }
    }
}
