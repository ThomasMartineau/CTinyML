// Minimal serial test harness for the CTinyML-compiled MNIST model.
// Wraps model_forward() (in model.c, alongside this sketch) with a
// trivial serial protocol so tools/serial_infer.py can stream real
// MNIST images to the board and read back predictions.
//
// Protocol, 115200 baud:
//   1. Board sends "READY\n" once, right after setup() completes.
//   2. Host sends: 1 sync byte (0xAA) + 784 int8 bytes -- a single
//      pre-encoded MNIST image, row-major, byte-for-byte what
//      model_forward()'s `input` parameter expects: the compiler's fixed
//      INT8 input encoding (scale 1/255, zero_point -128), i.e. uint8
//      pixel - 128. The board does zero quantization work, it only ever
//      sees already-encoded int8 bytes.
//   3. Board replies with exactly 1 byte: the predicted class (0-9).
//   4. Repeat from step 2 for as many images as the host wants to send.
//
// A leading sync byte (rather than relying purely on the fixed 784-byte
// frame length) lets the board resynchronize on its own if a byte is
// ever dropped or corrupted, instead of getting permanently misaligned.

#include <stdint.h>

// model_forward() is defined in model.c (plain C); this sketch compiles
// as C++, so it needs an explicit extern "C" declaration to link against
// it correctly.
extern "C" void model_forward(const int8_t *input, uint8_t *out_class);

#define IMG_BYTES 784
#define SYNC_BYTE 0xAA

// Static, not a stack/local array -- matches this whole project's
// locked "no dynamic allocation, static buffers" design (see
// buf_a/buf_b in model.c).
static int8_t input_buf[IMG_BYTES];

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    // Only matters on native-USB boards; a no-op on the Mega 2560's
    // UART-based Serial, harmless either way.
  }
  Serial.setTimeout(2000);
  Serial.println("READY");
}

void loop() {
  if (Serial.available() <= 0) {
    return;
  }

  int b = Serial.read();
  if (b != SYNC_BYTE) {
    return;  // not a frame start -- ignore stray bytes until we see one
  }

  size_t got = Serial.readBytes((char *)input_buf, IMG_BYTES);
  if (got != IMG_BYTES) {
    return;  // incomplete frame (read timed out) -- drop it and resync
  }

  uint8_t out_class = 0;
  model_forward(input_buf, &out_class);

  Serial.write(out_class);
}
