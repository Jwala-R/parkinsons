/*
 * firmware_reference.ino
 * Arduino firmware for the FoG therapy wrist sensor.
 *
 * Hardware (Arduino Uno / Nano):
 *   SDA (A4)  → MPU-6050 SDA
 *   SCL (A5)  → MPU-6050 SCL
 *   A0        → FSR pressure sensor (voltage divider with 10k to GND)
 *   D9 (PWM)  → ERM/LRA haptic motor driver IN (e.g. DRV2605 or simple NPN)
 *   USB       → Serial 115200 baud to Python
 *
 * Serial output (60 Hz, ASCII CSV):
 *   ACC_X,ACC_Y,ACC_Z,GYR_X,GYR_Y,GYR_Z,PRESSURE\n
 *   Units: acc in m/s^2, gyro in rad/s, pressure in raw ADC counts (0-1023)
 *
 * Serial input (haptic command from Python):
 *   H<strength>\n   e.g. H200\n  (PWM 0-255)
 *   H0\n            turn off haptic
 */

#include <Wire.h>
#include <MPU6050.h>  // https://github.com/jrowberg/i2cdevlib

MPU6050 mpu;

const int HAPTIC_PIN       = 9;
const int PRESSURE_PIN     = A0;
const int HAPTIC_DURATION  = 500;  // ms — matches Python HAPTIC_DURATION_MS
const int SAMPLE_INTERVAL  = 1000 / 60;  // ~16 ms for 60 Hz

// MPU-6050 sensitivity (default ±2g, ±250deg/s)
const float ACC_SCALE  = 9.80665 / 16384.0;   // LSB to m/s^2
const float GYRO_SCALE = (M_PI / 180.0) / 131.0;  // LSB to rad/s

unsigned long last_sample_ms = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 not found! Check wiring.");
    while (true);
  }
  pinMode(HAPTIC_PIN, OUTPUT);
  analogWrite(HAPTIC_PIN, 0);
}

void loop() {
  unsigned long now = millis();

  // ── Send sensor data at 60 Hz ──────────────────────────────────────────
  if (now - last_sample_ms >= SAMPLE_INTERVAL) {
    last_sample_ms = now;

    int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
    mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);

    float ax = ax_raw * ACC_SCALE;
    float ay = ay_raw * ACC_SCALE;
    float az = az_raw * ACC_SCALE;
    float gx = gx_raw * GYRO_SCALE;
    float gy = gy_raw * GYRO_SCALE;
    float gz = gz_raw * GYRO_SCALE;
    int   pressure = analogRead(PRESSURE_PIN);

    // CSV line: ACC_X,ACC_Y,ACC_Z,GYR_X,GYR_Y,GYR_Z,PRESSURE
    Serial.print(ax, 4); Serial.print(",");
    Serial.print(ay, 4); Serial.print(",");
    Serial.print(az, 4); Serial.print(",");
    Serial.print(gx, 4); Serial.print(",");
    Serial.print(gy, 4); Serial.print(",");
    Serial.print(gz, 4); Serial.print(",");
    Serial.println(pressure);
  }

  // ── Read haptic command from Python ────────────────────────────────────
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("H")) {
      int pwm = cmd.substring(1).toInt();
      if (pwm > 0) {
        analogWrite(HAPTIC_PIN, pwm);
        delay(HAPTIC_DURATION);
        analogWrite(HAPTIC_PIN, 0);
      } else {
        analogWrite(HAPTIC_PIN, 0);
      }
    }
  }
}
