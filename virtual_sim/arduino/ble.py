"""
arduino/ble.py — Live BLE communication with Arduino Nano 33 BLE Rev2.

The Nano 33 BLE Rev2 has an onboard LSM6DSOX IMU (accelerometer + gyroscope).
This module connects over BLE using the bleak library and reads the IMU GATT
characteristic that the Arduino firmware exposes.

Expected Arduino BLE firmware behaviour:
  - Advertises with device name "FoG-Nano" (configurable via BLE_DEVICE_NAME)
  - Exposes a notify characteristic (BLE_IMU_CHAR_UUID) that fires at ~60 Hz
  - Each notification payload is a 28-byte little-endian struct:
        float32 ax, ay, az    (m/s^2, 12 bytes)
        float32 gx, gy, gz    (deg/s, 12 bytes)
        float32 pressure      ( FSR 0-1023 mapped to float, 4 bytes)
    Total: 7 x float32 = 28 bytes

  If you use a simpler CSV-over-BLE approach (Arduino prints the same
  "ax,ay,az,gx,gy,gz,pressure" string as the serial firmware), set
  BLE_PAYLOAD_FORMAT = "csv" in config.py and this driver handles both.

Haptic feedback:
  - A writable characteristic (BLE_HAPTIC_CHAR_UUID) accepts a single byte:
      0x00 = off, 0x01-0xFF = on (strength is fixed in firmware or via PWM)
  - Python writes to it from the main thread (non-blocking via asyncio).

Because bleak is async, this class runs a dedicated asyncio event loop in a
background thread.  read_frame() and send_haptic() are synchronous and
thread-safe for the rest of the sim to call as normal.

Installation:
    pip install bleak

Reference sketch (Arduino Nano 33 BLE Rev2):
    #include <ArduinoBLE.h>
    #include <Arduino_LSM6DSOX.h>

    BLEService imuService("12345678-1234-1234-1234-123456789abc");
    BLECharacteristic imuChar("12345678-1234-1234-1234-123456789abd",
                               BLENotify, 28);
    BLECharacteristic hapticChar("12345678-1234-1234-1234-123456789abe",
                                  BLEWrite, 1);

    void setup() {
        IMU.begin();
        BLE.begin();
        BLE.setLocalName("FoG-Nano");
        imuService.addCharacteristic(imuChar);
        imuService.addCharacteristic(hapticChar);
        BLE.addService(imuService);
        BLE.advertise();
    }

    void loop() {
        BLEDevice central = BLE.central();
        if (central && central.connected()) {
            float ax,ay,az,gx,gy,gz;
            IMU.readAcceleration(ax,ay,az);
            IMU.readGyroscope(gx,gy,gz);
            // convert g -> m/s^2
            float buf[7] = {ax*9.81,ay*9.81,az*9.81,gx,gy,gz,analogRead(A0)};
            imuChar.writeValue((byte*)buf, 28);
            if (hapticChar.written()) {
                byte v = hapticChar.value()[0];
                analogWrite(D9, v);
            }
        }
        delay(16); // ~60 Hz
    }
"""

import asyncio
import struct
import threading
import queue
import time
from typing import Optional

# bleak is imported lazily so the rest of the sim works even if it is not installed
_bleak_available = False
try:
    from bleak import BleakClient, BleakScanner
    _bleak_available = True
except ImportError:
    pass

from virtual_sim.config import (
    BLE_DEVICE_NAME,
    BLE_IMU_CHAR_UUID,
    BLE_HAPTIC_CHAR_UUID,
    BLE_PAYLOAD_FORMAT,
    SAMPLING_RATE,
)


class ArduinoBLE:
    """
    Connects to an Arduino Nano 33 BLE Rev2 over Bluetooth Low Energy.

    Drop-in replacement for ArduinoComm: exposes the same
    connect() / disconnect() / read_frame() / send_haptic() interface.

    Thread model:
        - An internal asyncio loop runs in a daemon thread (_loop_thread).
        - Incoming BLE notifications are placed into _frame_queue.
        - read_frame() pops from _frame_queue (non-blocking).
        - send_haptic() posts a coroutine to the async loop.
    """

    def __init__(self, device_name: Optional[str] = None):
        self._device_name = device_name or BLE_DEVICE_NAME
        self._client: Optional["BleakClient"] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=60)
        self._connected = False
        self._connect_error: Optional[str] = None
        self.parse_errors = 0

    # ── Connection ──────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Start the asyncio loop thread and block until BLE connection is
        established (or fails).  Returns True on success.
        """
        if not _bleak_available:
            print("[ArduinoBLE] bleak library not found. Install with: pip install bleak")
            return False

        ready_event = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            args=(ready_event,),
            daemon=True,
        )
        self._loop_thread.start()

        # Wait up to 20 s for the BLE handshake
        if not ready_event.wait(timeout=20.0):
            self._connected = False
            print(f"[ArduinoBLE] Timed out scanning for '{self._device_name}'.")
            return False

        if self._connect_error:
            print(f"[ArduinoBLE] Connection failed: {self._connect_error}")
            return False

        return self._connected

    def disconnect(self):
        if self._loop and self._connected:
            asyncio.run_coroutine_threadsafe(
                self._async_disconnect(), self._loop
            ).result(timeout=5.0)
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Reading ─────────────────────────────────────────────────────────────

    def read_frame(self) -> Optional[dict]:
        """
        Non-blocking pop from the BLE notification queue.
        Returns {"acc": [ax,ay,az], "gyro": [gx,gy,gz], "pressure": float}
        or None if no new data.
        """
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    # ── Writing ─────────────────────────────────────────────────────────────

    def send_haptic(self, strength: int, duration_ms: int = 500):
        """
        Write a single byte to the haptic characteristic.
        strength: 0-255 (0 = off).
        Non-blocking: posted to the async loop.
        """
        if not self._connected or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(
            self._async_send_haptic(strength), self._loop
        )

    # ── Async internals ─────────────────────────────────────────────────────

    def _run_loop(self, ready_event: threading.Event):
        """Entry point for the background asyncio thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_connect(ready_event))
        if self._connected:
            # Keep the loop alive to service haptic writes and notifications
            self._loop.run_until_complete(self._async_keep_alive())

    async def _async_connect(self, ready_event: threading.Event):
        try:
            print(f"[ArduinoBLE] Scanning for '{self._device_name}'...")
            device = await BleakScanner.find_device_by_name(
                self._device_name, timeout=15.0
            )
            if device is None:
                self._connect_error = f"Device '{self._device_name}' not found."
                ready_event.set()
                return

            self._client = BleakClient(device)
            await self._client.connect()
            print(f"[ArduinoBLE] Connected to {device.name} ({device.address})")

            await self._client.start_notify(
                BLE_IMU_CHAR_UUID, self._on_notification
            )
            self._connected = True
        except Exception as e:
            self._connect_error = str(e)
        finally:
            ready_event.set()

    async def _async_keep_alive(self):
        """Hold the loop open until disconnected."""
        while self._connected and self._client and self._client.is_connected:
            await asyncio.sleep(0.5)

    async def _async_disconnect(self):
        if self._client and self._client.is_connected:
            await self._client.stop_notify(BLE_IMU_CHAR_UUID)
            await self._client.disconnect()
        self._connected = False

    async def _async_send_haptic(self, strength: int):
        if self._client and self._client.is_connected:
            try:
                await self._client.write_gatt_char(
                    BLE_HAPTIC_CHAR_UUID,
                    bytes([max(0, min(255, int(strength)))]),
                    response=False,
                )
            except Exception as e:
                print(f"[ArduinoBLE] Haptic write error: {e}")

    def _on_notification(self, _sender, data: bytearray):
        """Called by bleak on the asyncio thread for each IMU notification."""
        frame = self._parse_payload(bytes(data))
        if frame is not None:
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # drop oldest implicitly by ignoring

    def _parse_payload(self, data: bytes) -> Optional[dict]:
        """Parse binary (28-byte struct) or CSV payload."""
        if BLE_PAYLOAD_FORMAT == "csv":
            return self._parse_csv(data)
        return self._parse_binary(data)

    def _parse_binary(self, data: bytes) -> Optional[dict]:
        if len(data) != 28:
            self.parse_errors += 1
            return None
        try:
            vals = struct.unpack("<7f", data)
            return {
                "acc":      [vals[0], vals[1], vals[2]],
                "gyro":     [vals[3], vals[4], vals[5]],
                "pressure": vals[6],
            }
        except struct.error:
            self.parse_errors += 1
            return None

    def _parse_csv(self, data: bytes) -> Optional[dict]:
        try:
            parts = data.decode("ascii", errors="ignore").strip().split(",")
            if len(parts) != 7:
                self.parse_errors += 1
                return None
            vals = [float(p) for p in parts]
            return {
                "acc":      [vals[0], vals[1], vals[2]],
                "gyro":     [vals[3], vals[4], vals[5]],
                "pressure": vals[6],
            }
        except (ValueError, UnicodeDecodeError):
            self.parse_errors += 1
            return None
