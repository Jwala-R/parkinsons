"""
virtual_sim/main.py — Entry point for the FoG therapy simulator.

Usage:
    cd parkinsons
    python virtual_sim/main.py

At launch a text menu is shown (keyboard-driven, before Panda3D starts):
    Task:  [E] Eating  [M] Whack-a-Mole
    Mode:  [D] Demo    [L] Live (USB serial)  [B] BLE  [C] Camera
    [Q] Quit

Modes:
    demo   — synthetic data, no hardware required
    live   — real Arduino over USB serial (original Uno/Nano)
    ble    — Arduino Nano 33 BLE Rev2 over Bluetooth (requires bleak)
    camera — webcam + MediaPipe hand tracking (requires mediapipe, opencv-python)

The threshold can be adjusted via the --threshold flag:
    python virtual_sim/main.py --threshold 0.35
"""

import sys
import os
import argparse

# Ensure ml/src is importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ML_ROOT   = os.path.join(_REPO_ROOT, "ml")
for p in [_REPO_ROOT, _ML_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    parser = argparse.ArgumentParser(description="FoG Therapy Simulator")
    parser.add_argument("--task",      choices=["eating", "whackamole"],
                        help="Task to run (skips menu prompt)")
    parser.add_argument("--mode",      choices=["demo", "live", "ble", "camera"],
                        help="Connection mode (skips menu prompt)")
    parser.add_argument("--port",      default=None,
                        help="Serial port for live mode (e.g. COM3 or /dev/ttyACM0)")
    parser.add_argument("--ble-device", default=None, dest="ble_device",
                        help="BLE device name (default: FoG-Nano)")
    parser.add_argument("--camera-id", default=None, type=int, dest="camera_id",
                        help="Camera device index for camera mode (default: 0)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="FoG detection threshold (default 0.4)")
    return parser.parse_args()


def text_menu() -> dict:
    """
    Simple terminal menu (runs before Panda3D window opens).
    Returns config dict.
    """
    from virtual_sim.config import SERIAL_PORT, FOG_THRESHOLD

    print("\n" + "=" * 50)
    print("  FoG Therapy Simulator")
    print("  Parkinson's Neuroplasticity Training")
    print("=" * 50)
    print("  Select task:")
    print("    [E] Eating       — practice using a spoon")
    print("    [M] Whack-a-Mole — tilt wrist to aim, squeeze to whack")
    print()
    print("  Select mode:")
    print("    [D] Demo mode    (no hardware — synthetic data)")
    print("    [L] Live mode    (USB serial Arduino, e.g. Uno/Nano)")
    print("    [B] BLE mode     (Nano 33 BLE Rev2 via Bluetooth)")
    print("    [C] Camera mode  (webcam + hand tracking, no Arduino)")
    print()
    print("  [Q] Quit")
    print("=" * 50)

    task = None
    mode = None

    while task is None:
        c = input("Task [E/M]: ").strip().upper()
        if c == "E":
            task = "eating"
        elif c == "M":
            task = "whackamole"
        elif c == "Q":
            sys.exit(0)
        else:
            print("  Enter E or M.")

    while mode is None:
        c = input("Mode [D/L/B/C]: ").strip().upper()
        if c == "D":
            mode = "demo"
        elif c == "L":
            mode = "live"
        elif c == "B":
            mode = "ble"
        elif c == "C":
            mode = "camera"
        elif c == "Q":
            sys.exit(0)
        else:
            print("  Enter D, L, B, or C.")

    port = SERIAL_PORT
    ble_device = None
    camera_id = None

    if mode == "live":
        entered = input(f"Serial port [{SERIAL_PORT}]: ").strip()
        if entered:
            port = entered
    elif mode == "ble":
        from virtual_sim.config import BLE_DEVICE_NAME
        entered = input(f"BLE device name [{BLE_DEVICE_NAME}]: ").strip()
        if entered:
            ble_device = entered
    elif mode == "camera":
        from virtual_sim.config import CAMERA_DEVICE_ID
        entered = input(f"Camera device ID [{CAMERA_DEVICE_ID}]: ").strip()
        try:
            camera_id = int(entered) if entered else CAMERA_DEVICE_ID
        except ValueError:
            camera_id = CAMERA_DEVICE_ID

    threshold_str = input(f"FoG threshold [{FOG_THRESHOLD}]: ").strip()
    try:
        threshold = float(threshold_str) if threshold_str else FOG_THRESHOLD
    except ValueError:
        threshold = FOG_THRESHOLD

    return {
        "task": task, "mode": mode, "port": port,
        "ble_device": ble_device, "camera_id": camera_id,
        "threshold": threshold,
    }


def main():
    args = parse_args()

    # Resolve config either from CLI args or interactive menu
    if args.task and args.mode:
        from virtual_sim.config import SERIAL_PORT, FOG_THRESHOLD, BLE_DEVICE_NAME, CAMERA_DEVICE_ID
        cfg = {
            "task":       args.task,
            "mode":       args.mode,
            "port":       args.port or SERIAL_PORT,
            "ble_device": args.ble_device or BLE_DEVICE_NAME,
            "camera_id":  args.camera_id if args.camera_id is not None else CAMERA_DEVICE_ID,
            "threshold":  args.threshold or FOG_THRESHOLD,
        }
    else:
        cfg = text_menu()
        if args.threshold:
            cfg["threshold"] = args.threshold
        if args.port:
            cfg["port"] = args.port
        if args.ble_device:
            cfg["ble_device"] = args.ble_device
        if args.camera_id is not None:
            cfg["camera_id"] = args.camera_id

    print(f"\n  Starting: task={cfg['task']}  mode={cfg['mode']}  threshold={cfg['threshold']:.2f}")
    print("  (Press ESC inside the window to quit)\n")

    # ── Communication layer ────────────────────────────────────────────────
    mode = cfg["mode"]
    if mode == "demo":
        from virtual_sim.arduino.mock import MockArduino
        comm = MockArduino()
    elif mode == "live":
        from virtual_sim.arduino.comm import ArduinoComm
        comm = ArduinoComm(cfg["port"], 115200)
    elif mode == "ble":
        from virtual_sim.arduino.ble import ArduinoBLE
        comm = ArduinoBLE(device_name=cfg.get("ble_device"))
    elif mode == "camera":
        from virtual_sim.arduino.camera import CameraHands
        comm = CameraHands(device_id=cfg.get("camera_id"))
    else:
        print(f"[ERROR] Unknown mode: {mode}")
        sys.exit(1)

    comm.connect()
    if mode in ("live", "ble", "camera") and not comm.is_connected:
        if mode == "live":
            print(f"[ERROR] Could not connect to Arduino on {cfg['port']}.")
            print("        Check the port and try again, or run in demo mode.")
        elif mode == "ble":
            print(f"[ERROR] Could not connect to BLE device '{cfg['ble_device']}'.")
            print("        Ensure the Nano 33 BLE Rev2 is powered and advertising.")
            print("        Install bleak with: pip install bleak")
        elif mode == "camera":
            print(f"[ERROR] Could not open camera {cfg['camera_id']}.")
            print("        Check CAMERA_DEVICE_ID in config.py.")
            print("        Install deps with: pip install mediapipe opencv-python")
        sys.exit(1)

    # ── FoG detector ───────────────────────────────────────────────────────
    from virtual_sim.fog.detector import FogDetector
    detector = FogDetector(threshold=cfg["threshold"])

    # ── Haptic controller ──────────────────────────────────────────────────
    from virtual_sim.haptic.feedback import HapticController
    haptic = HapticController(comm)

    # ── Task ───────────────────────────────────────────────────────────────
    if cfg["task"] == "eating":
        from virtual_sim.tasks.eating import EatingTask
        task = EatingTask()
        haptic.set_pattern("triple")
    else:  # whackamole
        from virtual_sim.tasks.whackamole import WhackAMoleTask
        task = WhackAMoleTask()
        haptic.set_pattern("triple")

    # ── Launch Panda3D app ─────────────────────────────────────────────────
    from virtual_sim.sim.app import SimApp
    app = SimApp(
        comm=comm,
        detector=detector,
        haptic=haptic,
        task=task,
        mode=cfg["mode"],
    )
    app.run()


if __name__ == "__main__":
    main()
