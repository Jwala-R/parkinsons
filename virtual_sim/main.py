"""
virtual_sim/main.py — Entry point for the FoG therapy simulator.

Usage:
    cd parkinsons
    python virtual_sim/main.py

At launch a text menu is shown (keyboard-driven, before Panda3D starts):
    [W] Walking Task
    [E] Eating Task
    [D] Demo mode (no Arduino)
    [L] Live mode (real Arduino)
    [Q] Quit

In live mode you will be prompted for the serial port (default COM3).
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
    parser.add_argument("--task",      choices=["walking", "eating"],
                        help="Task to run (skips menu prompt)")
    parser.add_argument("--mode",      choices=["demo", "live"],
                        help="Connection mode (skips menu prompt)")
    parser.add_argument("--port",      default=None,
                        help="Serial port (e.g. COM3 or /dev/ttyACM0)")
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
    print("    [W] Walking — walk down a corridor")
    print("    [E] Eating  — practice using a spoon")
    print()
    print("  Select mode:")
    print("    [D] Demo mode (no Arduino needed)")
    print("    [L] Live mode (real Arduino via Serial)")
    print()
    print("  [Q] Quit")
    print("=" * 50)

    task = None
    mode = None

    while task is None:
        c = input("Task [W/E]: ").strip().upper()
        if c == "W":
            task = "walking"
        elif c == "E":
            task = "eating"
        elif c == "Q":
            sys.exit(0)
        else:
            print("  Enter W or E.")

    while mode is None:
        c = input("Mode [D/L]: ").strip().upper()
        if c == "D":
            mode = "demo"
        elif c == "L":
            mode = "live"
        elif c == "Q":
            sys.exit(0)
        else:
            print("  Enter D or L.")

    port = SERIAL_PORT
    if mode == "live":
        entered = input(f"Serial port [{SERIAL_PORT}]: ").strip()
        if entered:
            port = entered

    threshold_str = input(f"FoG threshold [{FOG_THRESHOLD}]: ").strip()
    try:
        threshold = float(threshold_str) if threshold_str else FOG_THRESHOLD
    except ValueError:
        threshold = FOG_THRESHOLD

    return {"task": task, "mode": mode, "port": port, "threshold": threshold}


def main():
    args = parse_args()

    # Resolve config either from CLI args or interactive menu
    if args.task and args.mode:
        from virtual_sim.config import SERIAL_PORT, FOG_THRESHOLD
        cfg = {
            "task":      args.task,
            "mode":      args.mode,
            "port":      args.port or SERIAL_PORT,
            "threshold": args.threshold or FOG_THRESHOLD,
        }
    else:
        cfg = text_menu()
        if args.threshold:
            cfg["threshold"] = args.threshold
        if args.port:
            cfg["port"] = args.port

    print(f"\n  Starting: task={cfg['task']}  mode={cfg['mode']}  "
          f"port={cfg['port']}  threshold={cfg['threshold']:.2f}")
    print("  (Press ESC inside the window to quit)\n")

    # ── Communication layer ────────────────────────────────────────────────
    if cfg["mode"] == "demo":
        from virtual_sim.arduino.mock import MockArduino
        comm = MockArduino()
    else:
        from virtual_sim.arduino.comm import ArduinoComm
        comm = ArduinoComm(cfg["port"], 115200)

    comm.connect()
    if cfg["mode"] == "live" and not comm.is_connected:
        print(f"[ERROR] Could not connect to Arduino on {cfg['port']}.")
        print("        Check the port and try again, or run in demo mode.")
        sys.exit(1)

    # ── FoG detector ───────────────────────────────────────────────────────
    from virtual_sim.fog.detector import FogDetector
    detector = FogDetector(threshold=cfg["threshold"])

    # ── Haptic controller ──────────────────────────────────────────────────
    from virtual_sim.haptic.feedback import HapticController
    haptic = HapticController(comm)

    # ── Task ───────────────────────────────────────────────────────────────
    if cfg["task"] == "walking":
        from virtual_sim.tasks.walking import WalkingTask
        task = WalkingTask()
    else:
        from virtual_sim.tasks.eating import EatingTask
        task = EatingTask()
        haptic.set_pattern("triple")   # rhythmic cueing for eating

    # ── Launch Panda3D app ─────────────────────────────────────────────────
    from virtual_sim.sim.app import SimApp
    app = SimApp(
        comm=comm,
        detector=detector,
        haptic=haptic,
        task=task,
        live_mode=(cfg["mode"] == "live"),
    )
    app.run()


if __name__ == "__main__":
    main()
