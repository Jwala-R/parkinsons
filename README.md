# Parkinson's FoG Project

## Virtual Simulators

### Install dependencies

```bash
pip install -r virtual_sim/requirements.txt
```

For BLE mode: `pip install bleak`
For camera mode: `pip install mediapipe opencv-python`

### Modes

| Mode | Flag | Hardware |
|------|------|----------|
| Demo | `--mode demo` | None — synthetic FoG every 20 s |
| Live | `--mode live` | Arduino Uno/Nano via USB serial |
| BLE  | `--mode ble`  | Arduino Nano 33 BLE Rev2 via Bluetooth |
| Camera | `--mode camera` | Webcam + hand tracking (no Arduino) |

### Quick start (no hardware)

```bash
python virtual_sim/main.py --task eating --mode demo
python virtual_sim/main.py --task whackamole --mode demo
```

### BLE (Nano 33 BLE Rev2)

```bash
python virtual_sim/main.py --task eating --mode ble
python virtual_sim/main.py --task whackamole --mode ble --ble-device FoG-Nano
```

### Camera / hand tracking

```bash
python virtual_sim/main.py --task eating --mode camera
python virtual_sim/main.py --task whackamole --mode camera --camera-id 0
```

Calibration runs automatically on launch — hold your hand still for ~1 second.
To test the camera feed standalone: `python virtual_sim/arduino/camera.py`

### USB serial (original Arduino)

```bash
python virtual_sim/main.py --task eating --mode live --port COM3
```

Press **ESC** to quit.
