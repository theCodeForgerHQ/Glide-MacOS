# Glide

Glide is a lightweight macOS menu bar utility that allows basic mouse and system control using hand gestures captured via the built-in webcam.

The app runs quietly in the background and can be enabled or disabled from the menu bar. When enabled, it tracks a single hand in real time and maps a small set of predefined gestures to mouse and keyboard actions.

This project is intentionally minimal. It focuses on responsiveness and usability rather than configurability or UI complexity.

---

## Features

- macOS menu bar utility (no main window)
- Enable / Disable / Quit controls
- Real-time hand tracking via webcam
- Gesture-based cursor movement
- Gesture-triggered system actions (mouse and keyboard)
- Low overhead, background-friendly design

---

## How It Works

1. The webcam feed is processed in real time to detect and track hand landmarks.
2. Gesture states are derived from landmark positions using deterministic rules (distances, angles, finger states).
3. Recognized gestures are translated into standard macOS mouse and keyboard events.
4. When the app is disabled, all gesture processing and input injection are paused.

No gesture training or calibration is required.

---

## Tech Stack

- **Language:** Python
- **Hand Tracking:** MediaPipe Hands
- **Computer Vision:** OpenCV
- **System Input:** pyautogui / Quartz
- **Menu Bar UI:** rumps

---

## Design Goals

- Minimal user interface
- Fast iteration and low setup cost
- Reasonable real-time responsiveness
- Clear separation between gesture logic and UI
- Small enough to be built and understood in a single pass

---

## Non-Goals

- Full mouse replacement
- Gesture customization UI
- Machine learning model training
- Cross-platform support
- App Store distribution

---

## Usage

- Launch the app from the menu bar.
- Select **Enable** to start gesture tracking.
- Select **Disable** to pause all input.
- Select **Quit** to exit the app.

---

## Notes

This project is intended as a personal exploration of computer vision–based input on macOS. It prioritizes simplicity and restraint over feature completeness.

Gesture mappings, thresholds, and behavior are defined in code and are expected to be adjusted by developers, not end users.
