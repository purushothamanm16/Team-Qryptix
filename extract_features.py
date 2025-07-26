import numpy as np
import time
from datetime import datetime # Import datetime for session_start formatting

def extract_features(data):
    # Extract key and mouse logs (fallback to alternative key names)
    keyLogs = data.get('keyLogs') or data.get('keystrokes', [])
    mouseLogs = data.get('mouseLogs') or data.get('mouseMovements', [])

    # Keyboard feature calculations
    keydown_times = []
    keyup_times = []
    key_hold_times = []
    interkey_latencies = []
    last_keyup_time = None

    for entry in keyLogs:
        if entry.get("type") == "keydown":
            keydown_times.append(entry["time"])
        elif entry.get("type") == "keyup":
            keyup_time = entry["time"]
            keyup_times.append(keyup_time)
            if keydown_times:
                hold_time = keyup_time - keydown_times[-1]
                key_hold_times.append(hold_time)
                if last_keyup_time is not None:
                    if keydown_times[-1] > last_keyup_time: # Ensure current keydown is after previous keyup
                        interkey_latencies.append(keydown_times[-1] - last_keyup_time)
                last_keyup_time = keyup_time

    typing_duration = keyLogs[-1]["time"] - keyLogs[0]["time"] if keyLogs else 0
    avg_key_hold = np.mean(key_hold_times) if key_hold_times else 0
    avg_interkey_latency = np.mean(interkey_latencies) if interkey_latencies else 0

    # Mouse feature calculations
    speeds = []
    accel = []
    max_speed = 0
    if len(mouseLogs) > 1:
        for i in range(1, len(mouseLogs)):
            dx = mouseLogs[i]['x'] - mouseLogs[i-1]['x']
            dy = mouseLogs[i]['y'] - mouseLogs[i-1]['y']
            dt = mouseLogs[i]['time'] - mouseLogs[i-1]['time']
            if dt > 0:
                speed = np.sqrt(dx**2 + dy**2) / dt
                speeds.append(speed)
                if i > 1:
                    dv = abs(speed - speeds[-2])
                    accel.append(dv / dt)
                max_speed = max(max_speed, speed)

    avg_mouse_speed = np.mean(speeds) if speeds else 0
    avg_mouse_accel = np.mean(accel) if accel else 0

    # Behavioral extras
    mouse_movements = len(mouseLogs)
    keystrokes = len(keydown_times) # Number of actual keydown events
    paste_detected = int(data.get('pasteDetected', False))
    hover_count = len(data.get('hoverEvents', []))
    scroll_count = len(data.get('scrollEvents', []))
    focus_events = len(data.get('focusEvents', []))
    click_count = data.get('clicks', 0)

    # Session start and duration
    session_start_ms = data.get('startTime', int(time.time() * 1000))
    # Determine end_time_ms based on latest event or current time
    end_time_ms = int(time.time() * 1000) # Default to current time

    if keyLogs:
        end_time_ms = max(end_time_ms, keyLogs[-1]['time'])
    if mouseLogs:
        end_time_ms = max(end_time_ms, mouseLogs[-1]['time'])
    if data.get('hoverEvents'):
        end_time_ms = max(end_time_ms, max([e.get('leaveTime', e['enterTime']) for e in data['hoverEvents']]))
    if data.get('scrollEvents'):
        end_time_ms = max(end_time_ms, max([e['time'] for e in data['scrollEvents']]))
    if data.get('focusEvents'):
        end_time_ms = max(end_time_ms, max([e['time'] for e in data['focusEvents']]))

    session_duration_s = (end_time_ms - session_start_ms) / 1000

    # RETURN ALL 14 NUMERICAL FEATURES IN THE CORRECT ORDER
    return [
        avg_key_hold, avg_interkey_latency, typing_duration,
        avg_mouse_speed, max_speed, avg_mouse_accel,
        mouse_movements, keystrokes, paste_detected,
        hover_count, scroll_count, focus_events, click_count,
        session_duration_s # session_start_ms is NOT returned as a feature, only duration
    ]