import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import time
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = False

ip = ''

# Initialize MediaPipe Pose and webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Calibration variables
is_calibrated = False
calibration_frames = 0
calibration_shoulder_angles = []
calibration_neck_angles = []
shoulder_threshold = 0
neck_threshold = 0
last_alert_time = 0
alert_cooldown = 3  # seconds

def MPU6050_alert():
    # The external URL you want to send the GET request to
    url = f"http://{ip}/alert"
    
    try:
        # Send a GET request to the external URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            print("Alert sent successfully.")
        else:
            print(f"Failed to send alert. Status code: {response.status_code}")
    
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur while sending the request
        print(f"Error sending alert: {e}")

def calculate_angle(a, b, c):
    # Calculate the angle between three points
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # Last
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

def generate_frames():
    global is_calibrated, calibration_frames, calibration_shoulder_angles, calibration_neck_angles
    global shoulder_threshold, neck_threshold, last_alert_time

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    posture_status = "Good Posture"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract key body landmarks
            left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
            right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                              int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
            left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                        int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
            right_ear = (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * frame.shape[0]))
            left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1]),
                        int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]))
            right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1]),
                        int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]))

            # Calculate angles
            shoulder_angle_r = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
            shoulder_angle_l = calculate_angle(right_shoulder, left_shoulder, (left_shoulder[0], 0))
            neck_angle_r = calculate_angle(left_ear, left_shoulder, left_hip)  # Improved neck angle
            neck_angle_l = calculate_angle(right_ear, right_shoulder, right_hip)
            
            shoulder_angle = min(shoulder_angle_l, shoulder_angle_r)
            neck_angle = min(neck_angle_l, neck_angle_r)

            # Optional: Forward head posture detection (normalized coordinates)
            # normalized_displacement = abs(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x -
            #                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x)

            # Calibration logic
            if not is_calibrated and calibration_frames < 25:
                calibration_shoulder_angles.append(shoulder_angle)
                calibration_neck_angles.append(neck_angle)
                calibration_frames += 1
            elif not is_calibrated and calibration_frames >= 25:
                shoulder_threshold = np.mean(calibration_shoulder_angles) - 10
                neck_threshold = np.mean(calibration_neck_angles) - 10
                is_calibrated = True

            # Check posture
            if shoulder_angle < shoulder_threshold or neck_angle < neck_threshold:
                posture_status = "Poor Posture"
                # Show visual alert
                cv2.putText(frame, "⚠️ BAD POSTURE DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.rectangle(frame, (20, 80), (460, 140), (0, 0, 255), -1)
                cv2.putText(frame, "Please sit upright", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                current_time = time.time()
                if current_time - last_alert_time > alert_cooldown:
                    last_alert_time = current_time
            else:
                posture_status = "Good Posture"
                cv2.putText(frame, "Good Posture", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Draw pose landmarks and posture status
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, posture_status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 0) if posture_status == "Good Posture" else (0, 0, 255), 3)

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        # Yield frame to stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

# Route to stream video frames
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_posture', methods=['POST'])
def check_posture():
    # Placeholder to allow interaction with frontend (posture detection logic runs in background)
    return jsonify({"status": "Posture Check Complete"})

@app.route('/recalibrate', methods=['POST'])
def recalibrate():
    global is_calibrated, calibration_frames, calibration_shoulder_angles, calibration_neck_angles
    is_calibrated = False
    calibration_frames = 0
    calibration_shoulder_angles.clear()
    calibration_neck_angles.clear()
    return jsonify({'status': 'Recalibration started.'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    # Let the frontend know camera can stop, actual release already handled inside generate_frames
    return jsonify({'status': 'Camera stopped'})

if __name__ == '__main__':
    app.run(debug=False)
