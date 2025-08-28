import cv2
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Counter
count = 0
stage = None  # "down" or "up"

# Function to calculate angle between 3 points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

cap = cv2.VideoCapture(0)  # front camera (0 usually works)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip so it feels like a mirror
    frame = cv2.flip(frame, 1)

    # Recolor image to RGB for mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Convert back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for elbows and shoulders
        shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Calculate elbow angles
        angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
        angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)

        # Draw angles on arms
        cv2.putText(image, str(int(angle_left)),
                    tuple(np.multiply(elbow_left, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                    )
        cv2.putText(image, str(int(angle_right)),
                    tuple(np.multiply(elbow_right, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                    )

        # Debug prints
        print(f"Left Elbow: {angle_left:.1f}, Right Elbow: {angle_right:.1f}, Stage: {stage}")

        # Push-up logic:
        # When elbows bend (goes below ~90°) = down
        if angle_left < 90 and angle_right < 90:
            stage = "down"
        # When elbows extend (goes above ~160°) = up -> count rep
        if angle_left > 160 and angle_right > 160 and stage == "down":
            stage = "up"
            count += 1
            print(f"REP COUNTED! Total: {count}")

        # Display rep count on screen
        cv2.putText(image, f'Reps: {count}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    except:
        pass

    # Draw landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show
    cv2.imshow('Pushup Counter', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
