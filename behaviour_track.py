import cv2
import numpy as np

# Load the pre-trained faculty member detection model (replace with your model loading code)
model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables for behavior tracking
tracking_started = False
track_window = None
roi_hist = None

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not tracking_started:
        # Perform faculty member detection
        faces = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Select the largest detected face as the ROI (Region of Interest)
            (x, y, w, h) = max(faces, key=lambda f: f[2])
            track_window = (x, y, w, h)

            # Extract the ROI for tracking
            roi = frame[y:y + h, x:x + w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Create a mask for histogram calculation
            mask = cv2.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))

            # Calculate the histogram of the ROI
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            # Set up the termination criteria for the mean shift algorithm
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

            tracking_started = True

    else:
        # Apply meanshift to track the ROI
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply meanshift algorithm to find the new position of the ROI
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw the tracked region on the frame
        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check for suspicious behavior (e.g., unusual movement, interactions)
if tracking_started:
    # Calculate the centroid of the tracked region
    centroid_x = int(track_window[0] + track_window[2] / 2)
    centroid_y = int(track_window[1] + track_window[3] / 2)

    # Define a region of interest (ROI) for detecting interactions
    interaction_roi = frame[max(0, centroid_y - 50):min(centroid_y + 50, frame.shape[0]),
                      max(0, centroid_x - 50):min(centroid_x + 50, frame.shape[1])]

    # Perform interaction detection on the ROI (e.g., using another model or algorithm)
    interactions = detect_interactions(interaction_roi)

    if len(interactions) > 0:
        # Suspicious behavior detected
        for interaction in interactions:
            action = interaction["action"]
            confidence = interaction["confidence"]

            # Perform appropriate actions based on the detected behavior
            if action == "stealing":
                send_alert_email("Suspicious behavior: stealing detected!")
                display_text(frame, "Stealing Detected", centroid_x, centroid_y)
            elif action == "damaging":
                send_alert_email("Suspicious behavior: damaging equipment detected!")
                display_text(frame, "Damaging Equipment", centroid_x, centroid_y)
            
            # You can add more behavior types and actions as per your requirements

        # Reset tracking to detect a new faculty member
        tracking_started = False
        track_window = None
        roi_hist = None


        # Check for suspicious behavior (e.g., unusual movement, interactions)
        # Implement your behavior tracking logic here
        # ...

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
