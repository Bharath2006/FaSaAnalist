import cv2
import sys
from fer import FER

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Initialize the FER model
emotion_detector = FER()

while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # Detect faces and emotions in the frame
    faces = emotion_detector.detect_emotions(frame)

    # Iterate through detected faces
    for face in faces:
        x, y, w, h = face['box']
        emotions = face['emotions']

        # Get the dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)

        # Draw rectangles around detected faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
