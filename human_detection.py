import cv2
import numpy as np

class HumanDetectionTracking:
    def __init__(self, video_path):
        self.video_path = video_path
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.cap = cv2.VideoCapture(video_path)

    def detect_and_track(self):
        while True:
            # Read a frame from the video
            ret, frame = self.cap.read()
            if not ret:
                break  # Exit if no frame is captured (end of video)

            # Resize frame for processing
            frame_resized = cv2.resize(frame, (640, 480))

            # Detect humans in the frame
            humans, _ = self.hog.detectMultiScale(frame_resized, winStride=(8, 8))

            # Draw bounding boxes around detected humans
            for (x, y, w, h) in humans:
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the frame with human detection
            cv2.imshow('Human Detection and Tracking', frame_resized)

            # Wait for 'q' key to exit
            if cv2.waitKey(1) == ord('q'):
                break

        # Release resources and close windows
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path =r"C:\Users\DELL\Desktop\human_detection1.mp4" # Replace with the path to your video
    tracker = HumanDetectionTracking(video_path)
    tracker.detect_and_track()