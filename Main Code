import cv2
import mediapipe as mp
import pyautogui

def main():
    # Initialize the webcam
    cam = cv2.VideoCapture(0)

    # Initialize the face mesh model
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    # Get the screen size
    screen_w, screen_h = pyautogui.size()

    while True:
        # Read the frame from the webcam
        ret, frame = cam.read()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the frame color to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with the face mesh model
        output = face_mesh.process(rgb_frame)

        # Get the landmarks from the output
        landmark_points = output.multi_face_landmarks

        # Get the frame size
        frame_h, frame_w, _ = frame.shape

        if landmark_points:
            # Get the landmarks for the eyes
            landmarks = landmark_points[0].landmark

            # Draw circles around the eye landmarks and move the mouse pointer
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))

                if id == 1:
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)

            # Get the landmarks for the left eye
            left = [landmarks[145], landmarks[159]]

            # Draw circles around the left eye landmarks
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))

            # If the y-coordinates of the two landmarks are close, click the mouse
            if (left[0].y - left[1].y) < 0.004:
                pyautogui.click()
                pyautogui.sleep(1)

        # Display the frame
        cv2.imshow('Eye Controlled Mouse', frame)

        # Wait for 1 ms before the next frame
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
