import cv2
import mediapipe as mp
import pyautogui

# Initialize variables for hand coordinates
x1 = y1 = x2 = y2 = 0

# Open webcam
webcam = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module and Drawing Utilities
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

while True:
    # Read a frame from the webcam
    _, image = webcam.read()

    # Flip the image horizontally for a mirrored view
    image = cv2.flip(image, 1)
    frame_height, frame_width, _ = image.shape

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            # Draw landmarks on the image
            drawing_utils.draw_landmarks(image, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                # Save coordinates for thumb (ID: 8) and index finger (ID: 4)
                if id == 8:
                    cv2.circle(
                        image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3
                    )
                    x1 = x
                    y1 = y

                if id == 4:
                    cv2.circle(
                        image, center=(x, y), radius=8, color=(0, 0, 255), thickness=3
                    )
                    x2 = x
                    y2 = y

        # Calculate the distance between thumb and index finger
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 // 4

        # Draw a line between thumb and index finger
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Adjust volume based on hand distance
        if dist > 50:
            pyautogui.press("volumeup")
        else:
            pyautogui.press("volumedown")

    # Display the image with added annotations
    cv2.imshow("Hand volume control using python", image)

    # Check for the 'Esc' key to exit the loop
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()
