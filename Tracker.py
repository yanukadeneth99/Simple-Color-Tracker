# Importing libraries
import cv2
import imutils

# Values to parse
redLower = (161, 155, 84)
redUpper = (179, 255, 255)
text = "Nothing Found"

# Initialising Camera
camera = cv2.VideoCapture(0)

while True:

    # Read the Frame from the camera
    (grabbed, frame) = camera.read()

    # Controlling the Framesize
    frame = imutils.resize(frame, width=600)

    # Applying Image Smoothning
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # Converting Image to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Masking the Image inbetween the bounds
    mask = cv2.inRange(hsv, redLower, redUpper)

    # Applying FX for closing any gaps
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Finding the hits
    cnts = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # There is a hit
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)

        # Get the circle coordinates
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Get the Middle point
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            # Draw the circles around the object
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Print Signs according to the radius
            if radius > 250:
                text = "Stop"
            else:
                if (center[0] < 150):
                    text = "Left"
                elif (center[0] > 250):
                    text = "Right"
                elif (radius < 250):
                    text = "Go"
                else:
                    text = "Stop"

        # Render the text onto screen
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the Frame from the camera
    cv2.imshow("Frame", frame)

    # Stop the program when the Q key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the camera and destroy all windows
camera.release()
cv2.destroyAllWindows()
