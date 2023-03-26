import cv2
import numpy as np

# Load the image
image = cv2.imread('GreenScreen.jpg')

# Define the lower and upper boundaries of the green color in HSV color space
lower_green = np.array([40, 100, 100])
upper_green = np.array([75, 255, 255])

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a mask using the green color range
mask = cv2.inRange(hsv, lower_green, upper_green)

# Apply morphological operations to the mask to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find the contours of the green screen
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a new image with only the green screen
green_screen = np.zeros_like(image, np.uint8)
cv2.drawContours(green_screen, contours, -1, (0, 255, 0), cv2.FILLED)

# Save the new image to disk
cv2.imwrite('GreenScreenP.jpg', green_screen)
