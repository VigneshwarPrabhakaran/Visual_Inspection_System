import cv2
import numpy as np

# Read the image
image = cv2.imread(r"C:\Users\vasum\Downloads\WhatsApp Image 2026-07-07 at 2.06.13 PM.jpeg")   # Replace with your image path

# Create an empty mask
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Approximate conveyor corner points
points = np.array([
    [75, 0],      # Top Left
    [660, 0],     # Top Right
    [710, 594],   # Bottom Right
    [145, 594]    # Bottom Left
], dtype=np.int32)

# Fill the polygon
cv2.fillPoly(mask, [points], 255)

# Apply the mask
result = cv2.bitwise_and(image, image, mask=mask)

# Show results
cv2.imshow("Original", image)
cv2.imshow("Mask", mask)
cv2.imshow("Conveyor Only", result)

# Save the output
cv2.imwrite("conveyor_only.png", result)

cv2.waitKey(0)
cv2.destroyAllWindows()