import cv2
import numpy as np

img = cv2.imread("../../Dataset/sample_000000/observation_start/depth_image.jpg")
print(img.shape)  # Print image shape
cv2.imshow("original", img)

# Cropping an image
cropped_image = img[191:725, 1035:1220]

# Display cropped image
cv2.imshow("cropped", cropped_image)

# Save the cropped image
cv2.imwrite("Cropped Image.jpg", cropped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()