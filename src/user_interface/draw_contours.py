import cv2
import numpy as np

def draw_contours(*contours, img_read='data/images/map.jpg'):
    # Read the image
    img = cv2.imread(img_read, cv2.IMREAD_COLOR)

    # Convert contours to integer type
    contours = [contour.astype(int) for contour in contours]

	# Define a list of colors
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    # Draw each contour on the image with a different color
    for i, (contour, color) in enumerate(zip(contours, colors), start=1):
        cv2.drawContours(img, [contour], -1, color, 2)

    # Display the image with contours
    cv2.imshow("Image with Contours", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()