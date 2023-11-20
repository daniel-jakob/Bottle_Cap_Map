import cv2
import numpy as np

def detect_circles(processed_image):


	# Apply Hough transform on the blurred image.
	detected_circles = cv2.HoughCircles(processed_image,
					cv2.HOUGH_GRADIENT, 1, 20, param1 = 100,
				param2 = 26, minRadius = 83, maxRadius = 110)

	# Draw circles that are detected.
	if detected_circles is not None:
		# Convert the circle parameters a, b and r to integers.
		detected_circles = np.uint16(np.around(detected_circles))
		# print(detected_circles)
		img= cv2.imread('data/images/map.jpg', cv2.IMREAD_COLOR)
		for pt in detected_circles[0, :]:
			a, b, r = pt[0], pt[1], pt[2]
			# Draw the circumference of the circle.
			cv2.circle(img, (a, b), r, (0, 255, 0), 2)

			# Draw a small circle (of radius 1) to show the center.
			cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
			cv2.imshow("Detected Circle", img)

	cv2.imwrite("final_image.jpg", img)