import cv2
import numpy as np

def detect_circles(processed_image, picutre_out_filename=None, txt_out_filename=None):
	# If processed_image is a string, assume it's a filename and read the numpy array from the file
	if isinstance(processed_image, str):
		processed_image = np.loadtxt(processed_image, dtype=int)

	# Apply Hough transform on the blurred image.
	detected_circles = cv2.HoughCircles(
		processed_image,
		cv2.HOUGH_GRADIENT,
		1,
		20,
		param1=100,
		param2=26,
		minRadius=83,
		maxRadius=110,
	)

	# Draw circles that are detected.
	if detected_circles is not None:
		# Convert the circle parameters a, b and r to integers.
		detected_circles = np.uint16(np.around(detected_circles))

		detected_circles = detected_circles[0, :]

		if txt_out_filename:
				np.savetxt(txt_out_filename, detected_circles, fmt='%d', delimiter=', ')

			# print(detected_circles)
		if picutre_out_filename:
			img = cv2.imread("data/images/map.jpg", cv2.IMREAD_COLOR)
			for pt in detected_circles:
				a, b, r = pt[0], pt[1], pt[2]
				# Draw the circumference of the circle.
				cv2.circle(img, (a, b), r, (0, 255, 0), 2)

				# Draw a small circle (of radius 1) to show the center.
				cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
				cv2.imshow("Detected Circle", img)

				cv2.imwrite(picutre_out_filename, img)

	return detected_circles