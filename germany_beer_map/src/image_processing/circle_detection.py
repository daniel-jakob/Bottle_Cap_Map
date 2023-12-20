import cv2
import numpy as np

def detect_circles(processed_image, picture_out_filename=None, txt_out_filename=None):
	# Check if processed_image is a string (filename)
	if isinstance(processed_image, str):
		# Determine the file extension
		file_extension = processed_image.split('.')[-1].lower()

		if file_extension == 'txt':
			# Read the numpy array from the .txt file
			processed_image = np.loadtxt(processed_image, dtype=int)
			# Normalize the image to 8-bit
			processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		elif file_extension in ['jpg', 'jpeg', 'png']:
			# Read the image using OpenCV
			processed_image = cv2.imread(processed_image, cv2.IMREAD_GRAYSCALE)

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

		# Filter out circles whose center is within another larger circle
		filtered_circles = []
		for i in range(len(detected_circles)):
			x1, y1, r1 = map(float, detected_circles[i])
			keep = True
			for j in range(len(detected_circles)):
				if i != j:
					x2, y2, r2 = map(float, detected_circles[j])
					if np.sqrt((x2 - x1)**2 + (y2 - y1)**2) < r2 and r1 < r2:
						keep = False
						break
			if keep:
				filtered_circles.append(detected_circles[i])

		if txt_out_filename:
				np.savetxt(txt_out_filename, detected_circles, fmt='%d', delimiter=', ')

		if picture_out_filename:
			img = cv2.imread("germany_beer_map/data/images/map.jpg", cv2.IMREAD_COLOR)
			for pt in filtered_circles:
				a, b, r = pt[0], pt[1], pt[2]
				# Draw the circumference of the circle.
				cv2.circle(img, (a, b), r, (0, 255, 0), 2)

				# Draw a small circle (of radius 1) to show the center.
				cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
				cv2.imshow("Detected Circle", img)

				cv2.imwrite(picture_out_filename, img)

	return filtered_circles