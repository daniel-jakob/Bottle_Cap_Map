import cv2
import matplotlib.pyplot as plt
import numpy as np

def detect_outline(processed_image, picture_out_filename=None, txt_out_filename=None):
	# If processed_image is a string, assume it's a filename and read the numpy array from the file
	if isinstance(processed_image, str):
		processed_image = np.loadtxt(processed_image, dtype=int)


	_, processed_image = cv2.threshold(processed_image, 120, 255, 0)
	# fig, axs = plt.subplots(1, 2)
	# axs[0].set_title("Thresholded")
	# axs[0].imshow(processed_image, aspect="auto", cmap="gray")

	# Find lines
	contours, heirarchy= cv2.findContours(
		processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
	)

	# Find the indices of the contours sorted by arc length in descending order
	sorted_indices = sorted(range(len(contours)), key=lambda i: cv2.arcLength(contours[i], True), reverse=True)

	# Access the index of the second-longest contour
	# This is to disregard the longest contour, which is the outline border of the image
	index_of_second_longest_contour = sorted_indices[1] if len(sorted_indices) > 1 else None

	# Check if there is a second-longest contour
	if index_of_second_longest_contour is not None:
		# Access the second-longest contour
		second_longest_contour = contours[index_of_second_longest_contour]


		if txt_out_filename:
			f = open(txt_out_filename, 'w')
			for t in second_longest_contour:
				line = ' '.join(str(x) for x in t)
				f.write(line + '\n')
			f.close()

		# print(detected_circles)
		if picture_out_filename:
			# Draw the second-longest contour on a copy of the original image
			img_with_second_longest_contour = cv2.imread('data/images/map.jpg', cv2.IMREAD_COLOR)
			cv2.drawContours(img_with_second_longest_contour, [second_longest_contour], -1, (0, 255, 0), 2)

			# Display the image with the second-longest contour
			cv2.imshow("Image with Second Longest Contour", img_with_second_longest_contour)
			cv2.imwrite(picture_out_filename,img_with_second_longest_contour)
			cv2.waitKey(0)
			cv2.destroyAllWindows()


	return second_longest_contour[:, 0, :]
