import cv2
import matplotlib.pyplot as plt
import numpy as np

def detect_outline(processed_image):


    _, processed_image = cv2.threshold(processed_image, 120, 255, 0)
    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("Thresholded")
    axs[0].imshow(processed_image, aspect="auto", cmap="gray")

    # Find lines
    contours, heirarchy= cv2.findContours(
        processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # print(type(contours))
    # print(len(contours))

    # Find the indices of the contours sorted by arc length in descending order
    sorted_indices = sorted(range(len(contours)), key=lambda i: cv2.arcLength(contours[i], True), reverse=True)

    # Access the index of the second-longest contour
    # This is to disregard the longest contour, which is the outline of the image
    index_of_second_longest_contour = sorted_indices[1] if len(sorted_indices) > 1 else None

    # Check if there is a second-longest contour
    if index_of_second_longest_contour is not None:
        # Access the second-longest contour
        second_longest_contour = contours[index_of_second_longest_contour]
        f = open('file.txt', 'w')
        for t in second_longest_contour:
            line = ' '.join(str(x) for x in t)
            f.write(line + '\n')
        f.close()

        # Draw the second-longest contour on a copy of the original image
        img_with_second_longest_contour = cv2.imread('data/images/map.jpg', cv2.IMREAD_COLOR)
        cv2.drawContours(img_with_second_longest_contour, [second_longest_contour], -1, (0, 255, 0), 2)

        # Display the image with the second-longest contour
        cv2.imshow("Image with Second Longest Contour", img_with_second_longest_contour)
        cv2.imwrite('outline.jpg',img_with_second_longest_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
