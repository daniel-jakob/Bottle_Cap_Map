# main.py
import cv2

from image_processing.image_preprocessor import ImageProcessor

from image_processing.circle_detection import detect_circles
from image_processing.map_outline_detection import detect_outline
from algorithms.scaling import *
from algorithms.tps_transform import *
from user_interface.draw_contours import *

# Create a single instance of ImageProcessor
preprocessor_photo = ImageProcessor("germany_beer_map/data/images/map.jpg")
preprocessor_ref = ImageProcessor("data/images/map_ref.jpg")



# Use the processed image in both feature detection modules
circles = detect_circles(preprocessor_photo.processed_image)
contour = detect_outline(preprocessor_photo.processed_image)

# Apply contour detection to a reference image of Germany outline
ref_contour = detect_outline(preprocessor_ref.processed_image)

# Resample the contours to smaller AND equal length
contour = resample_contour(contour)
ref_contour = resample_contour(ref_contour)


# Scale the contour and ref_contour and lay one atop the other for comparison
scale_factor = calculate_scale_factor(contour, ref_contour)
ref_contour_scaled = scale_contour(ref_contour, scale_factor) # sqrt of scale factor



ref_contour_scaled_aligned = translate_contour(contour, ref_contour_scaled).reshape(-1, 2)





rotation_angle = find_optimal_rotation(contour, ref_contour_scaled_aligned)
# print(rotation_angle*180/3.16259)

ref_contour_scaled_aligned_rotated = rotate_contour_around_centroid(ref_contour_scaled_aligned, rotation_angle)



