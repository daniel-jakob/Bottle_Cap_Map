# main.py
import cv2
import csv

from image_processing.image_preprocessor import ImageProcessor

from image_processing.circle_detection import detect_circles
from image_processing.map_outline_detection import detect_outline
from algorithms.scaling import *
from algorithms.tps_transform import *
from algorithms.two_dim_interp import *
from user_interface.draw_contours import *
from algorithms.geocode import get_geocoordinates
from algorithms.spatial_dist_min import spatial_dist_min

# Create a single instance of ImageProcessor
preprocessor_photo = ImageProcessor("germany_beer_map/data/images/map.jpg")
preprocessor_ref = ImageProcessor("germany_beer_map/data/images/map_ref.jpg")



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





rotation_angle = find_optimal_rotation(ref_contour_scaled_aligned, contour)
# print(rotation_angle*180/3.16259)

contour_rotated = rotate_contour_around_centroid(contour, -rotation_angle)

circles_coords = two_dim_interp(circles, True,  ref_contour_scaled_aligned)

print(circles_coords)


# Read in bottle_caps.csv
bottle_caps = []
with open("germany_beer_map/data/mapping/bottle_caps.csv", 'r') as f:
	reader = csv.reader(f, delimiter=',')
	next(reader)  # Skip the header
	for row in reader:
		bottle_caps.append(row)

bottle_cap_coords = []
for i in range(len(bottle_caps)):
	bottle_cap = get_geocoordinates(bottle_caps[i][2])
	bottle_cap_coords.append(bottle_cap)
	print(bottle_cap)
print(np.array(bottle_cap_coords))

# bottle_cap_coords = [ [13.0944453, 54.2907393 ], [12.5440824,  50.68292505], [12.5440824,  50.68292505],
#  [ 7.95650099, 50.9910225 ],
#  [ 9.653526,   51.4176975 ],
#  [11.55363663, 48.14608575],
#  [11.9069408,  48.3068168 ],
#  [16.36783075, 48.21118145]]

spatial_dist_min(bottle_cap_coords, circles_coords)
exit(0)

# draw_contours(ref_contour_scaled_aligned)

contour_grid_points = grid_gen(contour)
contour_refined_grid = adaptive_grid(contour_grid_points, contour)
ref_contour_grid_points = grid_gen(contour_rotated)
ref_contour_refined_grid = adaptive_grid(ref_contour_grid_points, contour_rotated)
#tps_transform(contour_refined_grid, ref_contour_refined_grid, contour, contour_rotated)


draw_contours(contour_rotated, ref_contour_scaled_aligned, contour)


