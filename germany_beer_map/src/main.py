# main.py
import cv2

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import necessary project modules
from image_processing.image_preprocessor import ImageProcessor
from image_processing.circle_detection import detect_circles
from image_processing.map_outline_detection import detect_outline
from algorithms.scaling import *
from algorithms.tps_transform import *
from algorithms.two_dim_interp import *
from user_interface.draw_contours import *
from algorithms.geocode import *
from algorithms.spatial_dist_min import spatial_dist_min
from user_interface.draw_caps_on_map import draw_caps_on_map

# vairables for file locations
map_file = "germany_beer_map/data/images/map.jpg"
ref_map_file = "germany_beer_map/data/images/map_ref.jpg"
bottle_cap_file = "germany_beer_map/data/mapping/bottle_caps.csv"

# Create a single instance of ImageProcessor
preprocessor_photo = ImageProcessor(map_file)
preprocessor_ref = ImageProcessor(ref_map_file)


# Use the processed image in both feature detection modules
circles = detect_circles(preprocessor_photo.processed_image, picture_out_filename=None, txt_out_filename=None)
contour = detect_outline(preprocessor_photo.processed_image, icture_out_filename=None, txt_out_filename=None)

# Apply contour detection to a reference image of Germany outline
ref_contour = detect_outline(preprocessor_ref.processed_image, picture_out_filename=None, txt_out_filename=None)

# Resample the contours to smaller AND equal array length
contour = resample_contour(contour)
ref_contour = resample_contour(ref_contour)


# Scale, align and rotate the contour and ref_contour and lay one atop the other for comparison
scale_factor = calculate_scale_factor(contour, ref_contour)
ref_contour_scaled = scale_contour(ref_contour, scale_factor) # sqrt of scale factor
ref_contour_scaled_aligned = translate_contour(contour, ref_contour_scaled).reshape(-1, 2)
rotation_angle = find_optimal_rotation(ref_contour_scaled_aligned, contour)
contour_rotated = rotate_contour_around_centroid(contour, -rotation_angle)

# draw_contours(ref_contour_scaled_aligned)
# draw_contours(contour_rotated, ref_contour_scaled_aligned, contour)

# Find the geo-coordinates of the circle centres and bottle caps associated breweries
circles_coords = two_dim_interp(circles, True,  ref_contour_scaled_aligned)
bottle_cap_coords = convert_address_to_coords(bottle_cap_file)

# Find the optimal placement of bottle caps on the map
placements, min_dist = spatial_dist_min(bottle_cap_coords, circles_coords, plotting=True)

# Draw the bottle caps on the map and save the image
draw_caps_on_map(bottle_cap_file, placements, circles, map_file)


# contour_grid_points = grid_gen(contour)
# contour_refined_grid = adaptive_grid(contour_grid_points, contour)
# ref_contour_grid_points = grid_gen(contour_rotated)
# ref_contour_refined_grid = adaptive_grid(ref_contour_grid_points, contour_rotated)
#tps_transform(contour_refined_grid, ref_contour_refined_grid, contour, contour_rotated)
