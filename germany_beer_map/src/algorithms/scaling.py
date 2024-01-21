import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import rotate

def calculate_scale_factor(contour1, contour2):

	# Use NumPy functions to calculate the area of each contour
	# Using the Shoelace formula https://en.wikipedia.org/wiki/Shoelace_formula
	area1 = 0.5 * np.abs(np.sum(contour1[:-1, 0] * contour1[1:, 1] - contour1[1:, 0] * contour1[:-1, 1]))
	area2 = 0.5 * np.abs(np.sum(contour2[:-1, 0] * contour2[1:, 1] - contour2[1:, 0] * contour2[:-1, 1]))

	scale_factor = (area1 / area2)**0.5
	return scale_factor

def scale_contour(contour, scale_factor):

	# Apply the scale factor to each coordinate
	scaled_coordinates = (contour * scale_factor).astype(int)

	return scaled_coordinates

def calculate_centroid(contour):
	#print(contour)
	# Calculate the centroid
	centroid = np.mean(contour, axis=0).astype(int)

	return centroid

def	translate_contour(contour1, contour2):
	centroid1 = calculate_centroid(contour1)
	centroid2 = calculate_centroid(contour2)
	translation = (centroid2[0] - centroid1[0], centroid2[1] - centroid1[1])
	translated_contour = np.array(contour2) - np.array([[translation]])
	return translated_contour

def calculate_mse(contour1, contour2):
	return np.mean((contour1 - contour2) ** 2)

def rotate_contour_around_centroid(contour, angle_radians):
	# Calculate the centroid
	centroid = calculate_centroid(contour)


	# Create a rotation matrix
	rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
								[np.sin(angle_radians), np.cos(angle_radians)]])

	# Rotate the translated contour
	rotated_contour = np.dot(contour - centroid, rotation_matrix).astype(int)

	# Translate back to the original centroid
	rotated_contour += centroid

	return rotated_contour

def find_optimal_rotation(contour1, contour2):
	# Find the optimal rotation angle that minimises the Mean Squared Error
	best_rotation_angle = None
	min_mse = float('inf')

	angle_range = (-0.075, 0.075 * np.pi)
	angle_step = 0.075*np.pi / 100


	for angle in np.arange(angle_range[0], angle_range[1], angle_step):
		# Rotate the contour around its centroid
		rotated_contour = rotate_contour_around_centroid(contour2, angle)

		# Calculate the Mean Squared Error
		mse = calculate_mse(contour1, rotated_contour)

		# Update best rotation if the current MSE is smaller
		if mse < min_mse:
			min_mse = mse
			best_rotation_angle = angle

	return best_rotation_angle



def resample_contour(contour, target_num_points=1500):
	# Assuming contour is a numpy array of shape (n, 2)
	x = np.arange(len(contour))
	f = interp1d(x, contour, kind='linear', axis=0)
	resampled_contour = f(np.linspace(0, len(contour) - 1, target_num_points)).astype(int)

	return resampled_contour


