import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import random

def two_dim_interp(ref_contour, centres_of_holes=None, mapping="germany_beer_map/data/mapping/mapping.csv"):

	reference_contour_x = ref_contour[:, 0]
	# negate the y values to match the mapping file
	reference_contour_y = -ref_contour[:, 1]

	# Read the mapping file
	mapping = np.loadtxt(mapping, delimiter=',', skiprows=1)

	# Extract the distinctive points
	distinctive_points_x = mapping[:, 0]
	distinctive_points_y = mapping[:, 1]
	distinctive_points_longitude = mapping[:, 3]
	distinctive_points_latitude = mapping[:, 2]


	# Interpolate longitude and latitude based on distinctive points
	interp_longitude = griddata(
		(distinctive_points_x, distinctive_points_y),
		distinctive_points_longitude,
		(reference_contour_x, reference_contour_y),
		method='cubic'
	)

	interp_latitude = griddata(
		(distinctive_points_x, distinctive_points_y),
		distinctive_points_latitude,
		(reference_contour_x, reference_contour_y),
		method='cubic'
	)


	# Compute the convex hull of the distinctive points
	hull = ConvexHull(np.column_stack((distinctive_points_x, distinctive_points_y)))

	# Create two subplots side by side
	fig, axs = plt.subplots(1, 2, figsize=(20, 10))

	# Plot the distinctive points, contour, and convex hull on the first subplot
	axs[0].plot(reference_contour_x, reference_contour_y, 'b-', label='Contour')
	# Plot the convex hull of distinctive points
	for i, simplex in enumerate(hull.simplices):
		if i == 0:
			# Plot the first simplex with a label
			axs[0].plot(distinctive_points_x[simplex], distinctive_points_y[simplex], 'k-', label='Convex Hull')
		else:
			axs[0].plot(distinctive_points_x[simplex], distinctive_points_y[simplex], 'k-')

	axs[0].set_xlabel('x-coordinate')
	axs[0].set_ylabel('y-coordinate')
	axs[0].legend()

	# Plot the interpolated contour on the second subplot
	axs[1].scatter(distinctive_points_longitude, distinctive_points_latitude, color='red', label='Distinctive Points')
	for i in range(len(distinctive_points_longitude)):
		axs[1].text(distinctive_points_longitude[i], distinctive_points_latitude[i], f'({distinctive_points_x[i]}, {distinctive_points_y[i]}, {i+1})', fontsize=8)
	axs[1].plot(interp_longitude, interp_latitude, 'b-', label='Interpolated Contour')

	# Add text annotations for every 50th point on the interpolated contour
	for i in range(0, len(interp_longitude), 50):
		if not np.isnan(interp_longitude[i]) and not np.isnan(interp_latitude[i]):
			rounded_longitude = round(interp_longitude[i], 3)
			rounded_latitude = round(interp_latitude[i], 3)
			axs[1].scatter(rounded_longitude, rounded_latitude, color='green')
			axs[1].text(rounded_longitude, rounded_latitude, f'({rounded_longitude}, {rounded_latitude})', fontsize=8)

	# Check if centres_of_holes is provided
	if centres_of_holes is not None:
		# Convert the list of arrays to a 2D numpy array
		centres_of_holes = np.array(centres_of_holes)
		# Extract the x and y coordinates
		centres_of_holes_x = centres_of_holes[:, 0]
		centres_of_holes_y = centres_of_holes[:, 1]. astype(np.int16)  # convert to signed integer

		# Compute the convex hull of the centres of the holes
		holes_hull = ConvexHull(np.column_stack((centres_of_holes_x, -centres_of_holes_y)))
		# Plot the centres of the hole cutouts as points on the first subplot
		axs[0].scatter(centres_of_holes_x, -centres_of_holes_y, color='purple', label='Centres of Holes')
		# Plot the convex hull of distinctive points
		for i, simplex in enumerate(holes_hull.simplices):
			if i == 0:
				# Plot the first simplex with a label
				axs[0].plot(centres_of_holes_x[simplex], -centres_of_holes_y[simplex], 'g-', label='Convex Hull of Circle Centres')
			else:
				axs[0].plot(centres_of_holes_x[simplex], -centres_of_holes_y[simplex], 'g-')

	axs[1].set_xlabel('Longitude')
	axs[1].set_ylabel('Latitude')
	axs[1].legend()

	plt.show()