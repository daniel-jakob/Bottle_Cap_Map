import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import random

def two_dim_interp(centres_of_holes, plotting=False, ref_contour=None, mapping="germany_beer_map/data/mapping/mapping.csv"):

	if plotting and ref_contour is None:
		raise ValueError("centres_of_holes must be provided when plotting is True")

	mapping = np.loadtxt(mapping, delimiter=',', skiprows=1)

	centres_of_holes = np.array(centres_of_holes)
	centres_of_holes_x = centres_of_holes[:, 0]
	# Convert to signed int16 to avoid errors when negating y-values
	centres_of_holes_y = centres_of_holes[:, 1].astype(np.int16)

	# print(centres_of_holes_x)
	# print(centres_of_holes_y)

	distinctive_points_x = mapping[:, 0]
	distinctive_points_y = mapping[:, 1]
	distinctive_points_latitude = mapping[:, 2]
	distinctive_points_longitude = mapping[:, 3]

	circles_interp_longitude = griddata(
		(distinctive_points_x, distinctive_points_y),
		distinctive_points_longitude,
		(centres_of_holes_x, -centres_of_holes_y),
		method='cubic'
	)

	circles_interp_latitude = griddata(
		(distinctive_points_x, distinctive_points_y),
		distinctive_points_latitude,
		(centres_of_holes_x, -centres_of_holes_y),
		method='cubic'
	)

	# print(circles_interp_latitude)

	circles_interp = np.column_stack((circles_interp_longitude, circles_interp_latitude))

	if plotting:
		reference_contour_x = ref_contour[:, 0]
		reference_contour_y = -ref_contour[:, 1]

		hull = ConvexHull(np.column_stack((distinctive_points_x, distinctive_points_y)))


		contour_interp_longitude = griddata(
			(distinctive_points_x, distinctive_points_y),
			distinctive_points_longitude,
			(reference_contour_x, reference_contour_y),
			method='cubic'
		)

		contour_interp_latitude = griddata(
			(distinctive_points_x, distinctive_points_y),
			distinctive_points_latitude,
			(reference_contour_x, reference_contour_y),
			method='cubic'
		)

		fig, axs = plt.subplots(1, 2, figsize=(20, 10))

		axs[0].plot(reference_contour_x, reference_contour_y, 'b-', label='Contour')

		for i, simplex in enumerate(hull.simplices):
			if i == 0:
				axs[0].plot(distinctive_points_x[simplex], distinctive_points_y[simplex], 'k-', label='Convex Hull')
			else:
				axs[0].plot(distinctive_points_x[simplex], distinctive_points_y[simplex], 'k-')

		axs[0].set_xlabel('x-coordinate')
		axs[0].set_ylabel('y-coordinate')
		axs[0].legend()

		axs[1].scatter(distinctive_points_longitude, distinctive_points_latitude, color='red', label='Distinctive Points')
		for i in range(len(distinctive_points_longitude)):
			axs[1].text(distinctive_points_longitude[i], distinctive_points_latitude[i], f'({distinctive_points_x[i]}, {distinctive_points_y[i]}, {i+1})', fontsize=8)
		axs[1].plot(contour_interp_longitude, contour_interp_latitude, 'b-', label='Interpolated Contour')

		for i in range(0, len(contour_interp_longitude), 50):
			if not np.isnan(contour_interp_longitude[i]) and not np.isnan(contour_interp_latitude[i]):
				rounded_contour_longitude = round(contour_interp_longitude[i], 3)
				rounded_contour_latitude = round(contour_interp_latitude[i], 3)
				axs[1].scatter(rounded_contour_longitude, rounded_contour_latitude, color='green')
				axs[1].text(rounded_contour_longitude, rounded_contour_latitude, f'({rounded_contour_longitude}, {rounded_contour_latitude})', fontsize=8)

		holes_hull = ConvexHull(np.column_stack((centres_of_holes_x, -centres_of_holes_y)))

		axs[0].scatter(centres_of_holes_x, -centres_of_holes_y, color='purple', label='Centres of Holes')

		for i in range(0, len(circles_interp_latitude)):
			rounded_circles_longitude = round(circles_interp_longitude[i], 3)
			rounded_circles_latitude = round(circles_interp_latitude[i], 3)
			if i ==0:
				axs[1].scatter(rounded_circles_longitude, rounded_circles_latitude, color='purple', label='Centres of Holes')
				axs[1].text(rounded_circles_longitude, rounded_circles_latitude, f'({rounded_circles_longitude}, {rounded_circles_latitude})', fontsize=8)
			else:
				axs[1].scatter(rounded_circles_longitude, rounded_circles_latitude, color='purple')
				axs[1].text(rounded_circles_longitude, rounded_circles_latitude, f'({rounded_circles_longitude}, {rounded_circles_latitude})', fontsize=8)

		for i, simplex in enumerate(holes_hull.simplices):
			if i == 0:
				axs[0].plot(centres_of_holes_x[simplex], -centres_of_holes_y[simplex], 'g-', label='Convex Hull of Circle Centres')
			else:
				axs[0].plot(centres_of_holes_x[simplex], -centres_of_holes_y[simplex], 'g-')


		axs[1].set_xlabel('Longitude')
		axs[1].set_ylabel('Latitude')
		axs[1].legend()

		plt.show()

	return circles_interp