import numpy as np
from scipy.optimize import linear_sum_assignment
from math import radians, sin, cos, sqrt, atan2

import matplotlib.pyplot as plt

def haversine(lat1, lon1, lat2, lon2):
	# Convert coordinates to radians
	lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

	# Differences
	dlon = lon2 - lon1
	dlat = lat2 - lat1

	# Haversine formula
	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))

	# Radius of earth in kilometers (mean radius = 6,371km)
	R = 6371.0

	# Calculate the distance
	distance = R * c

	return distance

def spatial_dist_min(points_fixed, points_movable, plotting=False):
	"""
	spatial_dist_min, calculates the optimal mapping between two sets of geographical points to minimize the total Euclidean distance. The first set of points, referred to as the reference set, represents fixed locations. The second set of points, which will be mapped to the reference set, represents movable locations. The function uses the Hungarian Algorithm to find the optimal one-to-one correspondence between the two sets that results in the smallest cumulative distance. The output is a list of index pairs representing the optimal assignments and the minimum total distance.

	Parameters:
	points_fixed (list): A list of tuples, where each tuple contains the longitude and latitude of a fixed point (holes in map).
	points_movable (list): A list of tuples, where each tuple contains the longitude and latitude of a movable point (bottle caps/brewery locations)).

	Returns:
	float: The minimum spatial distance between the points.
	"""
	# Create a cost matrix
	cost_matrix = [[haversine(lat1, lon1, lat2, lon2) for (lon2, lat2) in points_fixed] for (lon1, lat1) in points_movable]

	# If there are fewer movable points than fixed holes, pad the cost matrix with zeros
	if len(points_movable) < len(points_fixed):
		padding = np.zeros((len(points_fixed), len(points_fixed) - len(points_movable)))
		cost_matrix = np.hstack((cost_matrix, padding))

	# Use the Hungarian Algorithm to find the optimal assignment (specifically, the Jonker-Volgenant variant of the algorithm)
	row_ind, col_ind = linear_sum_assignment(cost_matrix)

	# Convert cost_matrix to numpy array for advanced indexing
	cost_matrix_np = np.array(cost_matrix)

	# Calculate the minimum total distance
	min_total_distance = cost_matrix_np[row_ind, col_ind].sum()

	# Return the optimal assignment and the minimum total distance
	if plotting:
		visualize_cost_matrix(cost_matrix_np, row_ind, col_ind)

	return list(zip(row_ind, col_ind)), min_total_distance


def visualize_cost_matrix(cost_matrix, row_ind, col_ind):
	# Create column and row labels
	col_labels = ['Movable Point ' + str(i) for i in range(len(cost_matrix[0]))]
	row_labels = ['Fixed Hole ' + str(i) for i in range(len(cost_matrix))]

	# Convert cost matrix entries to strings with 2 decimal places
	formatted_values = [["{:.2f}".format(val) for val in row] for row in cost_matrix]

	# Define a scaling factor for the figure size
	scale = 0.3

	# Create a new figure with width and height set to the corresponding lengths of the table
	fig, ax = plt.subplots(figsize=(len(col_labels)*scale, len(row_labels)*scale))

	# Hide axes
	ax.axis('off')

	# Create a table and add it to the figure
	table = ax.table(cellText=formatted_values, colLabels=col_labels, rowLabels=row_labels, cellLoc='center', loc='center')

	# Auto adjust the width of the columns
	table.auto_set_column_width(col=list(range(len(col_labels))))

		# Set the font size
	table.auto_set_font_size(False)
	table.set_fontsize(8)  # Change the number to the desired font size

	# Highlight the cells corresponding to the optimal assignment
	for i, j in zip(row_ind, col_ind):
		table.get_celld()[(i+1, j)].set_facecolor('lightgreen')

	# Show the table
	plt.show()