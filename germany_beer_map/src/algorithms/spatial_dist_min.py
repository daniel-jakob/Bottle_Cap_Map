import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def spatial_dist_min(points_fixed, points_movable):
	"""
	spatial_dist_min, calculates the optimal mapping between two sets of geographical points to minimize the total Euclidean distance. The first set of points, referred to as the reference set, represents fixed locations. The second set of points, which will be mapped to the reference set, represents movable locations. The function uses the Hungarian Algorithm to find the optimal one-to-one correspondence between the two sets that results in the smallest cumulative distance. The output is a list of index pairs representing the optimal assignments and the minimum total distance.

	Parameters:
	points (list): A list of points.

	Returns:
	float: The minimum spatial distance between the points.
	"""
	# Create a cost matrix
	cost_matrix = cdist(points_fixed, points_movable, metric='euclidean')

	# If there are fewer movable points than fixed holes, pad the cost matrix with zeros
	if len(points_movable) < len(points_fixed):
		padding = np.zeros((len(points_fixed), len(points_fixed) - len(points_movable)))
		cost_matrix = np.hstack((cost_matrix, padding))

	# Use the Hungarian Algorithm to find the optimal assignment
	row_ind, col_ind = linear_sum_assignment(cost_matrix)

	# Calculate the minimum total distance
	min_total_distance = cost_matrix[row_ind, col_ind].sum()

	# Return the optimal assignment and the minimum total distance
	return list(zip(row_ind, col_ind)), min_total_distance
