import cv2
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def filter_triangles(triangles, points, contour):
	filtered_triangles = []
	for triangle in triangles:
		# Compute the midpoints of the edges
		midpoints = [(points[triangle[i]] + points[triangle[(i+1)%3]]) / 2 for i in range(3)]
		# Check if all midpoints are inside the contour
		if all(cv2.pointPolygonTest(contour, (midpoint[0], midpoint[1]), False) > 0 for midpoint in midpoints):
			filtered_triangles.append(triangle)
	return np.array(filtered_triangles)


def reorder_points(points):
	# Sort points by their x-coordinates
	sorted_points = points[np.argsort(points[:, 0])]

	# Sort each subset of points with the same x-coordinate by their y-coordinates
	indices = np.argsort(sorted_points[:, 1])
	sorted_points = sorted_points[indices]

	# Assuming 'original_points' is your original set of points
	# and 'reordered_points' is the result of the reordering process

	# num_points_to_plot = 1000  # Choose the number of points to visualize
	# subset_reordered = sorted_points[:num_points_to_plot]
	# subset_original = points[:num_points_to_plot]

	# # Scatter plot for the original points with index numbers
	# plt.scatter(subset_original[:, 0], subset_original[:, 1], c='blue', label='Original Points', alpha=0.5)
	# for i, txt in enumerate(range(num_points_to_plot)):
	# 	plt.annotate(txt, (subset_original[i, 0], subset_original[i, 1]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)
	# plt.colorbar(label='Index')
	# plt.title(f'Original Points (First {num_points_to_plot} Points)')
	# plt.show()

	# # Scatter plot for the reordered points with index numbers
	# plt.scatter(subset_reordered[:, 0], subset_reordered[:, 1], c='red', label='Reordered Points', alpha=0.5)
	# for i, txt in enumerate(range(num_points_to_plot)):
	# 	plt.annotate(txt, (subset_reordered[i, 0], subset_reordered[i, 1]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)

	# plt.colorbar(label='Index')
	# plt.title(f'Reordered Points (First {num_points_to_plot} Points)')
	# plt.show()

	return sorted_points

def tps_transform(source_points, target_points, contour, ref_contour):
	# Reorder points
	target_points = reorder_points(target_points)

	# Plot the reordered points
	plt.plot(target_points[:, 0], target_points[:, 1], 'o', label='Original Points')

	# Plot the concave hull
	plt.plot(ref_contour[:, 0], ref_contour[:, 1], 'r--', lw=2, label='Concave Hull')

	plt.legend()
	plt.title('Concave Hull')
	plt.show()

	# Perform Delaunay triangulation on the target points within the convex hull
	triangulation = Delaunay(target_points)

	# # Plot the original points
	# plt.plot(target_points[:, 0], target_points[:, 1], 'o', label='Original Points')

	# # Plot the Delaunay triangulation
	# plt.triplot(target_points[:, 0], target_points[:, 1], triangulation.simplices, color='r', label='Delaunay Triangulation')
	# # Customize the plot
	# plt.xlabel('X-axis')
	# plt.ylabel('Y-axis')
	# plt.title('Delaunay Triangulation')
	# plt.legend()
	# plt.show()

	# Get the indices of the triangles formed by the target points
	triangles = triangulation.simplices
	print("triangles shape: ", triangles.shape)

	filtered_triangles = filter_triangles(triangles, target_points, ref_contour)

	# Plot the original points
	plt.plot(target_points[:, 0], target_points[:, 1], 'o', label='Original Points')

	# Plot the Delaunay triangulation
	plt.triplot(target_points[:, 0], target_points[:, 1], filtered_triangles, color='r', label='Delaunay Triangulation')
	# Customize the plot
	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')
	plt.title('Delaunay Triangulation')
	plt.legend()
	plt.show()


	# Compute the barycentric coordinates for each point in the source points
	bary_coords = np.asarray([triangulation.transform[:2, :2].dot(point - triangulation.transform[:2, 2])
        for point in target_points])
	print("Barycoords shape: ", bary_coords.shape)

	# Normalize barycentric coordinates so they sum to 1
	bary_coords = bary_coords / np.sum(bary_coords, axis=1, keepdims=True)

	# Convert barycentric coordinates to colors
	colors = plt.cm.viridis(bary_coords)

	# Visualize barycentric coordinates
	plt.figure(figsize=(8, 8))
	# plt.triplot(target_points[:, 0], target_points[:, 1], triangles)
	plt.triplot(target_points[:, 0], target_points[:, 1], filtered_triangles, color='r', label='Delaunay Triangulation')
	plt.tripcolor(target_points[:, 0], target_points[:, 1], filtered_triangles, facecolors=bary_coords[:, 0], cmap='viridis')
	# plt.plot(target_points[:, 0], target_points[:, 1], 'o', markersize=8)
	#  plt.triplot(source_points[:, 0], source_points[:, 1], color='r', linestyle='--')
	plt.title('Barycentric Coordinates')
	plt.show()

	# Define the thin-plate spline function
	def tps_function(r):
		return r**2 * np.log(r + 1e-6)

	# Compute the displacement field using the thin-plate spline function
	displacement_field = tps_function(np.linalg.norm(bary_coords, axis=2, keepdims=True))

	# Compute the weights for each vertex in the target triangles
	weights = np.einsum('...k,...k->...k', displacement_field, bary_coords / np.sum(bary_coords, axis=2, keepdims=True))

	# Reshape weights to match the shape of target_points[triangles]
	weights = weights.reshape(-1, 3, 2)

	# Apply the weights to compute the transformed points
	transformed_points = np.sum(weights[..., np.newaxis] * target_points[triangles], axis=1)

	return transformed_points

def grid_gen(contour):

	# Find the bounding box of the contour
	x, y, w, h = cv2.boundingRect(contour)
	grid_step = 20  # Adjust the step size as needed

	# Create a regular grid of points within the bounding box of the contour
	x_grid, y_grid = np.meshgrid(np.arange(x, x + w + grid_step, grid_step),
								np.arange(y, y + h + grid_step, grid_step))

	# Flatten the grid to get the points in the form needed for Delaunay triangulation
	grid_points = np.c_[x_grid.ravel(), y_grid.ravel()]

	# Filter grid points that are inside the contour
	inside_points = [point for point in grid_points if cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) == 1]

	inside_points = np.array(inside_points)

	# Add the original contour points to the grid
	inside_points = np.concatenate((inside_points, contour), axis=0)

	# Visualize the contour points and the grid points
	# y-coordinates are negated to match the image coordinate system of Matplotlib
	plt.plot(contour[:, 0], -contour[:, 1], 'bo', label='Contour Points')
	plt.plot(inside_points[:, 0], -inside_points[:, 1], 'rx', label='Grid Points Inside Contour')
	plt.legend()
	plt.title('Contour Points and Grid Points Inside Contour')
	plt.show()

	return inside_points

def adaptive_grid(inside_points, contour, distance_threshold=50):
	refined_points = inside_points.copy()

	for point in inside_points:
		# Check if the point is close to the contour
		distance = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True)

		if distance < distance_threshold:
			# Add points in the vicinity of the current point
			num_points_to_add = int((distance_threshold - distance) / 2.5)  # Adjust the factor as needed
			added_points = np.random.normal(loc=point, scale=5, size=(num_points_to_add, 2))

			# Check if each added point is inside the contour
			added_points_inside_contour = [cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) > 0
										   for p in added_points]

			refined_points = np.concatenate((refined_points, added_points[added_points_inside_contour]), axis=0)

	# Visualize the contour points, the original grid points, and the refined points
	plt.plot(contour[:, 0], -contour[:, 1], 'bo', label='Contour Points')
	plt.plot(refined_points[:, 0], -refined_points[:, 1], 'rx', label='Refined Points')
	plt.legend()
	plt.title('Contour Points and Refined Points')
	plt.show()

	return refined_points


# Example usage:
# source_contour and target_contour are assumed to be 2D arrays of shape (1500, 2)
#source_contour = np.array([[x, y] for x, y in zip(range(1500), np.random.rand(1500) * 100)])
#target_contour = np.array([[x, y] for x, y in zip(range(1500), np.random.rand(1500) * 100)])

# Use the tps_transform function to get the transformed source_contour
#transformed_contour = tps_transform(source_contour, target_contour)

# Visualize the results
# img = np.zeros((100, 1500, 3), dtype=np.uint8)
# cv2.polylines(img, [source_contour.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
# cv2.polylines(img, [target_contour.astype(int)], isClosed=True, color=(255, 0, 0), thickness=2)
# cv2.polylines(img, [transformed_contour.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)

# cv2.imshow('Contours', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
