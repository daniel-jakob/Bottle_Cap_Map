import cv2
from PIL import Image
from utils.csv_reader import read_csv_file
# from csv_reader import read_csv_file
import numpy as np

def draw_caps_on_map(csvfile, placements, circles, img_read):
	# Read the image
	print(img_read)
	map = Image.open(img_read)

	# tuple of images of bottle cap from 4th column of csv file
	# Open the csv file
	bottle_caps = read_csv_file(csvfile)
	cap_img = tuple(d['image'] for d in bottle_caps)
	circles = np.array(circles)
	centres_of_holes_x = circles[:, 0]
	# Convert to signed int16 to avoid errors when negating y-values
	centres_of_holes_y = circles[:, 1].astype(np.int16)
	# Create a tuple of (x, y) coordinates
	coordinates = tuple(zip(centres_of_holes_x-100, centres_of_holes_y-100))

	# Create a list of tuples of (image, coordinates) via list comprehension
	matched_pairs = [(cap_img[j], coordinates[i]) for i, j in placements]

	# Use PIL to paste the bottle cap images onto the map
	for cap_img, coordinates in matched_pairs:
		cap_img = Image.open('germany_beer_map/data/images/bottlecaps/'+ cap_img)
		cap_img = cap_img.crop(cap_img.getbbox())
		cap_img = cap_img.resize((200, 200))
		# Use the alpha channel of the image as the mask
		mask = cap_img.split()[3] if cap_img.mode == 'RGBA' else None
		map.paste(cap_img, coordinates, mask=mask)
		# show the image
	map.save('output.png')
