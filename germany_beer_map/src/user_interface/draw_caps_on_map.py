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

# draw_caps_on_map("germany_beer_map/data/mapping/bottle_caps.csv", [(2, 9), (5, 7), (16, 8), (17, 3), (27, 1), (33, 2), (34, 0), (37, 4), (44, 5), (45, 11), (47, 10), (48, 6)], [[ 902, 1734,  103], [ 958, 1998,  102], [1178, 1792,  102], [1076, 1020,  100], [ 280, 1444,  103], [ 312, 1702,  106], [ 660, 1950,  100], [ 356, 1962,  103], [ 430, 2204,  100], [ 714, 2224,  106], [1072, 1262,  100], [1644, 1676,  100], [ 798, 1196,  101], [1466, 1892,  100], [1862, 1256,   98], [ 904, 2760,  103], [1234, 2054,  102], [ 618, 1666,   98], [1618, 1158,  100], [1326, 2318,  106], [1642, 1416,   96], [ 820, 2496,  104], [1088, 1528,   93], [1120,  782,  100], [1410,  832,   98], [ 524, 1106,  103], [1910, 1576,   99], [ 828, 1470,   93], [828, 926,  97], [1360, 1598,   95], [ 630, 2800,  104], [1898,  712,   94], [1130, 2536,   97], [ 558, 1382,   96], [1744,  478,   98], [1922, 2532,  105], [1582,  658,  100], [1172, 2832,  103], [1008, 2276,  100], [1664, 2412,   93], [1076,  454,   93], [1928,  996,   94], [1688,  902,  100], [1338, 1334,   96], [1422, 2566,   95], [1468, 2856,   97], [1352, 1070,   94], [1584, 2166,   93], [1712, 2716,  102], [904, 646,  92], [568, 818,  96], [2084, 1376,   89], [1290,  614,   88]], "germany_beer_map/data/images/map.jpg")