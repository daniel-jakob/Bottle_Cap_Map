import cv2

class ImageProcessor:
	def __init__(self, image_path):
		self.image_path = image_path
		self.processed_image = self._process_image()

	def _process_image(self):
		# Greyscale image
		img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Morphological transformations
		se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
		bg=cv2.morphologyEx(img_gray, cv2.MORPH_DILATE, se)
		out_gray=cv2.divide(img_gray, bg, scale=255)
		out_binary=cv2.threshold(out_gray, 50, 255, cv2.THRESH_OTSU)[1]

		return out_binary