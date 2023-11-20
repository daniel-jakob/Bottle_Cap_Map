# main.py
import cv2

from image_processing.image_preprocessor import ImageProcessor
#from image_processing.circle_detection import detect_circles
from image_processing.map_outline_detection import detect_outline

# Create a single instance of ImageProcessor
image_processor_instance = ImageProcessor("data/images/map.jpg")

# Use the processed image in both feature detection modules
#detect_circles(image_processor_instance.processed_image)
detect_outline(image_processor_instance.processed_image)