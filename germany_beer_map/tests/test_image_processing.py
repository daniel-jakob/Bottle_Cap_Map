import numpy as np
from germany_beer_map.src.image_processing.circle_detection import detect_circles

def test_detect_circles():
    # Assuming 'processed_image.txt' is a file containing a processed image
    # in the form of a numpy array saved as text
    processed_image_filename = 'germany_beer_map/tests/fixtures/out_binary.jpg'
    detected_circles = detect_circles(processed_image_filename)

    assert detected_circles is not None
    assert len(detected_circles) >= 2
