import pytest
import os
from germany_beer_map.src.image_processing.circle_detection import detect_circles

@pytest.fixture(params=['fixtures/binary_out.jpg', 'fixtures/binary_out.txt'])
def processed_image_path(request):
    return os.path.join(os.path.dirname(__file__), request.param)


def test_detect_circles(processed_image_path):
    detected_circles = detect_circles(processed_image_path)

    # Assuming both .jpg and .txt files should have the same number of circles
    reference_image_path = os.path.join(os.path.dirname(__file__), 'fixtures/binary_out.jpg')
    reference_circles = detect_circles(reference_image_path)

    assert detected_circles is not None
    assert len(detected_circles) == len(reference_circles)
    assert len(detected_circles) >= 2