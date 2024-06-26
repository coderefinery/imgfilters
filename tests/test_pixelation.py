import os
import sys

# workaround so that we can import imgfilters without installing it as package
# this is not elegant but simplifies the discussion during the course
# which is not focused on packaging
sys.path.insert(0, os.path.abspath(".."))


from imgfilters.file_io import read_image
from imgfilters.filters import pixelate
from imgfilters.metrics import mean_squared_error


def test_pixelation():
    image = read_image("data/astronaut.jpg")

    image_pixelated = pixelate(image, scale=0.05, num_colors=8)
    image_reference = read_image("tests/reference/astronaut-pixelated.jpg")

    assert mean_squared_error(image_pixelated, image_reference) < 1e-3
