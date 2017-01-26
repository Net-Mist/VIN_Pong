import numpy as np
from PIL import Image
import sys
import time
import glob
from image_preprocess import process

start_time = time.time()

# load the image
image = Image.open(sys.argv[1])
y_player, y_opponent, x_b, y_b = process(image)
duration = time.time() - start_time
print(duration)
print(y_player, y_opponent, x_b, y_b)
