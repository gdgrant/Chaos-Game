from parameters import *
from game import *

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


print('Running simulation...')
xs, ys, cs = run_game()

print('Simulation complete!  Rendering...')
image_array = np.zeros([*par['image_size'],3])
w = par['image_size'][0]
h = par['image_size'][1]
for i, (x, y, c) in enumerate(zip(xs, ys, cs)):
	if i%1000==0:
		print('Pixel {} of {}.'.format(i, w*h), end='\r')
	image_array[int(w*x), int(h*y)] += c

print('Render complete!  Now saving.')
image_array += 0.2
image_array /= np.amax(image_array, (0,1), keepdims=True)
image_array = np.transpose(image_array, [1,0,2])
img = Image.fromarray((255*image_array).astype(np.uint8))
img.save('out2.png')