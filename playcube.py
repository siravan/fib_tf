import numpy as np
import screen
from time import sleep

x = np.load('cube.npy')
n, h, w = x.shape
sc = screen.Screen(h, w, 'reentry!')

i = 0
while not sc.peek():
    sc.imshow(x[i % n,:,:])
    sleep(0.025)
    i += 1

sc.destroy()
