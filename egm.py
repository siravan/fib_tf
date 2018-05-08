import numpy as np
import br
from screen import Screen

def create_mask(model, x, y, radius):
    """
        create_mask creates a circular guassian mask, centered at (x,y)
    """
    xx, yy = np.meshgrid(np.arange(model.width), np.arange(model.height))
    dist = np.hypot(xx - x, yy - y)
    mask = np.array(np.exp(-(dist/radius)**2), dtype=np.float32)
    return mask


config = {
    'width': 512,           # screen width in pixels
    'height': 512,          # screen height in pixels
    'dt': 0.1,              # integration time step in ms
    'dt_per_plot': 10,      # screen refresh interval in dt unit
    'diff': 1.0,            # diffusion coefficient
    'duration': 3000,       # simulation duration in ms
    'skip': False,          # optimization flag: activate multi-rate
    'cheby': True,          # optimization flag: activate Chebysheb polynomials
    'timeline': False,      # flag to save a timeline (profiler)
    'timeline_name': 'timeline_br.json',
    'save_graph': False     # flag to save the dataflow graph
}

model = br.BeelerReuter(config)
model.add_hole_to_phase_field(150, 256, 50) # center=(150,200), radius=40
model.define()
model.add_pace_op('s2', 'luq', 10.0)
im = Screen(model.height, model.width, 'Beeler-Reuter Model')

s2 = model.millisecond_to_step(300)     # 300 ms

mask1 = create_mask(model, 300 + 15, 256, 5)
mask2 = create_mask(model, 300 - 15, 256, 5)
l = []

for i in model.run(im):
    if i == s2:
        model.fire_op('s2')
    if i % int(10 / model.dt_per_step) == 0:    # every 1 ms
        x1 = np.mean(model.image() * mask1)
        x2 = np.mean(model.image() * mask2)
        l.append([x1, x2])

l = np.asarray(l)
np.savetxt('test.dat', l)
