import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter

cdict = {'red': [(0.0,  0.0, 0.0),
                 (0.33, 0.0, 0.0),
                 (0.66,  1.0, 1.0),
                 (1.0,  1.0, 1.0)],
         'blue': [(0.0,  0.0, 0.0),
                  (0.33, 1.0, 1.0),
                  (0.66,  0.0, 0.0),
                  (1.0,  0.0, 0.0)],
         'green': [(0.0,  0.0, 0.0),
                   (0.33, 0.0, 0.0),
                   (0.66,  0.0, 0.0),
                   (1.0,  1.0, 1.0)]}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

def scale(x, pos):
    'The two args are the value and tick position'
    return '%1.1f' % (x / 100.0)
def scale2(x, pos):
    'The two args are the value and tick position'
    return '%1.1f' % (x / 100.0)






size = int(sys.argv[2])

x, y, z = np.loadtxt(sys.argv[1]).T

data = np.zeros((size, size))
m = 0
x_m = 0
y_m = 0
for i in range(0, len(z)):
    data[round(x[i] * size), round(y[i] * size)] = z[i]
    if z[i] > m:
        x_m = round(x[i] * size)
        y_m = round(y[i] * size)
        m = z[i]
data = np.ma.masked_where(data == 0, data)

print "best:"+str(max(z))

def load_points(fname):
    p_z, p_y, p_x = np.loadtxt(fname).T
    p_x *= size
    p_y *= size
    p_p_x = []
    p_p_y = []
    np_p_x = []
    np_p_y = []

    for i in range(0, len(p_x)):
        if p_z[i] == 1.0:
            p_p_x += [p_x[i]]
            p_p_y += [p_y[i]]
        else:
            np_p_x += [p_x[i]]
            np_p_y += [p_y[i]]
    return p_p_x, p_p_y, np_p_x, np_p_y



fig = plt.figure()
im = plt.imshow(data.T, origin='lower', cmap=my_cmap)
im.set_interpolation('nearest')
fig.subplots_adjust(top=0.98)
cb = plt.colorbar()
for t in cb.ax.get_xticklabels():
    t.set_fontsize(130)


ax = fig.add_subplot(111)
ax.yaxis.set_major_formatter(FuncFormatter(scale))
ax.xaxis.set_major_formatter(FuncFormatter(scale2))

plt.savefig('heatmap.pdf')
