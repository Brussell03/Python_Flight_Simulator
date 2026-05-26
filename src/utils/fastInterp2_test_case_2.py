import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from time import time

from interpolators import fastInterp2

# Test 1: Compare interpolated images
#--------------------------------------------------------------------------
x = np.arange(-0.234, 2.1 + 1e-2, 1e-1)
y = np.arange( 1.222, 3.0 + 1e-2, 1e-1)

def zfun(x, y):
  return np.sin(2 * x * y) + np.cos(3 * x) - np.exp(x * np.sin(20 * y))

X, Y = np.meshgrid(x, y)
z = zfun(X, Y)

xi = np.arange(-0.5, 2.6 + 1e-2, 1e-1)
yi = np.arange( 0.8, 3.4 + 1e-2, 1e-1)

zj = np.zeros((len(xi), len(yi)))
zk = np.zeros((len(xi), len(yi)))

# Prepare data for RegularGridInterpolator 
interp_func = RegularGridInterpolator((x, y), z.T, method='linear', bounds_error=False, fill_value=None)

start_time = time()
for k in range(len(xi)):
    for l in range(len(yi)):
        
        # Fast interpolator
        zj[k, l] = fastInterp2(x, y, z.T, xi[k], yi[l])

        # RegularGridInterpolator
        zk[k, l] = interp_func((xi[k], yi[l]))
        
        # Depricated interpolator
        #zk[k, l] = scipy.interpolate.interp2d(x, y, z, kind='linear')(xi[k], yi[l]).item()  
        
end_time = time()
print(f"Interpolation time: {end_time - start_time:.4f} seconds")

z_min = -7
z_max = 2

plt.figure(1)
plt.imshow(zj.T, extent=[xi[0], xi[-1], yi[0], yi[-1]], origin='lower', aspect='auto', vmin=z_min, vmax=z_max)
plt.xlabel('x', fontsize=14, color='white')
plt.xlim(-0.5,2.6)
plt.ylim( 0.8,3.4)
plt.gca().set_facecolor('black')
plt.gcf().patch.set_facecolor('black')
plt.gcf().patch.set_edgecolor('white') 
plt.ylabel('y', fontsize=14, color='white')
plt.tick_params(colors = 'white')
plt.title('fastInterp2 Image', fontsize=14, color='white')
colorbar = plt.colorbar()
colorbar.ax.yaxis.set_tick_params(color='white')
colorbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
plt.set_cmap('viridis')  # Choose a colormap

ax = plt.gca() 
ax.spines['top'].set_edgecolor('white') 
ax.spines['bottom'].set_edgecolor('white') 
ax.spines['left'].set_edgecolor('white') 
ax.spines['right'].set_edgecolor('white')

plt.figure(2)
plt.imshow(zk.T, extent=[xi[0], xi[-1], yi[0], yi[-1]], origin='lower', aspect='auto', vmin=z_min, vmax=z_max)
plt.xlabel('x', fontsize=14, color='white')
plt.ylabel('y', fontsize=14, color='white')
plt.xlim(-0.5,2.6)
plt.ylim( 0.8,3.4)
plt.gca().set_facecolor('black')
plt.gcf().patch.set_facecolor('black')
plt.gcf().patch.set_edgecolor('white') 
plt.tick_params(colors = 'white')
plt.title('griddata Image', fontsize=14, color='white')
colorbar = plt.colorbar()
colorbar.ax.yaxis.set_tick_params(color='white')
colorbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
plt.set_cmap('viridis')

ax = plt.gca() 
ax.spines['top'].set_edgecolor('white') 
ax.spines['bottom'].set_edgecolor('white') 
ax.spines['left'].set_edgecolor('white') 
ax.spines['right'].set_edgecolor('white')

plt.figure(3)
plt.imshow(z, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto', vmin=z_min, vmax=z_max)
plt.xlabel('x', fontsize=14, color='white')
plt.ylabel('y', fontsize=14, color='white')
plt.xlim(-0.5,2.6)
plt.ylim( 0.8,3.4)
plt.gca().set_facecolor('black')
plt.gcf().patch.set_facecolor('black')
plt.tick_params(colors = 'white')
plt.title('True Image', fontsize=14, color='white')
colorbar = plt.colorbar()
colorbar.ax.yaxis.set_tick_params(color='white')
colorbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
plt.set_cmap('viridis')

ax = plt.gca() 
ax.spines['top'].set_edgecolor('white') 
ax.spines['bottom'].set_edgecolor('white') 
ax.spines['left'].set_edgecolor('white') 
ax.spines['right'].set_edgecolor('white')

plt.show()
