#!/usr/bin/python3
#
# Copyright 2024 Dustin Kleckner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unstructured_resample import LegacyVTKReader
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


if len(sys.argv) != 2:
    raise ValueError('This script should be called with one argument (a VTK filename)')

ifn = sys.argv[1]

bfn = os.path.splitext(ifn)[0]
ifn_npz = bfn + '.npz'
ffn = bfn + '.png'

if os.path.exists(ifn_npz):
    print('Found NPZ version of data, loading that instead of original...')
    src = LegacyVTKReader(ifn_npz)
elif os.path.exists(ifn):
    print('Loading source file... this may take a minute')
    src = LegacyVTKReader(ifn)
    print(f'Saving NPZ version of source to "{ifn_npz}"')
    src.save(ifn_npz)
else:
    raise ValueError('Source file not found!')


x = np.linspace(-25, 25, 50)
X = np.zeros((len(x), len(x), 3))
X[..., 2] = x.reshape(-1, 1)
X[..., 1] = x.reshape(1, -1)
X[..., 0] = 50
imshow_kwargs = dict(origin='lower', extent=[x.min(), x.max(), x.min(), x.max()], interpolation='quadric')
imshow_level_kwargs = imshow_kwargs.copy()
imshow_level_kwargs['interpolation'] = 'nearest'
imshow_level_kwargs['clim'] = (-0.5, 8.5)
imshow_level_kwargs['cmap'] = "Set1"


print("Resampling velocity and vorticity...")
# Note: resampling multiple fields at once is faster than one at a time
# This is because the cell index can be reused.
V, omega = src.resample(X, 'velocity', 'vorticity')
grid_size = src.find_grid_size(X).max(-1) # Maxmium along any axis
print(f'Grid size range: {grid_size.min():2f}-{grid_size.max():2f}')

g0 = grid_size.min()
level = np.log2(grid_size / g0)

plt.figure(figsize=(11, 11))

plt.subplot(221)
plt.title('X Velocity')
plt.imshow(V[..., 0], **imshow_kwargs)
plt.xlabel('Y (mm)')
plt.ylabel('Z (mm)')
plt.colorbar()

plt.subplot(222)
plt.title('Y Velocity')
plt.imshow(V[..., 1], cmap='RdBu', **imshow_kwargs)
plt.xlabel('Y (mm)')
plt.ylabel('Z (mm)')
plt.colorbar()

plt.subplot(223)
plt.title('X Vorticity')
plt.imshow(omega[..., 0], cmap='BrBG', **imshow_kwargs)
plt.xlabel('Y (mm)')
plt.ylabel('Z (mm)')
plt.colorbar()

plt.subplot(224)
plt.title('Grid Size')
plt.imshow(level, **imshow_level_kwargs)
plt.xlabel('Y (mm)')
plt.ylabel('Z (mm)')
lt = np.arange(9)
cbar = plt.colorbar(ticks=lt)
cbar.ax.set_yticklabels([f'{g0 * 2**i:f}'[:4] for i in lt]) 

print(f'Saving figure to "{ffn}".')
plt.savefig(ffn)