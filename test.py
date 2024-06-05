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

print("Resampling velocity and vorticity...")
# Note: resampling multiple fields at once is faster than one at a time
# This is because the cell index can be reused.
V, omega = src.resample(X, 'velocity', 'vorticity')

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.title('X Velocity')
plt.imshow(V[..., 0], **imshow_kwargs)
plt.xlabel('Y (mm)')
plt.ylabel('Z (mm)')
plt.colorbar()

plt.subplot(122)
plt.title('X Vorticity')
plt.imshow(omega[..., 0], cmap='RdBu', **imshow_kwargs)
plt.xlabel('Y (mm)')
plt.ylabel('Z (mm)')
plt.colorbar()

print(f'Saving figure to "{ffn}".')
plt.savefig(ffn)