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

import re
import numpy as np
import os
os.environ['PYTHON_JULIACALL_THREADS'] = "auto"
os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = "yes"
import juliacall
jl = juliacall.Main


for pkg in ["StaticArrays"]:
    try: 
        jl.seval(f"using {pkg}")
    except:
        print(f'Installing Julia Package "{pkg}"; this make take a minute or more...')
        sys.stdout.flush()
        jl.seval(f'import Pkg; Pkg.add("{pkg}")')

SRC_DIR = os.path.split(__file__)[0]
jl.include(os.path.join(SRC_DIR, 'unstructured_resample.jl'))

class LegacyVTKReader:
    _TOKEN_MATCH = re.compile('([+-]?\d+\.?\d*([Ee][+-]?\d+)?|[a-zA-z0-9_]\w*)')
    _SUPPORTED_TYPES = ('UNSTRUCTURED_GRID')
    _DATA_TYPES = {
        'unsigned_char':'u1',
        'char':'i1',
        'unsigned_short':'u2',
        'short':'i2',
        'unsigned_int':'u4',
        'int':'i4',
        'unsigned_long':'u8',
        'long':'i8',
        'float':'f',
        'double':'d'
    }
    _SUB_DATA_TYPES = ('SCALARS', 'VECTORS', 'NORMALS')
    _SAVED_STR = ('dataset', 'fn')
    _SAVED_INT = ('num_points', 'num_cells')
    _SAVED_ARRAYS = ('points', 'cell_start_index', 'cell_num_points', 'cell_point_index', 'cell_types')
    _SAVED_CATEGORIES = ('point_data', 'cell_data', 'fields')

    def __init__(self, fn):
        ext = os.path.splitext(fn)[1].lower()

        if ext == '.npz':
            # For speed we may wish to resave the VTK to an npz file, which is
            # much quicker to load
            # This just unpacks them into the same data formats
            dat = np.load(fn)
            if str(dat["__ID__"]) != 'VTKResaved':
                raise ValueError('This file does not appear to be a resaved VTK file.')
            
            for k in self._SAVED_STR:
                if k in dat:
                    setattr(self, k, str(dat[k]))
                
            for k in self._SAVED_INT:
                if k in dat:
                    setattr(self, k, int(dat[k]))

            for k in self._SAVED_ARRAYS:
                if k in dat:
                    setattr(self, k, dat[k])   

            for cat in self._SAVED_CATEGORIES:
                prefix = f'__{cat.upper()}__'
                d = {}
                setattr(self, cat, d)

                for k in dat.keys():
                    if k.startswith(prefix):
                        d[k[len(prefix):]] = dat[k]

        elif ext == '.vtk':
            # This routine reads everything from a plain text VTK.
            # This is rather slow!
            self.fn = fn
            self.f = open(fn, 'rt')

            first_line = self.f.readline()
            if first_line.strip() != '# vtk DataFile Version 2.0':
                raise ValueError('The first line of this file should be "# vtk DataFile Version 2.0"')
            
            self.title = self.f.readline()

            self._tokens = []

            self._line = 2
            self.data_type = self.get_token()
            if self.data_type != "ASCII": 
                raise ValueError(f'This reader only supports ASCII data types, found "{self.data_type}"')
            
            token = self.get_token()
            if token != 'DATASET':
                raise ValueError(f'Expected "DATASET" after "ASCII" signifier in line {self._line}, found "{token}"')
            
            self.dataset = self.get_token()
            if self.dataset not in self._SUPPORTED_TYPES:
                raise ValueError(f'Unsupported dataset type "{self.dataset}" in line {self._line}')

            self.point_data = {}
            self.cell_data = {}
            self.fields = {}

            token = self.get_token()
            while token is not None:
                if token == 'FIELD':
                    field_name = self.get_token() # Not really used, but we have to clear a token

                    num_fields = self.get_int()
                    for n in range(num_fields):
                        name = self.get_token()
                        components = self.get_int()
                        entries = self.get_int()
                        dt = self.get_type()
                        arr = np.empty((entries, components), dtype=dt)
                        self.fields[name] = arr

                        for i in range(entries):
                            for j in range(components):
                                arr[i, j] = self.get_value()

                elif token == 'POINTS':
                    num_points = self.get_int()
                    dt = self.get_type()
                    arr = np.empty(num_points*3, dtype=dt)
                    self.points = arr.reshape(-1, 3)

                    for i in range(num_points*3):
                        arr[i] = self.get_value()

                elif token == 'CELLS':
                    num_cells = self.get_int()
                    total_size = self.get_int()
                    self.cell_start_index = np.empty(num_cells, 'i8')
                    self.cell_num_points = np.empty(num_cells, 'i8')
                    self.cell_point_index = np.empty(total_size, 'i8')

                    n = 0
                    for i in range(num_cells):
                        self.cell_start_index[i] = n
                        num_points = self.get_int()
                        self.cell_num_points[i] = num_points
                        for j in range(num_points):
                            self.cell_point_index[n+j] = self.get_int()
                        n += num_points

                elif token == 'CELL_TYPES':
                    num_cells = self.get_int()
                    self.cell_types = np.empty(num_cells, 'u1')

                    for i in range(num_cells):
                        self.cell_types[i] = self.get_int()

                elif token == "POINT_DATA":
                    self.num_points = self.unpack_data(self.point_data)

                elif token == "CELL_DATA":
                    self.num_cells = self.unpack_data(self.cell_data)

                elif token is None: # EOF
                    break

                else:
                    raise ValueError(f'Expected data type indictor in line {self._line} of "{self.fn}", found "{token}" instead.')
                
                token = self.get_token()

            self.close
        else:
            raise ValueError(f'Extension should be .npz or .vtk, found {ext}')

    def unpack_data(self, target):
        num_points = self.get_int()

        while self.peek_token() in self._SUB_DATA_TYPES:
            sd_type = self.get_token()

            if sd_type == "SCALARS":
                name = self.get_token()
                dt = self.get_type()
                components = self.get_int()

                if self.peek_token() == 'LOOKUP_TABLE':
                    self.get_token()
                    target[name + "_LOOKUP_TABLE"] = self.get_token()
            elif sd_type in ("VECTORS", "NORMALS"):
                name = self.get_token()
                dt = self.get_type()
                components = 3
            
            arr = np.empty(num_points * components, dtype=dt)
            target[name] = arr.reshape(num_points, components)
            for i in range(num_points * components):
                arr[i] = self.get_value()

        return num_points

    def get_token(self, consume=True):
        while not self._tokens:
            line = self.f.readline()
            self._line += 1
            if line.strip().startswith('#'):
                continue
            if not line: #EOF
                return None
            
            self._tokens = [g[0] for g in self._TOKEN_MATCH.findall(line)]
        if self._tokens:
            if consume:
                return self._tokens.pop(0)
            else: 
                return self._tokens[0]
        else:
            return None
        
    def peek_token(self):
        return self.get_token(consume=False)
    
    def close(self):
        self.f.close()
        delattr(self, 'f')

    def __del__(self):
        if hasattr(self, 'f'):
            self.close()

    def get_int(self, err_msg=None):
        token = self.get_token()
        try:
            return int(token)
        except:
            raise ValueError(f'Expected integer token in line {self._line} of VTK file "{self.fn}"\nFound {token} instead {"(" + err_msg + ")" if err_msg else ""}')
        
    def get_float(self, err_msg=None):
        token = self.get_token()
        try:
            return float(token)
        except:
            raise ValueError(f'Expected float token in line {self._line} of VTK file "{self.fn}"\nFound "{token}" instead {"(" + err_msg + ")" if err_msg else ""}')
            
    def get_value(self, err_msg=None):
        raise ValueError('get_type must be called before get_value to determine data type')

    def get_type(self, err_msg=None):
        token = self.get_token()
        if token not in self._DATA_TYPES:
            raise ValueError(f'Expected data type token in line {self._line} of VTK file "{self.fn}"\nFound "{token}" instead {"(" + err_msg + ")" if err_msg else ""}')
        else:
            dt = self._DATA_TYPES[token]
            self.get_value = self.get_float if dt in ('f', 'd') else self.get_int
            return dt
        
    def save(self, fn):
        ext = os.path.splitext(fn)[1]
        if ext.lower() != '.npz':
            raise ValueError('Saving is only supported to .npz files')
        
        data = {'__ID__':'VTKResaved'}

        for k in self._SAVED_STR + self._SAVED_INT + self._SAVED_ARRAYS:
            if hasattr(self, k):
                data[k] = getattr(self, k)

        for cat in self._SAVED_CATEGORIES:
            for k, v in getattr(self, cat).items():
                data[f'__{cat.upper()}__{k}'] = v

        np.savez(fn, **data)

    def resample(self, X, *fields):
        output_shape = X.shape[:-1]
        for field in fields:
            if field not in self.point_data:
                raise ValueError(f'"{field}" not in point_data\nAvailable options: {", ".join(k for k in self.point_data.keys() if self.point_data[k].dtype in ("f", "d"))}')
            
        Xj = X.reshape(-1, X.shape[-1]).T
        if not hasattr(self, 'jl_cell_data'):
            self.jl_cell_data = jl.analyze_cells(self.points.T, self.cell_start_index, self.cell_point_index, self.cell_types)
    
        cell_index = jl.find_cell_index(Xj, self.jl_cell_data)

        output = tuple(
            jl.trilinear_resample(Xj, self.point_data[field].T, cell_index, self.jl_cell_data).to_numpy().T.reshape(output_shape + (-1,))
            for field in fields
        )

        if len(output) == 1: 
            return output[0]
        else:
            return output