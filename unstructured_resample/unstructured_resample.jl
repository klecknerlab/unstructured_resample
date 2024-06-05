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

# using StaticArrays
# imports already done on the Python end of things

function analyze_cells(points_a::AbstractArray{T, 2}, cell_start_index::AbstractVector{IT}, cell_point_index::AbstractVector{IT}, cell_type::AbstractVector{IT2})::Tuple{Array{T, 2}, Array{T, 2}, Array{Int64, 2}} where {T <: AbstractFloat, IT <: Integer, IT2 <: Integer}
    # Returns: (
    #    cell_start: (x0, y0, z0),
    #    cell_end: (x1, y1, z1)
    #    cell_corners: (eight components, ordered)
    # )
    ndim, NP = size(points_a)
    if ndim != 3
        error("Points should be an array whose first dimension is 3 (found $ndim)")
    end

    NC = length(cell_start_index)

    cell_start_a = Array{T}(undef, 3, NC)
    cell_end_a = Array{T}(undef, 3, NC)
    # cell_corners = Array{Int64}(undef, 8, NC)
    cell_corners = fill(Int64(1), (8, NC))
    Vec = SVector{3, T}

    points = reinterpret(Vec, points_a)
    cell_start = reinterpret(Vec, cell_start_a)
    cell_end = reinterpret(Vec, cell_end_a)

    @Threads.threads for i in 1:NC
        ct = cell_type[i]
        if (ct < 11) || (ct > 12)
            error("Cell type for tri-linear resample must be 11 or 12 (voxel or hexahedron)")
        end 

        # Start and end indices of cell_index, which gives the indices of the
        # points corresponding to the edges of the cell
        ics = cell_start_index[i] + 1
        ice = ics + 7 # There should always be 8 points in this type of cell!

        # Find the min/max values of each cell
        # Note that the index arrays are zero indexed!
        cs = points[cell_point_index[ics] + 1]
        ce = cs 

        for ic in (ics+1):ice
            p = points[cell_point_index[ic] + 1]
            cs = min.(cs, p)
            ce = max.(ce, p)
        end

        cell_start[i] = cs
        cell_end[i] = ce

        # Now identify which point in each cell corresponds to each corner.
        # Note the "id" of each corner:
        #   0 = 0b000: x-, y-, z-
        #   1 = 0b001: x+, y-, z-
        #   2 = 0b010: x-, y+, z-
        #   3 = 0b011: x+, y+, z-
        #   and so on...
        found_ids = 0
        for ic in ics:ice
            ip = cell_point_index[ic]
            p = points[ip + 1]
            id = 0
            for d in 1:ndim
                if p[d] == ce[d]
                    id += 1 << (d-1)
                elseif p[d] != cs[d]
                    error("point $ip does not correspond to an edge of cell $(ic-1); is this unstructured grid rectilinear?")            
                end
            end
            found_ids += 1 << id
            cell_corners[id+1, i] = ip
        end

        # We should have found all eight unique corners
        if found_ids != 255
            error("cell $(i-1) does not appear to be rectilinear")
        end
    end

    return (cell_start_a, cell_end_a, cell_corners)
end

function find_cell_index(X_a::AbstractArray{T, 2}, cell_data::Tuple{AbstractArray{T, 2}, AbstractArray{T, 2}, Array{Int64, 2}})::Vector{Int64} where {T <: Number}
    ndim, NX = size(X_a)
    if ndim != 3
        error("Points should be an array whose first dimension is 3 (found $ndim)")
    end

    Vec = SVector{3, T}
    X = reinterpret(Vec, X_a)
    cell_start = reinterpret(Vec, cell_data[1])
    cell_end = reinterpret(Vec, cell_data[2])
    cell_index = fill(Int64(-1), NX)

    # Brute force check if each point is in each cell.
    @Threads.threads for i in 1:NX
        XX = X[i]
        for ic in 1:length(cell_start)
            if all(XX .>= cell_start[ic]) && all(XX .<= cell_end[ic])
                cell_index[i] = ic-1 # all input/output arrays zero indexed
                break
            end
        end
    end

    return cell_index
end

function trilinear_resample(X::AbstractArray{T, 2}, V::AbstractArray{T, 2}, cell_index::Vector{Int64}, cell_data::Tuple{AbstractArray{T, 2}, AbstractArray{T, 2}, Array{Int64, 2}})::Array{T, 2} where {T <: Number}
    cell_start, cell_end, cell_corners = cell_data

    ndim, NX = size(X)
    if ndim != 3
        error("X should be an array whose first dimension is 3 (found $ndim)")
    end


    Vec = SVector{3, T}

    ndimV, NV = size(V)
    V_out = fill(T(NaN), (ndimV, NX))

    @Threads.threads for i in 1:NX
        x, y, z = X[1, i], X[2, i], X[3, i]
        ic = cell_index[i] + 1
        if ic != 0 # Make sure it is in a valid cell
            x0, y0, z0 = cell_start[1, ic], cell_start[2, ic], cell_start[3, ic]
            x1, y1, z1 =   cell_end[1, ic],   cell_end[2, ic],   cell_end[3, ic]

            # https://en.wikipedia.org/wiki/Trilinear_interpolation
            xd = (x - x0) / (x1 - x0)
            oxd = 1 - xd
            yd = (y - y0) / (y1 - y0)
            oyd = 1 - yd
            zd = (z - z0) / (z1 - z0)
            ozd = 1 - zd

            p000, p100, p010, p110, p001, p101, p011, p111 = cell_corners[1:8, ic]

            for j in 1:ndimV
                v00 = V[j, p000+1] * oxd + V[j, p100+1] * xd
                v10 = V[j, p010+1] * oxd + V[j, p110+1] * xd
                v01 = V[j, p001+1] * oxd + V[j, p101+1] * xd
                v11 = V[j, p011+1] * oxd + V[j, p111+1] * xd
                v0 = v00 * oyd + v10 * yd
                v1 = v01 * oyd + v11 * yd
                V_out[j, i] = v0 * ozd + v1 * zd
            end
        end
    end

    return V_out
end