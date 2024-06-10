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

using StaticArrays

struct CellListGeometry{T, N}
    origin::SVector{N, T}
    h::SVector{N, T}
    N::SVector{N, Int64}
end

struct CellList{T, N}
    geometry::CellListGeometry{T, N}
    index_start::Array{Int64, N}
    index_end::Array{Int64, N}
    cell_start::Matrix{T}
    cell_end::Matrix{T}
    contents::Vector{Int64}

    function CellList(clg::CellListGeometry{T, 3}, cell_start_a::AbstractMatrix{T}, cell_end_a::AbstractMatrix{T}) where {T <: AbstractFloat}
        ndim, NC = size(cell_start_a)
        Vec = SVector{3, T}

        cell_start = reinterpret(Vec, cell_start_a)
        cell_end = reinterpret(Vec, cell_end_a)

        shape = (clg.N[1], clg.N[2], clg.N[3])

        # First figure out how many input grid cells span each cell
        counts = fill(Int64(0), shape)
        for ic in 1:NC
            # Input grid cells may span more than one cell list cell.
            i0, j0, k0 = cl_index(cell_start[ic], clg)
            i1, j1, k1 = cl_index(cell_end[ic], clg)
            for i in i0:i1
                for j in j0:j1
                    for k in k0:k1
                        counts[i, j, k] += 1
                    end
                end
            end
        end

        # Now build the starts array, which tells us the first index in
        # contents that corresponds to each cell
        starts = Array{Int64}(undef, shape)
        accum = 1
        for ic in 1:length(counts)
            starts[ic] = accum
            accum += counts[ic]
        end

        # Nominamlly the ends is starts[i+1]-1, but we can also use it as a 
        # counter to fill in each cell
        ends = copy(starts)

        # The list of input grid cells corresponding to each cell
        contents = Vector{Int64}(undef, accum)
        for ic in 1:NC
            i0, j0, k0 = cl_index(cell_start[ic], clg)
            i1, j1, k1 = cl_index(cell_end[ic], clg)
            for i in i0:i1
                for j in j0:j1
                    for k in k0:k1
                        contents[ends[i, j, k]] = ic
                        ends[i, j, k] += 1
                    end
                end
            end
        end

        new{T, 3}(clg, starts, ends, cell_start_a, cell_end_a, contents)
    end
end

function cl_index(X::AbstractVector{T}, clg::CellListGeometry{T, 3})::SVector{3, Int64} where {T <: AbstractFloat}
    return SVector{3, Int64}(
        clamp((X[1] - clg.origin[1]) ÷ clg.h[1] + 1, 1, clg.N[1]),
        clamp((X[2] - clg.origin[2]) ÷ clg.h[2] + 1, 1, clg.N[2]),
        clamp((X[3] - clg.origin[3]) ÷ clg.h[3] + 1, 1, clg.N[3]),
    )
end

function cl_index(X::AbstractVector{T}, cl::CellList{T, 3})::SVector{3, Int64} where {T <: AbstractFloat}
    return cl_index(X, cl.geometry)
end

function find_cell(X::SVector{3, T}, cl::CellList{T, 3})::Int64 where {T <: AbstractFloat}
    i, j, k = cl_index(X, cl)

    for n in cl.index_start[i, j, k]:cl.index_end[i, j, k]
        ic = cl.contents[n]
        if (
            (X[1] >= cl.cell_start[1, ic]) && 
            (X[2] >= cl.cell_start[2, ic]) && 
            (X[3] >= cl.cell_start[3, ic]) && 
            (X[1] <= cl.cell_end[1, ic]) &&
            (X[2] <= cl.cell_end[2, ic]) &&
            (X[3] <= cl.cell_end[3, ic])
        )

            return ic - 1 # Outputs are zero indexed, even though cell list itself is 1-indexed.
        end
    end

    return Int64(-1)
end

function build_cell_list(points_a::AbstractMatrix{T}, cell_start_index::AbstractVector{IT}, cell_point_index::AbstractVector{IT}, cell_type::AbstractVector{IT2}, sx::Real, sy::Real=-1.0, sz::Real=-1.0)::Tuple{CellList{T, 3}, Matrix{Int64}} where {T <: AbstractFloat, IT <: Integer, IT2 <: Integer}
    Vec = SVector{3, T}
    sx = T(sx)
    sy = sy > 0 ? T(sy) : sx
    sz = sz > 0 ? T(sz) : sy 
    # Nominal size of a grid cell
    s = Vec(sx, sy, sz)

    cell_start, cell_end, cell_corners = analyze_cells(points_a, cell_start_index, cell_point_index, cell_type)

    X0 = Vec(minimum(cell_start, dims=2))
    L = Vec(maximum(cell_end, dims=2)) - X0
    
    # Number of grid cells is rounded up
    N = SVector{3, Int64}(ceil.(L ./ s))

    # Actual cell size is slightly smaller due to rounding up of number along
    # each axis.  THis way it exactly fits the required size
    geometry = CellListGeometry(X0, L ./ N, N)

    return (CellList(geometry, cell_start, cell_end), cell_corners)
end


function analyze_cells(points_a::AbstractArray{T, 2}, cell_start_index::AbstractVector{IT}, cell_point_index::AbstractVector{IT}, cell_type::AbstractVector{IT2})::Tuple{Array{T, 2}, Array{T, 2}, Array{Int64, 2}} where {T <: AbstractFloat, IT <: Integer, IT2 <: Integer}
    # Returns: (
    #    cell_start: (x0, y0, z0),
    #    cell_end: (x1, y1, z1)
    #    cell_corners: (eight components, ordered)
    #    cell_list: CellList type
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

# Brute force cell finding, now depreciated in favor of the cell list version
function find_cell_index(X_a::AbstractArray{T, 2}, cell_data::Tuple{AbstractArray{T, 2}, AbstractArray{T, 2}, Array{Int64, 2}})::Vector{Int64} where {T <: AbstractFloat}
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

# Version using cell lists: much faster!
function find_cell_index(X_a::AbstractArray{T, 2}, cell_data::Tuple{CellList{T, 3}, Array{Int64, 2}})::Vector{Int64} where {T <: AbstractFloat}
    cell_list, corners = cell_data

    ndim, NX = size(X_a)
    if ndim != 3
        error("Points should be an array whose first dimension is 3 (found $ndim)")
    end

    Vec = SVector{3, T}
    X = reinterpret(Vec, X_a)
    
    cell_index = fill(Int64(-1), NX)

    @Threads.threads for i in 1:NX
        cell_index[i] = find_cell(X[i], cell_list)
    end

    return cell_index
end


function trilinear_resample(X::AbstractArray{T, 2}, V::AbstractArray{T, 2}, cell_index::Vector{Int64}, cell_data::Tuple{CellList{T, 3}, Array{Int64, 2}})::Array{T, 2} where {T <: AbstractFloat}
    cl, cell_corners = cell_data
    return trilinear_resample(X, V, cell_index, (cl.cell_start, cl.cell_end, cell_corners))
end


function trilinear_resample(X::AbstractArray{T, 2}, V::AbstractArray{T, 2}, cell_index::Vector{Int64}, cell_data::Tuple{AbstractArray{T, 2}, AbstractArray{T, 2}, Array{Int64, 2}})::Array{T, 2} where {T <: AbstractFloat}
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