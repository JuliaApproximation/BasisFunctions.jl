
"""
Makes sure that (i-1)/N <= x < i/N holds.
Return (i, true) if x ≈ i/N
else (i, false)
"""
function interval_index(B::Dictionary,x::Real)
    L = length(B)
    s = x*L
    r =  round(s)
    floor(Int,s)+1, s≈r
end

function first_index(b::BSplineTranslatesBasis, x::Real)
    ii, on_edge = BasisFunctions.interval_index(b, x)
    d = degree(b)
    if d == 0
        return ii, 1
    end
    if on_edge
        return mod(ii-2, length(b))+1, d
    else
        return mod(ii-1, length(b))+1, d+1
    end
end

_element_spans_one(b::BSplineTranslatesBasis) = degree(b) == 0

function _grid_index_limits_in_element_support(B::Dictionary, g::AbstractEquispacedGrid, i)
    dx = stepsize(g)
    x0 = g[1]
    s = support(B,i)
    if isa(s,AbstractInterval)
        start = ceil(Int,(infimum(s)-x0)/dx)
        stop = floor(Int,(supremum(s)-x0)/dx)
        !_element_spans_one(B) && ((infimum(s)-x0)/dx ≈ start) && (start += 1)
        ((supremum(s)-x0)/dx ≈ stop) && (stop -= 1)
        return (start+1, stop+1)
    else
        interval = elements(s)[1]
        # start = 0
        stop = floor(Int,(supremum(interval)-x0)/dx)
        ((supremum(interval)-x0)/dx ≈ stop) && (stop -= 1)
        # push!(indices,(start+1:stop+1)...)
        interval = elements(s)[2]
        start = ceil(Int,(infimum(interval)-x0)/dx)
        # stop = length(g)-1
        ((infimum(interval)-x0)/dx ≈ start) && (start += 1)
        return (start+1-length(g), stop+1)
    end
end

"""
Limits of the indices of `g` of points in the support of `B[i]`.
This is a tupple of (number of elements in tuple depends is equal to dimension) of two element tuples.
"""
grid_index_limits_in_element_support(B::Dictionary1d, g::AbstractGrid1d, i::Int) =
    tuple(_grid_index_limits_in_element_support(B, g, i))

grid_index_limits_in_element_support(B::TensorProductDict, g::ProductGrid, cartindex::CartesianIndex{N}) where {N} =
    [_grid_index_limits_in_element_support(s,element(g,i),cartindex[i]) for (i,s) in enumerate(elements(B))]

"""
Grid cartesian index limits of `g` of points in the support of `B[index]`.
"""
function grid_cartesian_index_limits_in_element_support(B::Dictionary, g::AbstractGrid, index)
    t = grid_index_limits_in_element_support(B, g, index)
    CartesianIndex([i[1]for i in t]...), CartesianIndex([i[2]for i in t]...)
end

"""
Grid indices of `g` of points in the support of `B[index]`.
"""
grid_index_range_in_element_support(B::Dictionary, g::AbstractGrid, index) =
    ModCartesianRange(size(g), grid_cartesian_index_limits_in_element_support(B, g, index)...)

grid_index_mask_in_element_support(B::Dictionary, g::AbstractGrid, indices) =
    grid_index_mask_in_element_support!(BitArray(size(g)), B, g, indices)

function grid_index_mask_in_element_support!(mask::BitArray, B::Dictionary, g::AbstractGrid, indices)
    fill!(mask, 0)
    for index in indices, i in grid_index_range_in_element_support(B, g, index)
        mask[i] = 1
    end
    mask
end

function grid_index_mask_in_element_support!(mask::BitArray, B::Dictionary, g::AbstractGrid, indices::BitArray)
    fill!(mask, 0)
    for i in eachindex(B)
        if indices[i]
            for j in grid_index_range_in_element_support(B, g, i)
                mask[j] = 1
            end
        end
    end
    mask
end

function _coefficient_index_limits_of_overlapping_elements(B::Dictionary1d, x::Real)
    # The init_index is the starting index of all spline elements that overlap with x
    init_index, no_elements = first_index(B,x)
    (init_index-no_elements+1, init_index)
end

"""
Limits of the indices of the coefficients of B that overlap with x.
This is a tupple of (number of elements in tuple depends is equal to dimension) of two element tuples.
"""
coefficient_index_limits_of_overlapping_elements(B::Dictionary, x::Real) =
    tuple(_coefficient_index_limits_of_overlapping_elements(B, x))

coefficient_index_limits_of_overlapping_elements(B::TensorProductDict, x::SVector{N}) where {N} =
    [_coefficient_index_limits_of_overlapping_elements(Bi,xi) for (Bi, xi) in zip(elements(B), x)]

"""
Cartesian index limits of the coefficients of B that overlap with x.
"""
function coefficient_cartesian_index_limits_of_overlapping_elementst(B::Dictionary, x)
    t = coefficient_index_limits_of_overlapping_elements(B, x)
    CartesianIndex([i[1]for i in t]...), CartesianIndex([i[2]for i in t]...)
end

"""
Range of coefficient indices of B that overlap with the point x.
"""
coefficient_index_range_of_overlapping_elements(B::Dictionary, x) =
    ModCartesianRange(size(B), coefficient_cartesian_index_limits_of_overlapping_elementst(B, x)...)

coefficient_index_mask_of_overlapping_elements(d::Dictionary, g::AbstractGrid) =
    coefficient_index_mask_of_overlapping_elements!(BitArray(size(d)), d, g)

function coefficient_index_mask_of_overlapping_elements!(mask::BitArray, B::Dictionary, g::AbstractGrid)
    fill!(mask, 0)
    for x in g, i in coefficient_index_range_of_overlapping_elements(B, x)
        mask[i] = 1
    end
    mask
end

coefficient_indices_of_overlapping_elements(dict::Dictionary, boundary::AbstractGrid) =
    find(BasisFunctions.coefficient_index_mask_of_overlapping_elements(dict, boundary))

struct ModCartesianRange{N}
    size::NTuple{N,Int}
    range::CartesianRange{CartesianIndex{N}}
end
ModCartesianRange(size::NTuple{N,Int}, index1::CartesianIndex{N}, index2::CartesianIndex{N}) where {N} =
    ModCartesianRange(size, CartesianRange(index1, index2))

Base.length(m::ModCartesianRange) = length(m.range)
Base.start(m::ModCartesianRange{N}) where {N} = start(m.range)
@generated function Base.next(m::ModCartesianRange{N}, state) where N
    t = Expr(:tuple, [:(mod(index[$i]-1,m.size[$i])+1) for i in 1:N]...)
    return quote
        index, state = next(m.range, state)
        CartesianIndex($t), state
    end
end

@generated function Base.next(m::ModCartesianRange{1}, state)
    t = :(mod(index[1]-1,m.size[1])+1)
    return quote
        index, state = next(m.range, state)
        $t, state
    end
end

Base.done(m::ModCartesianRange{N}, state) where N= done(m.range,  state)

function restriction_operator(dict::Dictionary, mask::BitArray)
     indices = find(mask)
     IndexRestrictionOperator(dict, dict[indices], indices)
end

##################
# Platform
##################

# 1D generators
primal_bspline_generator(::Type{T}, degree::Int) where {T} = n->BSplineTranslatesBasis(n, degree, T)
struct DualBSplineGenerator
    primal_generator
    oversampling::Int
    DualBSplineGenerator(primal_generator, oversampling::Int) = (@assert BasisFunctions.isdyadic(oversampling); new(primal_generator, oversampling))
end

function (DG::DualBSplineGenerator)(n::Int)
    B = DG.primal_generator(n)
    DG = DiscreteDualGram(B, oversampling=DG.oversampling)
    OperatedDict(DG)
end

dual_bspline_generator(primal_bspline_generator, oversampling::Int) = DualBSplineGenerator(primal_bspline_generator, oversampling)

# ND generators
primal_bspline_generator(::Type{T}, degree1::Int, degree2::Int, degree::Int...) where {T} = primal_bspline_generator(T, [degree1, degree2, degree...])

primal_bspline_generator(::Type{T}, degree::AbstractVector{Int}) where {T} = tensor_generator(T, map(d->primal_bspline_generator(T, d), degree)...)

(DG::DualBSplineGenerator)(n1::Int, n2::Int, ns::Int...) = DG([n1,ns...])

function (DG::DualBSplineGenerator)(n::AbstractVector{Int})
    B = DG.primal_generator(n)
    DG = DiscreteDualGram(B, oversampling=DG.oversampling)
    tensorproduct([OperatedDict(DGi) for DGi in elements(DG)]...)
end

# Sampler
bspline_sampler(::Type{T}, primal, oversampling::Int) where {T} = n-> GridSamplingOperator(gridbasis(grid(primal(n*oversampling)),T))

# params
bspline_param(init::Int) = DoublingSequence(init)

bspline_param(init::AbstractVector{Int}) = TensorSequence([BasisFunctions.MultiplySequence(i,2.^(1/length(init))) for i in init])

# Platform
function bspline_platform(::Type{T}, init::Union{Int,AbstractVector{Int}}, degree::Union{Int,AbstractVector{Int}}, oversampling::Int) where {T}
	primal = primal_bspline_generator(T, degree)
	dual = dual_bspline_generator(primal, oversampling)
	sampler = bspline_sampler(T, primal, oversampling)
	params = bspline_param(init)
	BasisFunctions.GenericPlatform(primal = primal, dual = dual, sampler = sampler,
		params = params, name = "B-Spline translates")
end
