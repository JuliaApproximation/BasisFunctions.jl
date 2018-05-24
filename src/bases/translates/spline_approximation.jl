
"""
Makes sure that (i-1)/N < x < i/N holds
If x ≈ i/N it return -i
"""
interval_index(B::BSplineTranslatesBasis,x::Real) = round(x*length(B))≈x*length(B) ? -round(Int,x*length(B))-1 : ceil(Int,x*length(B))

function BasisFunctions.support(B::BSplineTranslatesBasis{K,T}, i) where {K,T}
    start = T(i-1)/length(B)
    width = T(degree(B)+1)/length(B)
    stop  = start+width
    stop <=1 ? (return [start,stop]) : (return [T(0),stop-1], [start,T(1)])
end

"""
The linear index of the spline elements of B that are non-zero in x.
"""
function overlapping_elements(B::BSplineTranslatesBasis, x::Real)
    # The interval_index is the starting index of all spline elements that overlap with x
    init_index = interval_index(B,x)
    (init_index == -1-length(B)) && (init_index += length(B))
    degree(B) == 0 && return abs(init_index)
    # The number of elements that overlap with one interval
    no_elements = degree(B)+1
    if init_index < 0
        init_index = -init_index-1
        no_elements = no_elements-1
    end
    [mod(init_index+i-2,length(B)) + 1 for i in 1:-1:2-no_elements]
end

"""
The linear indices of the points of `g` at which B[i] is not zero.
"""
function support_indices(B::BSplineTranslatesBasis, g::AbstractEquispacedGrid, i)
    indices = Vector{Int}()
    dx = stepsize(g)
    x0 = g[1]
    s = support(B,i)
    if length(s[1]) == 1
        start = ceil(Int,(s[1]-x0)/dx)
        stop = floor(Int,(s[2]-x0)/dx)
        (degree(B) != 0) && ((s[1]-x0)/dx ≈ start) && (start += 1)
        ((s[2]-x0)/dx ≈ stop) && (stop -= 1)
        push!(indices,(start+1:stop+1)...)
    else
        interval = s[1]
        start = 0
        stop = floor(Int,(interval[2]-x0)/dx)
        ((interval[2]-x0)/dx ≈ stop) && (stop -= 1)
        push!(indices,(start+1:stop+1)...)

        interval = s[2]
        start = ceil(Int,(interval[1]-x0)/dx)
        stop = length(g)-1
        ((interval[1]-x0)/dx ≈ start) && (start += 1)
        push!(indices,(start+1:stop+1)...)
    end
    indices
end

function support_indices(B::TensorProductDict, g::ProductGrid, index::Int)
    cartindex = ind2sub(size(B),index)
    index_sets = [support_indices(s,element(g,i),cartindex[i]) for (i,s) in enumerate(elements(B))]
    create_indices(g,index_sets...)
end

function support_indices(B::TensorProductDict, g::ProductGrid, cartindex::CartesianIndex{N}) where {N}
    index_sets = [support_indices(s,element(g,i),cartindex[i]) for (i,s) in enumerate(elements(B))]
    create_indices(g,index_sets...)
end

# function create_indices(B, i1, i2)
#     [linear_index(B,(i,j)) for i in i1 for j in i2]
# end
#
# function create_indices(B, i1, i2, i3)
#     [linear_index(B,(i,j,k)) for i in i1 for j in i2 for k in i3]
# end

function create_indices(B, i1, i2)
    [CartesianIndex(i,j) for i in i1 for j in i2]
end

function create_indices(B, i1, i2, i3)
    [CartesianIndex(i,j,k) for i in i1 for j in i2 for k in i3]
end


function overlapping_elements(B::TensorProductDict, x::SVector)
    index_sets = [overlapping_elements(s,x[i]) for (i,s) in enumerate(elements(B))]
    create_indices(B,index_sets...)
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
    DG = DiscreteDualGram(Span(B), oversampling=DG.oversampling)
    OperatedDict(DG)
end

dual_bspline_generator(primal_bspline_generator, oversampling::Int) = DualBSplineGenerator(primal_bspline_generator, oversampling)

# ND generators
primal_bspline_generator(::Type{T}, degree1::Int, degree2::Int, degree::Int...) where {T} = primal_bspline_generator(T, [degree1, degree2, degree...])

primal_bspline_generator(::Type{T}, degree::AbstractVector{Int}) where {T} = tensor_generator(T, map(d->primal_bspline_generator(T, d), degree)...)

(DG::DualBSplineGenerator)(n1::Int, n2::Int, ns::Int...) = DG([n1,ns...])

function (DG::DualBSplineGenerator)(n::AbstractVector{Int})
    B = DG.primal_generator(n)
    DG = DiscreteDualGram(Span(B), oversampling=DG.oversampling)
    tensorproduct([OperatedDict(DGi) for DGi in elements(DG)]...)
end

# Sampler
bspline_sampler(::Type{T}, primal, oversampling::Int) where {T} = n-> GridSamplingOperator(gridspace(grid(primal(n*oversampling)),T))

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
