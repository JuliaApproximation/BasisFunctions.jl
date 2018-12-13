
"""
A `TensorProductDict` is itself a dictionary: the tensor product of a number of
dictionaries.

struct TensorProductDict{N,DT,S,T} <: Dictionary{S,T}

Parameters:
- DT is a tuple of types, representing the (possibly different) types of the dicts.
- N is the dimension of the product (equal to the length of the DT tuple)
- S is the domain type
- T is the codomain type.
"""
struct TensorProductDict{N,DT,S,T} <: Dictionary{S,T}
    dicts   ::  DT
    size    ::  NTuple{N,Int}

    TensorProductDict{N,DT,S,T}(dicts::DT) where {N,DT,S,T} = new(dicts, map(length, dicts))
end

const TensorProductDict1{DT,S,T} = TensorProductDict{1,DT,S,T}
const TensorProductDict2{DT,S,T} = TensorProductDict{2,DT,S,T}
const TensorProductDict3{DT,S,T} = TensorProductDict{3,DT,S,T}
const TensorProductDict4{DT,S,T} = TensorProductDict{4,DT,S,T}



# Generic functions for composite types:
is_composite(dict::TensorProductDict) = true
elements(dict::TensorProductDict) = dict.dicts
element(dict::TensorProductDict, j::Int) = dict.dicts[j]
element(dict::TensorProductDict, range::AbstractRange) = tensorproduct(dict.dicts[range]...)
numelements(dict::TensorProductDict{N}) where {N} = N

tolerance(dict::TensorProductDict)=minimum(map(tolerance,elements(dict)))

function TensorProductDict(dict::Dictionary)
    @warn("A one element tensor product function set should not exist, use tensorproduct instead of TensorProductDict.")
    dict
end

product_domaintype(dict::Dictionary...) = Tuple{map(domaintype, dict)...}

function TensorProductDict(dicts::Dictionary...)
    N = length(dicts)
    S = product_domaintype(dicts...)
    T = promote_type(map(codomaintype, dicts)...)
    c = promote_type(map(coefficienttype, dicts)...)
    dicts2 = map(s->promote_coefficienttype(s,c),dicts)
    DT = typeof(dicts2)
    TensorProductDict{N,DT,S,T}(dicts2)
end

size(d::TensorProductDict) = d.size

similar(dict::TensorProductDict, ::Type{T}, size::Int...) where {T} =
    TensorProductDict(map(similar, elements(dict), T.parameters, size)...)

coefficienttype(s::TensorProductDict) = promote_type(map(coefficienttype,elements(s))...)

IndexStyle(d::TensorProductDict) = IndexCartesian()

^(s::Dictionary, n::Int) = tensorproduct(s, n)

## Properties

for op in (:isreal, :is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
    @eval $op(s::TensorProductDict) = reduce(&, map($op, elements(s)))
end


## Native indices are of type ProductIndex

# The native indices of a tensor product dict are of type ProductIndex
ordering(d::TensorProductDict{N}) where {N} = ProductIndexList{N}(size(d))

native_index(d::TensorProductDict, idx) = product_native_index(size(d), idx)

"""
A recursive native index of a `TensorProductDict` is a tuple consisting of
native indices of each of the elements of the dictionary.
"""
recursive_native_index(d::TensorProductDict, idxn::ProductIndex) =
    map(native_index, elements(d), indextuple(idxn))

recursive_native_index(d::TensorProductDict, idx::LinearIndex) =
    recursive_native_index(d, native_index(d, idx))

# We have to amend the boundscheck ecosystem to catch some cases:
# - This line will catch indexing with tuples of integers, and we assume
#   the user wanted to use a CartesianIndex
checkbounds(::Type{Bool}, d::TensorProductDict{N}, idx::NTuple{N,Int}) where {N} =
    checkbounds(Bool, d, CartesianIndex(idx))
checkbounds(::Type{Bool}, d::TensorProductDict{2}, idx::NTuple{2,Int}) =
    checkbounds(Bool, d, CartesianIndex(idx))
# - Any other tuple we assume is a recursive native index, which we convert
#   elementwise to a tuple of linear indices
checkbounds(::Type{Bool}, d::TensorProductDict, idx::Tuple) =
    checkbounds(Bool, d, map(linear_index, elements(d), idx))


## Feature methods

for op in (:has_interpolationgrid, :has_extension, :has_derivative, :has_antiderivative)
    @eval $op(s::TensorProductDict) = reduce(&, map($op, elements(s)))
end

has_grid_transform(s::TensorProductDict, gb, grid::ProductGrid) =
    reduce(&, map(has_transform, elements(s), elements(grid)))

has_grid_transform(s::TensorProductDict, gb, grid::AbstractGrid) = false

for op in (:derivative_dict, :antiderivative_dict)
    @eval $op(s::TensorProductDict, order; options...) =
        tensorproduct( map( i -> $op(element(s,i), order[i]; options...), 1:length(order))... )
end

resize(d::TensorProductDict, n::Int) = resize(d, approx_length(d, n))



# Delegate dict_in_support to _dict_in_support with the composing dicts as extra arguments,
# in order to avoid extra memory allocation.
dict_in_support(dict::TensorProductDict, idx, x) =
    _dict_in_support(dict, elements(dict), idx, x)

# This line is a bit slower than the lines below:
_dict_in_support(::TensorProductDict, dicts, idx, x) = reduce(&, map(in_support, dicts, idx, x))

# That is why we handcode a few cases:
_dict_in_support(::TensorProductDict1, dicts, idx, x) =
    in_support(dicts[1], idx[1], x[1])

_dict_in_support(::TensorProductDict2, dicts, idx, x) =
    in_support(dicts[1], idx[1], x[1]) && in_support(dicts[2], idx[2], x[2])

_dict_in_support(::TensorProductDict3, dicts, idx, x) =
    in_support(dicts[1], idx[1], x[1]) && in_support(dicts[2], idx[2], x[2]) && in_support(dicts[3], idx[3], x[3])

_dict_in_support(::TensorProductDict4, dicts, idx, x) =
    in_support(dicts[1], idx[1], x[1]) && in_support(dicts[2], idx[2], x[2]) && in_support(dicts[3], idx[3], x[3]) && in_support(dicts[4], idx[4], x[4])


function approx_length(s::TensorProductDict, n::Int)
    # Rough approximation: distribute n among all dimensions evenly, rounded upwards
    N = dimension(s)
    m = ceil(Int, n^(1/N))
    tuple([approx_length(element(s, j), m^dimension(s, j)) for j in 1:numelements(s)]...)
end

extension_size(s::TensorProductDict) = map(extension_size, elements(s))

# It would be odd if the first method below was ever called, because LEN=1 makes
# little sense for a tensor product. But perhaps in generic code somewhere...
name(s::TensorProductDict) = "tensor product (" * name(element(s,1)) * names(s.dicts[2:end]...) * ")"
names(s1::Dictionary) = " x " * name(s1)
names(s1::Dictionary, s::Dictionary...) = " x " * name(s1) * names(s...)


getindex(s::TensorProductDict, ::Colon, i::Int) = (@assert numelements(s)==2; element(s,1))
getindex(s::TensorProductDict, i::Int, ::Colon) = (@assert numelements(s)==2; element(s,2))
getindex(s::TensorProductDict, ::Colon, ::Colon) = (@assert numelements(s)==2; s)

getindex(s::TensorProductDict, ::Colon, i::Int, j::Int) = (@assert numelements(s)==3; element(s,1))
getindex(s::TensorProductDict, i::Int, ::Colon, j::Int) = (@assert numelements(s)==3; element(s,2))
getindex(s::TensorProductDict, i::Int, j::Int, ::Colon) = (@assert numelements(s)==3; element(s,3))
getindex(s::TensorProductDict, ::Colon, ::Colon, i::Int) =
    (@assert numelements(s)==3; TensorProductDict(element(s,1),element(s,2)))
getindex(s::TensorProductDict, ::Colon, i::Int, ::Colon) =
    (@assert numelements(s)==3; TensorProductDict(element(s,1),element(s,3)))
getindex(s::TensorProductDict, i::Int, ::Colon, ::Colon) =
    (@assert numelements(s)==3; TensorProductDict(element(s,2),element(s,3)))
getindex(s::TensorProductDict, ::Colon, ::Colon, ::Colon) = (@assert numelements(s)==3; s)


interpolation_grid(s::TensorProductDict) =
    ProductGrid(map(interpolation_grid, elements(s))...)
#grid(b::TensorProductDict, j::Int) = grid(element(b,j))

# In general, left(f::Dictionary, j::Int) returns the left of the jth function in the set, not the jth dimension.
# The methods below follow this convention.
#left(s::TensorProductDict) = SVector(map(left, elements(s)))
# left(s::TensorProductDict, j::Int) = SVector{N}([left(element(s,i),multilinear_index(s,j)[i]) for i=1:numelements(s)])
#left(b::TensorProductDict, idx::Int, j) = left(b, multilinear_index(b,j), j)
#left(b::TensorProductDict, idxt::NTuple, j) = left(b.dicts[j], idxt[j])

#right(s::TensorProductDict) = SVector(map(right, elements(s)))
# right{DT,N,T}(s::TensorProductDict{DT,N,T}, j::Int) = SVector{N}([right(element(s,i),multilinear_index(s,j)[i]) for i=1:numelements(s)])
#right(b::TensorProductDict, j::Int) = right(element(b,j))
#right(b::TensorProductDict, idx::Int, j) = right(b, multilinear_index(b,j), j)
#right(b::TensorProductDict, idxt::NTuple, j) = right(b.dicts[j], idxt[j])

support(s::TensorProductDict) = cartesianproduct(map(support, elements(s)))

support(s::TensorProductDict, idx::LinearIndex) = support(s, native_index(s,idx))

support(s::TensorProductDict, idx::ProductIndex) = cartesianproduct(map(support, elements(s), indextuple(idx)))


# We pass on the elements of s as an extra argument in order to avoid
# memory allocations in the lines below
# unsafe_eval_element(set::TensorProductDict, idx, x) = _unsafe_eval_element(set, elements(set), indexable_index(set, idx), x)
unsafe_eval_element(set::TensorProductDict, idx::ProductIndex, x) = _unsafe_eval_element(set, elements(set), idx, x)

# We assume that x has exactly as many components as the index i does
# We can call unsafe_eval_element on the subsets because we have already done bounds checking
_unsafe_eval_element(set::TensorProductDict1, dicts, i, x) =
    unsafe_eval_element(dicts[1], i[1], x[1])

_unsafe_eval_element(set::TensorProductDict2, dicts, i, x) =
    unsafe_eval_element(dicts[1], i[1], x[1]) * unsafe_eval_element(dicts[2], i[2], x[2])

_unsafe_eval_element(set::TensorProductDict3, dicts, i, x) =
    unsafe_eval_element(dicts[1], i[1], x[1]) * unsafe_eval_element(dicts[2], i[2], x[2]) * unsafe_eval_element(dicts[3], i[3], x[3])

_unsafe_eval_element(set::TensorProductDict4, dicts, i, x) =
    unsafe_eval_element(dicts[1], i[1], x[1]) * unsafe_eval_element(dicts[2], i[2], x[2]) * unsafe_eval_element(dicts[3], i[3], x[3]) * unsafe_eval_element(dicts[4], i[4], x[4])

# Generic implementation, slightly slower
_unsafe_eval_element(s::TensorProductDict, dicts, i, x) =
    reduce(*, map(unsafe_eval_element, dicts, i, x))





"Return a list of all tensor product indices (1:s+1)^n."
index_set_tensorproduct(s,n) = CartesianIndices(ntuple(k->1,n), tuple(ntuple(k->s+1,n)))

"Return a list of all indices of total degree at most s, in n dimensions."
function index_set_total_degree(s, n)
    # We make a list of arrays because that is easier
    I = _index_set_total_degree(s, n)
    # and then we convert to tuples
    [tuple((1+i)...) for i in I]
end

function _index_set_total_degree(s, n)
    if n == 1
        I = [[i] for i in 0:s]
    else
        I = Array{Array{Int,1}}(0)
        I_rec = _index_set_total_degree(s, n-1)
        for idx in I_rec
            for m in 0:s-sum(abs.(idx))
                push!(I, [idx...; m])
            end
        end
        I
    end
end

"Return a list of all indices in an n-dimensional hyperbolic cross."
function index_set_hyperbolic_cross(s, n, α = 1)
    I = _index_set_hyperbolic_cross(s, n, α)
    [tuple((1+i)...) for i in I]
end

function _index_set_hyperbolic_cross(s, n, α = 1)
    if n == 1
        smax = floor(Int, s^(1/α))-1
        I = [[i] for i in 0:smax]
    else
        I = Array{Array{Int,1}}(0)
        I_rec = _index_set_total_degree(s, n-1)
        for idx in I_rec
            for m in 0:floor(Int,s^(1/α)/prod(1+abs.(idx)))-1
                push!(I, [idx...; m])
            end
        end
        I
    end
end

oversampled_grid(b::TensorProductDict, oversampling::Real) = ProductGrid([oversampled_grid(bi, oversampling) for bi in elements(b)]...)

BasisFunctions.DiscreteGram(s::BasisFunctions.TensorProductDict; oversampling = 1) =
    tensorproduct([DiscreteGram(si, oversampling=oversampling) for si in elements(s)]...)

function stencil(op::TensorProductDict)
    A = Any[]
    push!(A,element(op,1))
    for i=2:length(elements(op))
        push!(A," ⊗ ")
        push!(A,element(op,i))
    end
    A
end
