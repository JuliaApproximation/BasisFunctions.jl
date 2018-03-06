# tensorproduct_dict.jl


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

const TensorProductSpan{A,S,T,D <: TensorProductDict} = Span{A,S,T,D}

# Generic functions for composite types:
is_composite(dict::TensorProductDict) = true
elements(dict::TensorProductDict) = dict.dicts
element(dict::TensorProductDict, j::Int) = dict.dicts[j]
element(dict::TensorProductDict, range::Range) = tensorproduct(dict.dicts[range]...)
nb_elements(dict::TensorProductDict{N}) where {N} = N

# # The routine below is a type-stable way to obtain the length of the product set
# # Note that nb_elements above also returns N, but as a value and not as a type parameter
# @generated function product_length(s::TensorProductDict{DT}) where {DT}
#     LEN = tuple_length(DT)
#     :(Val{$LEN}())
# end


function TensorProductDict(dict::Dictionary)
    warn("A one element tensor product function set should not exist, use tensorproduct instead of TensorProductDict.")
    dict
end

product_domaintype(dict::Dictionary...) = Tuple{map(domaintype, dict)...}

function TensorProductDict(dicts::Dictionary...)
    N = length(dicts)
    DT = typeof(dicts)
    S = product_domaintype(dicts...)
    T = promote_type(map(codomaintype, dicts)...)
    TensorProductDict{N,DT,S,T}(dicts)
end

size(d::TensorProductDict) = d.size
size(d::TensorProductDict, j::Int) = d.size[j]

length(d::TensorProductDict) = prod(size(d))

# We need a more generic definition, but one can't iterate over a tuple type nor index it
dict_promote_domaintype(s::TensorProductDict{N}, ::Type{NTuple{N,T}}) where {N,T} =
    TensorProductDict(map(x->promote_domaintype(x, T), elements(s))...)

dict_promote_domaintype(s::TensorProductDict2, ::Type{Tuple{A,B}}) where {A,B} =
    TensorProductDict(promote_domaintype(element(s, 1), A), promote_domaintype(element(s, 2), B))

dict_promote_domaintype(s::TensorProductDict3, ::Type{Tuple{A,B,C}}) where {A,B,C} =
    TensorProductDict(promote_domaintype(element(s, 1), A),
                        promote_domaintype(element(s, 2), B),
                        promote_domaintype(element(s, 3), C))

dict_promote_domaintype(s::TensorProductDict4, ::Type{Tuple{A,B,C,D}}) where {A,B,C,D} =
    TensorProductDict(promote_domaintype(element(s, 1), A),
                        promote_domaintype(element(s, 2), B),
                        promote_domaintype(element(s, 3), C),
                        promote_domaintype(element(s, 4), D))


^(s::Dictionary, n::Int) = tensorproduct(s, n)

## Properties

for op in (:isreal, :is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
    @eval $op(s::TensorProductDict) = reduce(&, map($op, elements(s)))
end


##################
# Native indices
##################


ordering(d::TensorProductDict{N}) where {N} = ProductIndexList{N}(size(d))

# A tensor product set s has three types of indices:
# - Linear index: this is an Int, ranging from 1 to length(s)
# - Multilinear index: tuple of Ints, or CartesianIndex. Each element of the tuple
#   is a linear index for the corresponding element of the set
# - Native index: any other tuple type. Each element is the native index of the
#   corresponding element of the set.
# The storage of a product set is an array, and it can be indexed using either a
# linear index or a multilinear (cartesian) index.
#
# We define some conversion routines below.

# Convert the given index to a linear index.
# - If it is an Int, it already is a linear index
linear_index(s::TensorProductDict, i::Int) = i
# - If the argument is a tuple of integers or a CartesianIndex, then it is
#   a multilinear index.
linear_index(s::TensorProductDict, i::NTuple{N,Int}) where {N} = sub2ind(size(s), i...)
linear_index(s::TensorProductDict, i::CartesianIndex) = sub2ind(size(s), i.I...)
# - If its type is anything else, it may be a tuple of native indices
linear_index(s::TensorProductDict, idxn::Tuple) = linear_index(s, map(linear_index, elements(s), idxn))

# Convert the given index to a multilinear index.
# - A tuple of Int's is already a multilinear index
multilinear_index(s::TensorProductDict, idx::NTuple{N,Int}) where {N} = idx
# - From linear index to multilinear
multilinear_index(s::TensorProductDict, idx::Int) = ind2sub(size(s), idx)
# - Convert a CartesianIndex to a tuple (! this uses CartesianIndex internals currently)
multilinear_index(s::TensorProductDict, idx::CartesianIndex) = idx.I
# - From any other tuple
multilinear_index(s::TensorProductDict, idx::Tuple) = map(linear_index, elements(s), idx)

# Convert the given index to a native index.
# - From a multilinear index
native_index(s::TensorProductDict, idx::NTuple{N,Int}) where {N} = map(native_index, elements(s), idx)
# - Assume that another kind of tuple is the native index
native_index(s::TensorProductDict, idx::Tuple) = idx
# - From a linear index
native_index(s::TensorProductDict, idx::Int) = native_index(s, multilinear_index(s, idx))
# - From a Cartesian index
native_index(s::TensorProductDict, idx::CartesianIndex) = native_index(s, multilinear_index(s, idx))

# Convert an index into an index that is indexable, with length equal to the length of the product set
# - we convert a linear index into a multilinear one
indexable_index(set::TensorProductDict, idx::Int) = multilinear_index(set, idx)
# - for a tuple we don't need to do anything
indexable_index(set::TensorProductDict, idx::Tuple) = idx
# - we catch a CartesianIndex and convert it to multilinear as well
indexable_index(set::TensorProductDict, idx::CartesianIndex) = multilinear_index(set, idx)


## Feature methods

for op in (:has_grid, :has_extension, :has_derivative, :has_antiderivative)
    @eval $op(s::TensorProductDict) = reduce(&, map($op, elements(s)))
end

has_grid_transform(s::TensorProductDict, gb, grid::ProductGrid) =
    reduce(&, map(has_transform, elements(s), elements(grid)))

has_grid_transform(s::TensorProductDict, gb, grid::AbstractGrid) = false

for op in (:derivative_space, :antiderivative_space)
    @eval $op(s::TensorProductSpan, order; options...) =
        tensorproduct( map( i -> $op(element(s,i), order[i]; options...), 1:length(order))... )
end

resize(d::TensorProductDict, n) = TensorProductDict(map( (d_i,n_i)->resize(d_i, n_i), elements(d), n)...)
resize(d::TensorProductDict, n::Int) = resize(d, approx_length(d, n))


# nested_vector{DT}(set::TensorProductDict{DT}, x) = _nested_vector(DT, set, x)
#
# _nested_vector{A}(::Type{Tuple{A}}, set, x::SVector{1}) = x
# _nested_vector{A}(::Type{Tuple{A}}, set, x::Number) = x
#
# _nested_vector{A,B}(::Type{Tuple{A,B},2,T}, x::SVector{2}) = x
# _nested_vector{A,B,C}(::Type{Tuple{A,B,C},3,T}, x::SVector{3}) = x
# _nested_vector{A,B,C,D}(::Type{Tuple{A,B,C,D},4,T}, x::SVector{4}) = x


# Delegate in_support to _in_support with the composing dicts as extra arguments,
# in order to avoid extra memory allocation.
in_support(set::TensorProductDict, idx, x) =
    _in_support(set, elements(set), indexable_index(set, idx), x)

# This line is a bit slower than the lines below:
_in_support(::TensorProductDict, dicts, idx, x) = reduce(&, map(in_support, dicts, idx, x))

# That is why we handcode a few cases:
_in_support(::TensorProductDict1, dicts, idx, x) =
    in_support(dicts[1], idx[1], x[1])

_in_support(::TensorProductDict2, dicts, idx, x) =
    in_support(dicts[1], idx[1], x[1]) && in_support(dicts[2], idx[2], x[2])

_in_support(::TensorProductDict3, dicts, idx, x) =
    in_support(dicts[1], idx[1], x[1]) && in_support(dicts[2], idx[2], x[2]) && in_support(dicts[3], idx[3], x[3])

_in_support(::TensorProductDict4, dicts, idx, x) =
    in_support(dicts[1], idx[1], x[1]) && in_support(dicts[2], idx[2], x[2]) && in_support(dicts[3], idx[3], x[3]) && in_support(dicts[4], idx[4], x[4])


function approx_length(s::TensorProductDict, n::Int)
    # Rough approximation: distribute n among all dimensions evenly, rounded upwards
    N = dimension(s)
    m = ceil(Int, n^(1/N))
    tuple([approx_length(element(s, j), m^dimension(s, j)) for j in 1:nb_elements(s)]...)
end

extension_size(s::TensorProductDict) = map(extension_size, elements(s))

# It would be odd if the first method below was ever called, because LEN=1 makes
# little sense for a tensor product. But perhaps in generic code somewhere...
name(s::TensorProductDict) = "tensor product (" * name(element(s,1)) * names(s.dicts[2:end]...) * ")"
names(s1::Dictionary) = " x " * name(s1)
names(s1::Dictionary, s::Dictionary...) = " x " * name(s1) * names(s...)


getindex(s::TensorProductDict, ::Colon, i::Int) = (@assert nb_elements(s)==2; element(s,1))
getindex(s::TensorProductDict, i::Int, ::Colon) = (@assert nb_elements(s)==2; element(s,2))
getindex(s::TensorProductDict, ::Colon, ::Colon) = (@assert nb_elements(s)==2; s)

getindex(s::TensorProductDict, ::Colon, i::Int, j::Int) = (@assert nb_elements(s)==3; element(s,1))
getindex(s::TensorProductDict, i::Int, ::Colon, j::Int) = (@assert nb_elements(s)==3; element(s,2))
getindex(s::TensorProductDict, i::Int, j::Int, ::Colon) = (@assert nb_elements(s)==3; element(s,3))
getindex(s::TensorProductDict, ::Colon, ::Colon, i::Int) =
    (@assert nb_elements(s)==3; TensorProductDict(element(s,1),element(s,2)))
getindex(s::TensorProductDict, ::Colon, i::Int, ::Colon) =
    (@assert nb_elements(s)==3; TensorProductDict(element(s,1),element(s,3)))
getindex(s::TensorProductDict, i::Int, ::Colon, ::Colon) =
    (@assert nb_elements(s)==3; TensorProductDict(element(s,2),element(s,3)))
getindex(s::TensorProductDict, ::Colon, ::Colon, ::Colon) = (@assert nb_elements(s)==3; s)


grid(s::TensorProductDict) = ProductGrid(map(grid, elements(s))...)
#grid(b::TensorProductDict, j::Int) = grid(element(b,j))

# In general, left(f::Dictionary, j::Int) returns the left of the jth function in the set, not the jth dimension.
# The methods below follow this convention.
left(s::TensorProductDict) = SVector(map(left, elements(s)))
# left(s::TensorProductDict, j::Int) = SVector{N}([left(element(s,i),multilinear_index(s,j)[i]) for i=1:nb_elements(s)])
#left(b::TensorProductDict, idx::Int, j) = left(b, multilinear_index(b,j), j)
#left(b::TensorProductDict, idxt::NTuple, j) = left(b.dicts[j], idxt[j])

right(s::TensorProductDict) = SVector(map(right, elements(s)))
# right{DT,N,T}(s::TensorProductDict{DT,N,T}, j::Int) = SVector{N}([right(element(s,i),multilinear_index(s,j)[i]) for i=1:nb_elements(s)])
#right(b::TensorProductDict, j::Int) = right(element(b,j))
#right(b::TensorProductDict, idx::Int, j) = right(b, multilinear_index(b,j), j)
#right(b::TensorProductDict, idxt::NTuple, j) = right(b.dicts[j], idxt[j])


# Convert CartesianIndex argument to a tuple
getindex(s::TensorProductDict, idx::CartesianIndex) = getindex(s, idx.I)


# We pass on the elements of s as an extra argument in order to avoid
# memory allocations in the lines below
unsafe_eval_element(set::TensorProductDict, idx, x) = _unsafe_eval_element(set, elements(set), indexable_index(set, idx), x)

# For now, we assume that each set in the tensor product is a 1D set.
# This may not always be the case. If not, then x should have the same structure
# as i, i.e., both i and x should have N components.
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
index_set_tensorproduct(s,n) = CartesianRange(CartesianIndex(fill(1,n)...), CartesianIndex(fill(s+1,n)...))

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
