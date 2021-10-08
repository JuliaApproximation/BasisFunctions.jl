
"Supertype of tensor product dictionaries."
abstract type TensorProductDict{S,T} <: Dictionary{S,T} end


# component(dict::TensorProductDict, range::AbstractRange) = tensorproduct(dict.dicts[range]...)
function component(dict::TensorProductDict, range::AbstractRange)
    # TODO: remove this method in the future
    @warn "Range selection of product dictionaries now simply returns the components"
    [component(dict,i) for i in range]
end

factors(dict::TensorProductDict) = components(dict)

product_domaintype(dicts::Dictionary...) = Tuple{map(domaintype, dicts)...}
function product_domaintype(dicts::Vararg{Dictionary{<:Number},N}) where {N}
    T = promote_type(map(domaintype, dicts)...)
    SVector{N,T}
end

function TensorProductDict(dicts::Dictionary...)
    N = length(dicts)
    S = product_domaintype(dicts...)
    T = promote_type(map(codomaintype, dicts)...)
    C = promote_type(map(coefficienttype, dicts)...)
    dicts2 = map(dict->ensure_coefficienttype(C,dict),dicts)
    DT = typeof(dicts2)
    TupleProductDict{N,DT,S,T}(dicts2)
end



tolerance(dict::TensorProductDict) = minimum(map(tolerance,components(dict)))

dimensions(d::TensorProductDict) = map(dimensions, components(d))

coefficienttype(d::TensorProductDict) = promote_type(map(coefficienttype,components(d))...)

IndexStyle(d::TensorProductDict) = IndexCartesian()

^(d::Dictionary, n::Int) = tensorproduct(d, n)

similar(dict::TensorProductDict, ::Type{T}, size::Int) where {T} =
    similar(dict, T, approx_length(dict, size))


## Properties

for op in (:isreal, :isbasis, :isframe)
    @eval $op(s::TensorProductDict) = mapreduce($op, &, components(s))
end

for op in (:isorthogonal, :isorthonormal)
    @eval $op(s::TensorProductDict, m::ProductWeight) = mapreduce($op, &, components(s), components(m))
end

for op in (:isorthogonal, :isorthonormal)
    @eval BasisFunctions.$op(s::TensorProductDict, m::BasisFunctions.DiscreteProductWeight) = mapreduce($op, &, components(s), components(m))
end

for op in (:isorthogonal, :iscompatible)
    @eval BasisFunctions.$op(s::TensorProductDict, m::BasisFunctions.ProductGrid) = mapreduce($op, &, components(s), components(m))
end


## Feature methods

for op in (:hasinterpolationgrid, :hasextension, :hasderivative, :hasantiderivative)
    @eval $op(s::TensorProductDict) = mapreduce($op, &, components(s))
end
hasderivative(Φ::TensorProductDict, order) =
    mapreduce(hasderivative, &, components(Φ), order)
hasantiderivative(Φ::TensorProductDict, order) =
    mapreduce(hasantiderivative, &, components(Φ), order)

hasgrid_transform(s::TensorProductDict, gb, grid::ProductGrid) =
    mapreduce(hastransform, &, components(s), components(grid))

hasgrid_transform(s::TensorProductDict, gb, grid::AbstractGrid) = false

for op in (:derivative_dict, :antiderivative_dict)
    @eval $op(s::TensorProductDict, order; options...) =
        tensorproduct( map( (el,ord) -> $op(el, ord; options...), components(s), order)... )
end

for op in (:differentiation, :antidifferentiation)
    @eval function $op(::Type{T}, src::TensorProductDict, dest::TensorProductDict, order; options...) where {T}
        @assert length(order) == dimension(src)
        @assert length(order) == dimension(dest)
        tensorproduct(map( (el_s,el_d,ord) -> $op(T, el_s, el_d, ord; options...), components(src), components(dest), order)...)
    end
end

diff(dict::TensorProductDict, order; options...) =
    TensorProductDict( (diff(e,o) for (e,o) in zip(components(dict),order))...)

resize(d::TensorProductDict, n::Int) = resize(d, approx_length(d, n))
resize(d::TensorProductDict, dims) = TensorProductDict(map(resize, components(d), dims)...)


# Delegate dict_in_support to _dict_in_support with the composing dicts as extra arguments,
# in order to avoid extra memory allocation.
dict_in_support(dict::TensorProductDict, idx, x) =
    _dict_in_support(dict, components(dict), idx, x)
# catch CartesianIndex, convert to tuple, so that iteration works
dict_in_support(dict::TensorProductDict, idx::CartesianIndex, x) =
    _dict_in_support(dict, components(dict), Tuple(idx), x)

_dict_in_support(::TensorProductDict, dicts, idx, x) = mapreduce(in_support, &, dicts, idx, x)


function approx_length(s::TensorProductDict, n::Int)
    # Rough approximation: distribute n among all dimensions evenly, rounded upwards
    N = dimension(s)
    m = ceil(Int, n^(1/N))
    tuple([approx_length(component(s, j), m^dimension(s, j)) for j in 1:ncomponents(s)]...)
end

extensionsize(s::TensorProductDict) = map(extensionsize, components(s))

getindex(s::TensorProductDict, ::Colon, i::Int) = (@assert ncomponents(s)==2; component(s,1))
getindex(s::TensorProductDict, i::Int, ::Colon) = (@assert ncomponents(s)==2; component(s,2))
getindex(s::TensorProductDict, ::Colon, ::Colon) = (@assert ncomponents(s)==2; s)

getindex(s::TensorProductDict, ::Colon, i::Int, j::Int) = (@assert ncomponents(s)==3; component(s,1))
getindex(s::TensorProductDict, i::Int, ::Colon, j::Int) = (@assert ncomponents(s)==3; component(s,2))
getindex(s::TensorProductDict, i::Int, j::Int, ::Colon) = (@assert ncomponents(s)==3; component(s,3))
getindex(s::TensorProductDict, ::Colon, ::Colon, i::Int) =
    (@assert ncomponents(s)==3; TensorProductDict(component(s,1),component(s,2)))
getindex(s::TensorProductDict, ::Colon, i::Int, ::Colon) =
    (@assert ncomponents(s)==3; TensorProductDict(component(s,1),component(s,3)))
getindex(s::TensorProductDict, i::Int, ::Colon, ::Colon) =
    (@assert ncomponents(s)==3; TensorProductDict(component(s,2),component(s,3)))
getindex(s::TensorProductDict, ::Colon, ::Colon, ::Colon) = (@assert ncomponents(s)==3; s)


interpolation_grid(s::TensorProductDict) =
    ProductGrid(map(interpolation_grid, components(s))...)
gauss_rule(s::TensorProductDict) =
    productmeasure(map(gauss_rule, components(s))...)


support(s::TensorProductDict) = productdomain(map(support, components(s))...)
support(s::TensorProductDict, idx::LinearIndex) = support(s, native_index(s,idx))
support(s::TensorProductDict, idx::ProductIndex) = productdomain(map(support, components(s), indextuple(idx))...)


# We pass on the elements of s as an extra argument in order to avoid
# memory allocations in the lines below
# unsafe_eval_element(set::TensorProductDict, idx, x) = _unsafe_eval_element(set, components(set), indexable_index(set, idx), x)
unsafe_eval_element(dict::TensorProductDict, idx::ProductIndex, x) =
    _unsafe_eval_element(dict, components(dict), idx, x)

_unsafe_eval_element(dict::TensorProductDict, dicts, i, x) =
    mapreduce(unsafe_eval_element, *, dicts, i, x)
_unsafe_eval_element(dict::TensorProductDict, dicts, i::CartesianIndex, x) =
    mapreduce(unsafe_eval_element, *, dicts, Tuple(i), x)

unsafe_eval_element_derivative(dict::TensorProductDict, idx::ProductIndex, x, order) =
    _unsafe_eval_element_derivative(dict, components(dict), idx, x, order)

_unsafe_eval_element_derivative(dict::TensorProductDict, dicts, i, x, order) =
    mapreduce(unsafe_eval_element_derivative, *, dicts, i, x, order)
_unsafe_eval_element_derivative(dict::TensorProductDict, dicts, i::CartesianIndex, x, order) =
    mapreduce(unsafe_eval_element_derivative, *, dicts, Tuple(i), x, order)

hasmeasure(dict::TensorProductDict) = mapreduce(hasmeasure, &, components(dict))
measure(dict::TensorProductDict) = productmeasure(map(measure, components(dict))...)


innerproduct_native(Φ1::TensorProductDict, i, Φ2::TensorProductDict, j, measure::ProductWeight; options...) =
    mapreduce(innerproduct, *, components(Φ1), Tuple(i), components(Φ2), Tuple(j), components(measure))

evaluation(::Type{T}, dict::TensorProductDict, gb::GridBasis, grid::ProductGrid; options...) where {T} =
    tensorproduct(map( (d,g) -> evaluation(T, d, g; options...), components(dict), components(grid))...)

dual(dict::TensorProductDict, measure::Union{ProductWeight,DiscreteProductWeight}=measure(dict); options...) =
    TensorProductDict([dual(dicti, measurei; options...) for (dicti, measurei) in zip(components(dict),components(measure))]...)

gram(::Type{T}, dict::TensorProductDict, measure::Union{ProductWeight,DiscreteProductWeight}; options...) where {T} =
    TensorProductOperator(map((x,y)->gram(T, x,y; options...), components(dict), components(measure))...)
mixedgram(::Type{T}, dict1::TensorProductDict, dict2::TensorProductDict, measure::Union{ProductWeight,DiscreteProductWeight}; options...) where {T} =
    TensorProductOperator(map((x,y,z)->mixedgram(T, x,y,z; options...), components(dict1), components(dict2), components(measure))...)


Display.combinationsymbol(d::TensorProductDict) = Display.Symbol('⊗')
Display.displaystencil(d::TensorProductDict) = composite_displaystencil(d)
show(io::IO, mime::MIME"text/plain", d::TensorProductDict) = composite_show(io, mime, d)
show(io::IO, d::TensorProductDict) = composite_show_compact(io, d)



"A flat tensor product dict has `N` scalar components."
abstract type FlatTensorProductDict{N,S,T} <: TensorProductDict{S,T}
end

const TensorProductDict1{DT,S,T} = FlatTensorProductDict{1,S,T}
const TensorProductDict2{DT,S,T} = FlatTensorProductDict{2,S,T}
const TensorProductDict3{DT,S,T} = FlatTensorProductDict{3,S,T}
const TensorProductDict4{DT,S,T} = FlatTensorProductDict{4,S,T}


similar(dict::FlatTensorProductDict{N}, ::Type{T}, size::Vararg{Int,N}) where {T,N} =
    TensorProductDict(map(similar, components(dict), T.parameters, size)...)

## Native indices are of type ProductIndex

# The native indices of a tensor product dict are of type ProductIndex
ordering(d::FlatTensorProductDict{N}) where {N} = ProductIndexList{N}(size(d))

native_index(d::TensorProductDict, idx) = product_native_index(size(d), idx)

# We have to amend the boundscheck ecosystem to catch some cases:
# - This line will catch indexing with tuples of integers, and we assume
#   the user wanted to use a CartesianIndex
checkbounds(::Type{Bool}, d::FlatTensorProductDict{N}, idx::NTuple{N,Int}) where {N} =
    checkbounds(Bool, d, CartesianIndex(idx))
checkbounds(::Type{Bool}, d::FlatTensorProductDict{2}, idx::NTuple{2,Int}) =
    checkbounds(Bool, d, CartesianIndex(idx))
# - Any other tuple we assume is a recursive native index, which we convert
#   elementwise to a tuple of linear indices
checkbounds(::Type{Bool}, d::TensorProductDict, idx::Tuple) =
    checkbounds(Bool, d, map(linear_index, components(d), idx))



"""
    struct TupleProductDict{N,DT,S,T} <: Dictionary{S,T}

Parameters:
- DT is a tuple of types, representing the (possibly different) types of the dicts.
- N is the dimension of the product (equal to the length of the DT tuple)
- S is the domain type
- T is the codomain type.
"""
struct TupleProductDict{N,DT,S,T} <: FlatTensorProductDict{N,S,T}
    dicts   ::  DT
    size    ::  NTuple{N,Int}

    TupleProductDict{N,DT,S,T}(dicts::DT) where {N,DT,S,T} = new(dicts, map(length, dicts))
end

# Generic functions for composite types:
components(d::TupleProductDict) = d.dicts

size(d::TupleProductDict) = d.size




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
