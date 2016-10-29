# subsets.jl

checkbounds(set::FunctionSet, idx::Range) =
    (checkbounds(set, first(idx)); checkbounds(set, last(idx)))

checkbounds(set::FunctionSet, idx::CartesianRange) =
    (checkbounds(set, first(idx)); checkbounds(set, last(idx)))


"""
A FunctionSubSet is a subset of a function set. It is characterized by the
underlying larger set, and a collection of indices into that set.
"""
immutable FunctionSubSet{SET, IDX, N, T} <: FunctionSet{N,T}
    set ::  SET
    idx ::  IDX

    function FunctionSubSet(set::FunctionSet{N,T}, idx::IDX)
        checkbounds(set, idx)
        new(set, idx)
    end
end

FunctionSubSet{N,T}(set::FunctionSet{N,T}, idx) =
    FunctionSubSet{typeof(set),typeof(idx),N,T}(set, idx)

set(s::FunctionSubSet) = s.set
indices(s::FunctionSubSet) = s.idx

name(s::FunctionSubSet) = "Subset of " + name(set(s)) + " with indices " + string(indices(s))

promote_eltype{SET,IDX,N,T,S}(s::FunctionSubSet{SET,IDX,N,T}, ::Type{S}) =
    FunctionSubSet(promote_eltype(set(s), S), indices(s))

"Returns true if the subset consists of a single element."
has_single_index(s::FunctionSubSet) = _has_single_index(s, indices(s))
_has_single_index(s::FunctionSubSet, idx) = false
_has_single_index(s::FunctionSubSet, idx::Int) = true
_has_single_index(s::FunctionSubSet, idx::Tuple) = true
_has_single_index(s::FunctionSubSet, idx::CartesianIndex) = true
_has_single_index(s::FunctionSubSet, idx::NativeIndex) = true


length(s::FunctionSubSet) = has_single_index(s) ? 1 : length(indices(s))

size(s::FunctionSubSet) = has_single_index(s) ? (1,) : size(indices(s))


native_index(s::FunctionSubSet, idx::Int) = idx
linear_index(s::FunctionSubSet, idxn::Int) = idxn

# It is not clear in general what to do when attempting to resize a subset. We
# can not simply resize the underlying set, because the indices may no longer
# be correct. Instead, we relegate to a resize_subset routine with set(s) as
# an additional argument. This function can be implemented for specific sets
# if needed.
resize(s::FunctionSubSet, n) = resize_subset(s, set(s), n)

resize_subset(sub::FunctionSubSet, s::FunctionSet, n) = error("resize_subset not implemented for subset ", sub)

isreal(s::FunctionSubSet) = isreal(set(s))

# A subset of an orthogonal set may not be complete, but it will still be orthogonal.
# The same does not hold in general for biorthogonal sets.
is_orthogonal(s::FunctionSubSet) = is_orthogonal(set(s))

for op in [:left, :right]
    @eval $op{SET}(s::FunctionSubSet{SET,Int}) = $op(set(s), s.idx)
    @eval $op(s::FunctionSubSet) = $op(set(s))
end

for op in [:left, :right, :moment, :norm]
    @eval $op(s::FunctionSubSet, i) = $op(s.set, s.idx[i])
end

for op in [:has_derivative, :has_antiderivative]
    @eval $op(s::FunctionSubSet) = $op(set(s))
end

# TODO: think about these
#for op in [:has_grid, :has_transform]
#    @eval $op(s::FunctionSubSet) = $op(set(s))
#end

grid(s::FunctionSubSet) = grid(set(s))

rescale(s::FunctionSubSet, a, b) = FunctionSubSet(rescale(set(s), a, b), indices(s))

eval_element(s::FunctionSubSet, i, x) =
    has_single_index(s) ? eval_element(s.set, s.idx, x) : eval_element(s.set, s.idx[i], x)

# Internally, we use StaticArrays (SVector) to represent points, except in
# 1d where we use scalars. Here, for convenience, you can call a function with
# x, y, z arguments and so on. These are wrapped into an SVector.
(s::FunctionSubSet)(x) = eval_set_element(s.set, s.idx, x)
(s::FunctionSubSet)(x, y...) = eval_set_element(s.set, s.idx, SVector(x,y...))

eachindex(s::FunctionSubSet) = has_single_index(s) ? 1 : eachindex(s.idx)

subset(s::FunctionSet, idx) = FunctionSubSet(s, idx)
subset(s::FunctionSet, ::Colon) = s

getindex(s::FunctionSet, idx) = subset(s, idx)
getindex(s::FunctionSet, idx, indices...) = subset(s, (idx, indices...))

getindex(s::FunctionSubSet, idx) = has_single_index(s) && (idx == 1) ? s : FunctionSubSet(set(s), s.idx[idx])

function apply!(op::Extension, s2::FunctionSet, s1::FunctionSubSet, coef_dest, coef_src)
    @assert s2 == set(s1)
    fill!(coef_dest, 0)
    for (i,j) in enumerate(indices(s1))
        coef_dest[j] = coef_src[i]
    end
    coef_dest
end

function apply!(op::Restriction, s2::FunctionSubSet, s1::FunctionSet, coef_dest, coef_src)
    @assert s1 == set(s2)
    fill!(coef_dest, 0)
    for (i,j) in enumerate(indices(s2))
        coef_dest[i] = coef_src[j]
    end
    coef_dest
end

# In general, the derivative set of a subset can be the whole derivative set
# of the underlying set. We can not know generically whether the derivative set
# can be indexed as well. Same for antiderivative.
for op in [:derivative_set, :antiderivative_set]
    @eval $op(s::FunctionSubSet, order::Int) = $op(set(s), order)
end

function differentiation_operator(s1::FunctionSubSet, s2::FunctionSet, order::Int; options...)
    @assert s2 == derivative_set(s1, order)
    D = differentiation_operator(set(s1), s2, order; options...)
    E = Extension(s1, set(s1))
    D*E
end

function antidifferentiation_operator(s1::FunctionSubSet, s2::FunctionSet, order::Int; options...)
    @assert s2 == antiderivative_set(s1, order)
    D = antidifferentiation_operator(set(s1), s2, order; options...)
    E = Extension(s1, set(s1))
    D*E
end
