# expansions.jl

"Check whether a set of coefficients is compatible with the function set."
compatible_coefficients(set::FunctionSet, coefficients) =
    length(set) == length(coefficients)

"""
A `SetExpansion` describes a function using its expansion coefficients in a certain
function set.

Parameters:
- S is the function set.
- C is the type of the expansion coefficients
"""
struct SetExpansion{S,C}
    set             ::  S
    coefficients    ::  C

    function SetExpansion{S,C}(set, coefficients) where {S,C}
        @assert compatible_coefficients(set, coefficients)
        new(set, coefficients)
    end
end

SetExpansion(set::FunctionSet, coefficients = zeros(set)) =
    SetExpansion{typeof(set),typeof(coefficients)}(set, coefficients)

SetExpansion(span::Span) = SetExpansion(set(span), zeros(span))

similar(e::SetExpansion, coefficients) = SetExpansion(set(e), coefficients)

eltype{S,C}(::Type{SetExpansion{S,C}}) = eltype(C)

set(e::SetExpansion) = e.set

coefficients(e::SetExpansion) = e.coefficients

span(e::SetExpansion) = Span(set(e), eltype(e))

# For expansions of composite types, return a SetExpansion of a subset
element(e::SetExpansion, i) = SetExpansion(element(e.set, i), element(e.coefficients, i))

elements(e::SetExpansion) = map(SetExpansion, elements(e.set), elements(e.coefficients))

# Delegation of methods
for op in (:length, :size, :left, :right, :grid)
    @eval $op(e::SetExpansion) = $op(set(e))
end

# Delegation of property methods
for op in (:numtype, :dimension, :nb_elements)
    @eval $op(s::SetExpansion) = $op(set(s))
end

has_basis(e::SetExpansion) = is_basis(set(e))
has_frame(e::SetExpansion) = is_frame(set(e))

eachindex(e::SetExpansion) = eachindex(coefficients(e))

getindex(e::SetExpansion, i...) = e.coefficients[i...]

setindex!(e::SetExpansion, v, i...) = (e.coefficients[i...] = v)


# This indirect call enables dispatch on the type of the set of the expansion
(e::SetExpansion)(x) = call_set_expansion(e, set(e), coefficients(e), x)
(e::SetExpansion)(x, y) = call_set_expansion(e, set(e), coefficients(e), SVector(x, y))
(e::SetExpansion)(x, y, z) = call_set_expansion(e, set(e), coefficients(e), SVector(x, y, z))
(e::SetExpansion)(x, y, z, t) = call_set_expansion(e, set(e), coefficients(e), SVector(x, y, z, t))
(e::SetExpansion)(x, y, z, t, u...) = call_set_expansion(e, set(e), coefficients(e), SVector(x, y, z, t, u...))

call_set_expansion(e::SetExpansion, set::FunctionSet, coefficients, x) =
    eval_expansion(set, coefficients, x)

function differentiate(e::SetExpansion, order=1)
    op = differentiation_operator(e.set, order)
    SetExpansion(dest(op), apply(op,e.coefficients))
end

function antidifferentiate(e::SetExpansion, order=1)
    op = antidifferentiation_operator(e.set, order)
    SetExpansion(dest(op), apply(op,e.coefficients))
end

Base.broadcast(e::SetExpansion, grid::AbstractGrid) = eval_expansion(set(e), coefficients(e), grid)

Base.broadcast{T}(e::SetExpansion, x::LinSpace{T}) = broadcast(e, EquispacedGrid(x))

# Shorthands for partial derivatives
∂x(f::SetExpansion) = differentiate(f, 1, 1)
∂y(f::SetExpansion) = differentiate(f, 2, 1)
∂z(f::SetExpansion) = differentiate(f, 3, 1)
# Shorthands for partial integrals
∫∂x(f::SetExpansion) = antidifferentiate(f, 1, 1)
∫∂y(f::SetExpansion) = antidifferentiate(f, 2, 1)
∫∂z(f::SetExpansion) = antidifferentiate(f, 3, 1)
# little helper function
ei(dim,i, coefficients) = tuple((coefficients*eye(Int,dim)[:,i])...)
# we allow the differentiation of one specific variable through the var argument
differentiate(f::SetExpansion, var, order) = differentiate(f, ei(dimension(f), var, order))
antidifferentiate(f::SetExpansion, var, order) = antidifferentiate(f, ei(dimension(f), var, order))

# To be implemented: Laplacian (needs multiplying functions)
## Δ(f::SetExpansion)

# This is just too cute not to do: f' is the derivative of f. Then f'' is the second derivative, and so on.
ctranspose(f::SetExpansion) = differentiate(f)
∫(f::SetExpansion) = antidifferentiate(f)

roots(f::SetExpansion) = roots(set(f), coefficients(f))

# Delegate generic operators
for op in (:extension_operator, :restriction_operator, :transform_operator)
    @eval $op(s1::SetExpansion, s2::SetExpansion) = $op(set(s1), set(s2))
end

for op in (:interpolation_operator, :evaluation_operator, :approximation_operator)
    @eval $op(s::SetExpansion) = $op(set(s))
end

differentiation_operator(s1::SetExpansion, s2::SetExpansion, var::Int...) = differentiation_operator(set(s1), set(s2), var...)
differentiation_operator(s1::SetExpansion, var::Int...) = differentiation_operator(set(s1), var...)


show(io::IO, fun::SetExpansion) = show_setexpansion(io, fun, set(fun))

function show_setexpansion(io::IO, fun::SetExpansion, fs::FunctionSet)
    println(io, "A ", dimension(fun), "-dimensional SetExpansion with ", length(coefficients(fun)), " degrees of freedom.")
    println(io, "Basis: ", name(fs))
end


# Invoke split_interval on the set and compute the coefficients such that
# the piecewise function agrees with the original one.
split_interval(s::SetExpansion, x) = SetExpansion(split_interval_expansion(set(s), coefficients(s), x)...)

##############################
# Arithmetics with expansions
##############################

# Arithmetics are only possible when the basis type is equal.
is_compatible{S<:FunctionSet}(s1::S, s2::S) = true
is_compatible(s1::FunctionSet, s2::FunctionSet) = false

Base.broadcast(abs, set::SetExpansion) = abs.(coefficients(set))

for op in (:+, :-)
    @eval function ($op)(s1::SetExpansion, s2::SetExpansion)
        # First check if the FunctionSets are arithmetically compatible
        @assert is_compatible(set(s1),set(s2))
        # If the sizes are equal, we can just operate on the coefficients.
        # If not, we have to extend the smaller set to the size of the larger set.
        if size(s1) == size(s2)
            SetExpansion(set(s1), $op(coefficients(s1), coefficients(s2)))
        elseif length(s1) < length(s2)
            s3 = extension_operator(set(s1), set(s2)) * s1
            SetExpansion(set(s2), $op(coefficients(s3), coefficients(s2)))
        else
            s3 = extension_operator(set(s2), set(s1)) * s2
            SetExpansion(set(s1), $op(coefficients(s1), coefficients(s3)))
        end
    end
end

function (*)(s1::SetExpansion, s2::SetExpansion)
    @assert is_compatible(set(s1),set(s2))
    (mset,mcoefficients) = (*)(set(s1),set(s2),coefficients(s1),coefficients(s2))
    SetExpansion(mset,mcoefficients)
end

(*)(op::AbstractOperator, e::SetExpansion) = apply(op, e)

(*)(a::Number, e::SetExpansion) = SetExpansion(set(e), a*coefficients(e))
(*)(e::SetExpansion, a::Number) = a*e

function apply(op::AbstractOperator, e::SetExpansion)
    @assert set(e) == set(src(op))

    SetExpansion(set(dest(op)), op * coefficients(e))
end
