# expansions.jl

"Check whether a set of coefficients is compatible with the function set."
compatible_coefficients(dict::Dictionary, coefficients) =
    length(dict) == length(coefficients)

"""
An `Expansion` describes a function using its expansion coefficients in a certain
dictionary.

Parameters:
- D is the dictionary.
- C is the type of the expansion coefficients
"""
struct Expansion{D,C}
    dictionary      ::  D
    coefficients    ::  C

    function Expansion{D,C}(dict, coefficients) where {D,C}
        @assert compatible_coefficients(dict, coefficients)
        new(dict, coefficients)
    end
end

# Some constructors
Expansion(dict::Dictionary) = Expansion(dict, zeros(dict))
Expansion(dict::Dictionary, coef) = Expansion{typeof(dict),typeof(coef)}(dict, coef)

expansion(dict::Dictionary, coefficients) =
    Expansion(dict, native_coefficients(dict, coefficients))

similar(e::Expansion, coefficients) = Expansion(dictionary(e), coefficients)

eltype(::Type{Expansion{D,C}}) where {D,C} = eltype(C)

dictionary(e::Expansion) = e.dictionary

coefficients(e::Expansion) = e.coefficients

Span(e::Expansion) = Span(dictionary(e))

random_expansion(dict::Dictionary) = Expansion(dict, rand(dict))

# For expansions of composite types, return a Expansion of a subdict
element(e::Expansion, i) = Expansion(element(e.dictionary, i), element(e.coefficients, i))

elements(e::Expansion) = map(Expansion, elements(e.dictionary), elements(e.coefficients))

# Delegation of methods
for op in (:length, :size, :support, :grid)
    @eval $op(e::Expansion) = $op(dictionary(e))
end

# Delegation of property methods
for op in (:numtype, :dimension, :nb_elements)
    @eval $op(s::Expansion) = $op(dictionary(s))
end

has_basis(e::Expansion) = is_basis(dictionary(e))
has_frame(e::Expansion) = is_frame(dictionary(e))

eachindex(e::Expansion) = eachindex(coefficients(e))

getindex(e::Expansion, i...) = e.coefficients[i...]

setindex!(e::Expansion, v, i...) = (e.coefficients[i...] = v)


# This indirect call enables dispatch on the type of the dict of the expansion
(e::Expansion)(x; options...) = call_expansion(e, dictionary(e), coefficients(e), x; options...)
(e::Expansion)(x, y) = call_expansion(e, dictionary(e), coefficients(e), SVector(x, y))
(e::Expansion)(x, y, z) = call_expansion(e, dictionary(e), coefficients(e), SVector(x, y, z))
(e::Expansion)(x, y, z, t) = call_expansion(e, dictionary(e), coefficients(e), SVector(x, y, z, t))
(e::Expansion)(x, y, z, t, u...) = call_expansion(e, dictionary(e), coefficients(e), SVector(x, y, z, t, u...))

call_expansion(e::Expansion, dict::Dictionary, coefficients, x; options...) =
    eval_expansion(dict, coefficients, x; options...)

function differentiate(e::Expansion, order=1)
    op = differentiation_operator(dictionary(e), order)
    Expansion(dest(op), apply(op,e.coefficients))
end

function antidifferentiate(e::Expansion, order=1)
    op = antidifferentiation_operator(dictionary(e), order)
    Expansion(dest(op), apply(op,e.coefficients))
end

Base.broadcast(e::Expansion, grid::AbstractGrid) = eval_expansion(dictionary(e), coefficients(e), grid)

Base.broadcast{T}(e::Expansion, x::LinSpace{T}) = broadcast(e, EquispacedGrid(x))

# Shorthands for partial derivatives
∂x(f::Expansion) = differentiate(f, 1, 1)
∂y(f::Expansion) = differentiate(f, 2, 1)
∂z(f::Expansion) = differentiate(f, 3, 1)
# Shorthands for partial integrals
∫∂x(f::Expansion) = antidifferentiate(f, 1, 1)
∫∂y(f::Expansion) = antidifferentiate(f, 2, 1)
∫∂z(f::Expansion) = antidifferentiate(f, 3, 1)
# little helper function
ei(dim,i, coefficients) = tuple((coefficients*eye(Int,dim)[:,i])...)
# we allow the differentiation of one specific variable through the var argument
differentiate(f::Expansion, var, order) = differentiate(f, ei(dimension(f), var, order))
antidifferentiate(f::Expansion, var, order) = antidifferentiate(f, ei(dimension(f), var, order))

# To be implemented: Laplacian (needs multiplying functions)
## Δ(f::Expansion)

# This is just too cute not to do: f' is the derivative of f. Then f'' is the second derivative, and so on.
ctranspose(f::Expansion) = differentiate(f)
∫(f::Expansion) = antidifferentiate(f)

roots(f::Expansion) = roots(dictionary(f), coefficients(f))

# Delegate generic operators
for op in (:extension_operator, :restriction_operator, :transform_operator)
    @eval $op(s1::Expansion, s2::Expansion) = $op(dictionary(s1), dictionary(s2))
end

for op in (:interpolation_operator, :evaluation_operator, :approximation_operator)
    @eval $op(s::Expansion) = $op(dictionary(s))
end

differentiation_operator(s1::Expansion, s2::Expansion, var::Int...) = differentiation_operator(dictionary(s1), dictionary(s2), var...)
differentiation_operator(s1::Expansion, var::Int...) = differentiation_operator(dictionary(s1), var...)


show(io::IO, fun::Expansion) = show_setexpansion(io, fun, dictionary(fun))

function show_setexpansion(io::IO, fun::Expansion, fs::Dictionary)
    println(io, "A ", dimension(fun), "-dimensional Expansion with ", length(coefficients(fun)), " degrees of freedom.")
    println(io, "Basis: ", name(fs))
end


# Invoke split_interval on the set and compute the coefficients such that
# the piecewise function agrees with the original one.
split_interval(s::Expansion, x) = Expansion(split_interval_expansion(dictionary(s), coefficients(s), x)...)

##############################
# Arithmetics with expansions
##############################

# Arithmetics are only possible when the basis type is equal.
is_compatible{S<:Dictionary}(s1::S, s2::S) = true
is_compatible(s1::Dictionary, s2::Dictionary) = false

Base.broadcast(abs, e::Expansion) = abs.(coefficients(e))

for op in (:+, :-)
    @eval function ($op)(s1::Expansion, s2::Expansion)
        # First check if the Dictionarys are arithmetically compatible
        @assert is_compatible(dictionary(s1),dictionary(s2))
        # If the sizes are equal, we can just operate on the coefficients.
        # If not, we have to extend the smaller set to the size of the larger set.
        if size(s1) == size(s2)
            Expansion(dictionary(s1), $op(coefficients(s1), coefficients(s2)))
        elseif length(s1) < length(s2)
            s3 = extension_operator(dictionary(s1), dictionary(s2)) * s1
            Expansion(dictionary(s2), $op(coefficients(s3), coefficients(s2)))
        else
            s3 = extension_operator(dictionary(s2), dictionary(s1)) * s2
            Expansion(dictionary(s1), $op(coefficients(s1), coefficients(s3)))
        end
    end
end

function (*)(s1::Expansion, s2::Expansion)
    @assert is_compatible(dictionary(s1),dictionary(s2))
    (mset,mcoefficients) = (*)(dictionary(s1),dictionary(s2),coefficients(s1),coefficients(s2))
    Expansion(mset,mcoefficients)
end

(*)(op::DictionaryOperator, e::Expansion) = apply(op, e)

(*)(a::Number, e::Expansion) = Expansion(dictionary(e), a*coefficients(e))
(*)(e::Expansion, a::Number) = a*e

function apply(op::DictionaryOperator, e::Expansion)
    #@assert dictionary(e) == dictionary(src(op))

    Expansion(dest(op), op * coefficients(e))
end
