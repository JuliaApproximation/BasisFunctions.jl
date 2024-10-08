
"Check whether a set of coefficients is compatible with the function set."
compatible_coefficients(dict::Dictionary, coefficients) =
    length(dict) == length(coefficients)

"""
An `Expansion` describes a function using its expansion coefficients in a certain
dictionary.

An expansion acts both like an array, the array of coefficients, and like a
function.

Parameters:
- D is the dictionary.
- C is the type of the expansion coefficients
"""
struct Expansion{S,T,D<:Dictionary{S,T},C}
    dictionary      ::  D
    coefficients    ::  C

    function Expansion{S,T,D,C}(dict::D, coefficients::C) where {S,T,D<:Dictionary{S,T},C}
        @assert compatible_coefficients(dict, coefficients)
        new(dict, coefficients)
    end
end

Expansion(dict::Dictionary) = Expansion(dict, zeros(dict))
Expansion(dict::Dictionary, coef) =
    Expansion{domaintype(dict),codomaintype(dict),typeof(dict),typeof(coef)}(dict, coef)

export Expansion1d, Expansion2d, Expansion3d, Expansion4d
# Warning: not all 2d function sets have SVector{2,T} type, they could have (S,T) type
const Expansion1d{S <: Number,T,D,C} = Expansion{S,T,D,C}
const Expansion2d{S <: Number,T,D,C} = Expansion{SVector{2,S},T,D,C}
const Expansion3d{S <: Number,T,D,C} = Expansion{SVector{3,S},T,D,C}
const Expansion4d{S <: Number,T,D,C} = Expansion{SVector{4,S},T,D,C}

@forward Expansion.dictionary domaintype, prectype, dimension,
                                size, Span, support, interpolation_grid, numtype,
                                ncomponents,
                                measure

@forward Expansion.coefficients length, eachindex, getindex, setindex!
codomaintype(e::Expansion) = promote_type(codomaintype(dictionary(e)), eltype(coefficients(e)))
isreal(e::Expansion) = isreal(e.dictionary) && isreal(eltype(coefficients(e)))

"""
    expansion(dict::Dictionary, coefficients)

An expansion of a dictionary with given coefficients.
"""
function expansion(dict::Dictionary, coefficients)
    coef = native_coefficients(dict, coefficients)
    T = promote_type(coefficienttype(dict), eltype(coef))
    Expansion(ensure_coefficienttype(T, dict), coef)
end

similar(e::Expansion, coefficients) = Expansion(dictionary(e), coefficients)

dictionary(e::Expansion) = e.dictionary
coefficients(e::Expansion) = e.coefficients

"Map the expansion using the given map."
mapped_expansion(e::Expansion, m) =
    Expansion(mapped_dict(dictionary(e), m), coefficients(e))

export random_expansion
random_expansion(dict::Dictionary) = Expansion(dict, rand(dict))

# For expansions of composite types, return a Expansion of a subdict
component(e::Expansion, i) = Expansion(component(e.dictionary, i), component(e.coefficients, i))
components(e::Expansion) = iscomposite(e.dictionary) && iscomposite(e.coefficients) ?
    map(Expansion, components(e.dictionary), components(e.coefficients)) :
    ()

# This indirect call enables dispatch on the type of the dict of the expansion
(e::Expansion)(x; options...) =
    call_expansion(e, dictionary(e), coefficients(e), x; options...)
(e::Expansion)(x, y; options...) =
    call_expansion(e, dictionary(e), coefficients(e), SVector(x, y); options...)
(e::Expansion)(x, y, z; options...) =
    call_expansion(e, dictionary(e), coefficients(e), SVector(x, y, z); options...)
(e::Expansion)(x, y, z, t; options...) =
    call_expansion(e, dictionary(e), coefficients(e), SVector(x, y, z, t); options...)
(e::Expansion)(x, y, z, t, u...; options...) =
    call_expansion(e, dictionary(e), coefficients(e), SVector(x, y, z, t, u...); options...)

call_expansion(e::Expansion, dict::Dictionary, coefficients, x; options...) =
    eval_expansion(dict, coefficients, x; options...)

eval_expansion(e::Expansion, x) =
    eval_expansion(dictionary(e), coefficients(e), x)
unsafe_eval_expansion(e::Expansion, x) =
    unsafe_eval_expansion(dictionary(e), coefficients(e), x)

function differentiate(e::Expansion, order = difforder(dictionary(e)); options...)
    op = differentiation(codomaintype(e), dictionary(e), order; options...)
    Expansion(dest(op), apply(op,e.coefficients))
end

function antidifferentiate(e::Expansion, order = 1; options...)
    op = antidifferentiation(codomaintype(e), dictionary(e); options...)
    Expansion(dest(op), apply(op,e.coefficients))
end

integral(f::Expansion) = integral(f, measure(f))

# Shorthands for partial derivatives
∂x(f::Expansion) = differentiate(f, 1, 1)
∂y(f::Expansion) = differentiate(f, 2, 1)
∂z(f::Expansion) = differentiate(f, 3, 1)
# Shorthands for partial integrals
∫∂x(f::Expansion) = antidifferentiate(f, 1, 1)
∫∂y(f::Expansion) = antidifferentiate(f, 2, 1)
∫∂z(f::Expansion) = antidifferentiate(f, 3, 1)
# little helper function
ei(dim,i, coefficients) = tuple((coefficients*Matrix{Int}(I, dim, dim)[:,i])...)
# we allow the differentiation of one specific variable through the var argument
differentiate(f::Expansion, var, order) = differentiate(f, ei(dimension(f), var, order))
antidifferentiate(f::Expansion, var, order) = antidifferentiate(f, ei(dimension(f), var, order))

function Base.:^(f::Expansion, i::Int)
    @assert i >= 0
    if i == 1
        f
    elseif i == 2
        f * f
    else
        f^(i-1) * f
    end
end

# To be implemented: Laplacian (needs multiplying functions)
## Δ(f::Expansion)

# This is just too cute not to do: f' is the derivative of f. Then f'' is the second derivative, and so on.
adjoint(f::Expansion) = differentiate(f)

∫(f::Expansion) = antidifferentiate(f)

roots(f::Expansion) = expansion_roots(dictionary(f), coefficients(f))

Base.@deprecate roots(dict::Dictionary, coef::AbstractVector) expansion_roots(dict, coef)

show(io::IO, mime::MIME"text/plain", fun::Expansion) = composite_show(io, mime, fun)
Display.displaystencil(fun::Expansion) = ["Expansion(", dictionary(fun), ", ", coefficients(fun), ")"]

# Invoke split_interval on the set and compute the coefficients such that
# the piecewise function agrees with the original one.
split_interval(s::Expansion, x) = Expansion(split_interval_expansion(dictionary(s), coefficients(s), x)...)

##############################
# Arithmetics with expansions
##############################

# Arithmetics are only possible when the basis type is equal.
iscompatible(d1::D, d2::D) where {D<:Dictionary} = true
iscompatible(d1::Dictionary, d2::Dictionary) = false

iscompatible(e1::Expansion, e2::Expansion) = iscompatible(dictionary(e1),dictionary(e2))

compatible_dictionary(d1::Dictionary, d2::Dictionary) =
    default_compatible_dictionary(d1, d2)

function default_compatible_dictionary(d1::Dictionary, d2::Dictionary)
    if d1 == d2
        d1
    else
        # coefficient type and size may differ
        T = promote_type(domaintype(d1), domaintype(d2))
        newsize = map(max, size(d1), size(d2))
        common_dict = similar(d1, T, newsize)
        if isreal(coefficienttype(d1)) && isreal(coefficienttype(d2))
            common_dict
        else
            complex(common_dict)
        end
    end
end

"""
If two dictionaries are compatible, then they have to be convertible to a
common dictionary.
"""
function conversion_to_compatible_dicts(d1::Dictionary, d2::Dictionary)
    if d1 == d2
        return IdentityOperator(d1), IdentityOperator(d2)
    end
    common_dict = compatible_dictionary(d1, d2)
    conversion(d1, common_dict), conversion(d2, common_dict)
end

function to_compatible_expansions(e1::Expansion, e2::Expansion)
    if dictionary(e1) == dictionary(e2)
        e1, e2
    else
        E1, E2 = conversion_to_compatible_dicts(dictionary(e1), dictionary(e2))
        E1*e1, E2*e2
    end
end

function (+)(f::Expansion, g::Expansion)
    dict, coef = expansion_sum(dictionary(f),dictionary(g),coefficients(f),coefficients(g))
    Expansion(dict, coef)
end
(-)(e1::Expansion, e2::Expansion) = e1 + similar(e2, -coefficients(e2))

(-)(e::Expansion) = similar(e, -coefficients(e))

@deprecate dict_multiply expansion_multiply
expansion_multiply(dict1, dict2, coef1, coef2) =
    expansion_multiply1(dict1, dict2, coef1, coef2)
expansion_multiply1(dict1, dict2, coef1, coef2) =
    expansion_multiply2(dict1, dict2, coef1, coef2)
expansion_multiply2(dict1, dict2, coef1, coef2) =
    error("Multiplication of expansions in these dictionaries is not implemented.")

expansion_sum(dict1, dict2, coef1, coef2) =
    expansion_sum1(dict1, dict2, coef1, coef2)
expansion_sum1(dict1, dict2, coef1, coef2) =
    expansion_sum2(dict1, dict2, coef1, coef2)
function expansion_sum2(dict1, dict2, coef1, coef2)
    if iscompatible(dict1, dict2)
        default_expansion_sum(dict1, dict2, coef1, coef2)
    else
        error("Summation of expansions in these dictionaries is not implemented.")
    end
end

function default_expansion_sum(dict1, dict2, coef1, coef2)
    @assert iscompatible(dict1, dict2)
    e1, e2 = to_compatible_expansions(Expansion(dict1, coef1), Expansion(dict2, coef2))
    dictionary(e1), coefficients(e1)+coefficients(e2)
end

function (*)(f::Expansion, g::Expansion)
    dict, coef = expansion_multiply(dictionary(f),dictionary(g),coefficients(f),coefficients(g))
    Expansion(dict, coef)
end

(*)(op::DictionaryOperator, e::Expansion) = apply(op, e)

Base.complex(e::Expansion) = Expansion(complex(dictionary(e)), complex(coefficients(e)))
function Base.real(e::Expansion)
    dict, coef = expansion_real(dictionary(e), coefficients(e))
    Expansion(dict, coef)
end
function Base.imag(e::Expansion)
    dict, coef = expansion_imag(dictionary(e), coefficients(e))
    Expansion(dict, coef)
end
function expansion_real(dict::Dictionary, coefficients)
    @assert isreal(dict)
    real(dict), real(coefficients)
end
function expansion_imag(dict::Dictionary, coefficients)
    @assert isreal(dict)
    real(dict), imag(coefficients)
end

Base.one(e::Expansion) = similar(e, coefficients_of_one(dictionary(e)))

expansion_of_one(dict::Dictionary) = Expansion(dict, coefficients_of_one(dict))
expansion_of_x(dict::Dictionary) = Expansion(dict, coefficients_of_x(dict))

for op in (:+, :-)
    @eval Base.$op(a::Number, e::Expansion) = $op(a*one(e), e)
    @eval Base.$op(e::Expansion, a::Number) = $op(e, a*one(e))
end
Base.:*(e::Expansion, a::Number) = expansion(dictionary(e), coefficients(e)*a)
Base.:*(a::Number, e::Expansion) = expansion(dictionary(e), a*coefficients(e))
Base.:/(e::Expansion, a::Number) = expansion(dictionary(e), coefficients(e)/a)

apply(op::DictionaryOperator, e::Expansion) = Expansion(dest(op), op * coefficients(e))

# TODO: we may not want to identify an Expansion with its array of coefficients
iterate(e::Expansion) = iterate(coefficients(e))
iterate(e::Expansion, state) = iterate(coefficients(e), state)

Base.collect(e::Expansion) = coefficients(e)

Base.BroadcastStyle(e::Expansion) = Base.Broadcast.DefaultArrayStyle{dimension(e)}()

# Interoperate with basis functions.
# TODO: implement cleaner using promotion mechanism

Base.:*(φ1::AbstractBasisFunction, φ2::AbstractBasisFunction) = expansion(φ1) * expansion(φ2)
Base.:*(f::Expansion, φ2::AbstractBasisFunction) = f * expansion(φ2)
Base.:*(φ1::AbstractBasisFunction, f::Expansion) = expansion(φ1) * f

Base.:*(A::Number, φ::AbstractBasisFunction) = A * expansion(φ)
Base.:*(φ::AbstractBasisFunction, A::Number) = A * expansion(φ)

Base.:+(φ1::AbstractBasisFunction, φ2::AbstractBasisFunction) = expansion(φ1) + expansion(φ2)
Base.:+(f::Expansion, φ2::AbstractBasisFunction) = f + expansion(φ2)
Base.:+(φ1::AbstractBasisFunction, f::Expansion) = expansion(φ1) + f

Base.:-(φ1::AbstractBasisFunction, φ2::AbstractBasisFunction) = expansion(φ1) - expansion(φ2)
Base.:-(f::Expansion, φ2::AbstractBasisFunction) = f - expansion(φ2)
Base.:-(φ1::AbstractBasisFunction, f::Expansion) = expansion(φ1) - f


