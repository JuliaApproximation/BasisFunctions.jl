# Code to compute with functions

const FUN = Union{<:TypedFunction,<:Expansion}

"""
Promote the given two functions such that they have compatible type, suitable for further
operations such as addition and multiplication.

In most cases the results are two expansions with a dictionary of the same family.

See also: [`funpromote_samelength`](@ref)
"""
funpromote(f, g) = funpromote(expansion(f), expansion(g))

funpromote(f::Expansion, g::Expansion) =
    funpromote(dictionary(f), dictionary(g), coefficients(f), coefficients(g))

funpromote(d1, d2, c1, c2) = _funpromote(d1, d2, c1, c2, promote_convertible(d1, d2)...)
_funpromote(d1::D1, d2::D2, c1, c2, d3::D1, d4::D2) where {D1,D2} =
    expansion(d1, c1), expansion(d2, c2)    # no conversion happened
function _funpromote(d1, d2, c1, c2, d3, d4)
    C1 = conversion(d1, d3)
    C2 = conversion(d2, d4)
    C1 * expansion(d1, c1), C2 * expansion(d2, c2)
end


Base.:+(f::FUN, g::FUN) = expansion_sum(funpromote(f, g)...)
Base.:-(f::FUN) = -expansion(f)
Base.:-(f::Expansion) = expansion(dictionary(f), -coefficients(f))
Base.:-(f::FUN, g::FUN) = f + (-g)

expansion_sum(f::Expansion, g::Expansion) = expansion_sum(dictionary(f), dictionary(g), coefficients(f), coefficients(g))

function expansion_sum(d1, d2, c1, c2)
    # we assume that d1 and d2 are compatible, just looking for a possible size mismatch
    if size(d1) == size(d2)
        expansion(d1, c1+c2)
    else
        if length(d1) < length(d2)
            e12 = extension(d1, d2)
            expansion(d2, e12*c1 + c2)
        elseif length(d1) > length(d2)
            e21 = extension(d2, d1)
            expansion(d1, c1 + e21*c2)
        else
            error("Sizes of dictionaries don't match, but lengths do.")
        end
    end
end


Base.:*(f::FUN, g::FUN) = expansion_multiply(funpromote(f, g)...)

function expansion_multiply(f::Expansion, g::Expansion)
    dict, coef = expansion_multiply(dictionary(f), dictionary(g), coefficients(f), coefficients(g))
    expansion(dict, coef)
end

# Multiplication by scalars

Base.:*(f::FUN, a::Number) = a*f
Base.:*(a::Number, f::FUN) = a*expansion(f)
Base.:*(a::Number, f::Expansion) = expansion(dictionary(f), a*coefficients(f))

Base.:/(f::FUN, a::Number) = expansion(f) / a
Base.:/(f::Expansion, a::Number) = expansion(dictionary(f), coefficients(f)/a)

for op in (:+, :-)
    @eval Base.$op(a::Number, f::FUN) = $op(a*one(expansion(f)), f)
    @eval Base.$op(f::FUN, a::Number) = $op(f, a*one(expansion(f)))
end

# exponentiation

function Base.:^(f::FUN, i::Int)
    @assert i >= 0
    if i == 1
        f
    elseif i == 2
        f * f
    else
        f^(i-1) * f
    end
end
