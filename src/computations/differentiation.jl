
# from computations/differentiation.jl
export differentiation,
    antidifferentiation,
    derivative_dict,
    antiderivative_dict

"""
The differentation function returns an operator that can be used to differentiate
a function in the dictionary, with the result as an expansion in a second dictionary.

The differentiation operator is efficient for some combinations of source and
destination dictionary. If a dictionary supports at least one differentiation
operator, than `hasderivative(dict)` is true and `derivative_dict(dict)` returns
the destination dictionary.

The order of the differentation can be passed as an optional second argument.

Examples are:
```
differentiation(Φ::Dictionary)
```
This will use the default destination dictionary, order 1 and dimension 1.
Another example is:
```
differentiation(Φ, 1)
```
or
```
differentiation(Φ, (0,1))
```
"""
differentiation

"""
The antidifferentiation function returns an operator that can be used to find the antiderivative
of a function in the dictionary, with the result an expansion in a second dictionary.

See also: `differentiation`.
"""
antidifferentiation


orderiszero(order::Int) = order==0
orderiszero(order) = sum(order) == 0

"Determine the order of differentiation from the arguments given by the user."
difforder(Φ::Dictionary; order = nothing, dim = nothing, options...) =
    _difforder(Φ, order, dim)

# Integer order is given: we use that
_difforder(Φ::Dictionary, order::Int, dim::Nothing) = order
# Integer order and dimension are given: we make a tuple of the form (0,0,order,0)
_difforder(Φ::Dictionary, order::Int, dim::Int) = order .* dimension_tuple(dimension(Φ), dim)
# Nothing is given: assume integer order 1 and try again
_difforder(Φ::Dictionary, order::Nothing, dim::Nothing) =
    dimension(Φ) == 1 ? _difforder(Φ, 1, dim) : _difforder(Φ, 1, 1)
# Only dimension is given: use order 1 in that dimension.
_difforder(Φ::Dictionary, order::Nothing, dim::Int) = _difforder(Φ, 1, dim)
# None of the above: we don't know what to do.
_difforder(Φ::Dictionary, order, dim) = error("Supplied derivative order and dimension not understood.")

# Assign a default order if none is given
derivative_dict(Φ::Dictionary; options...) =
    derivative_dict(Φ, difforder(Φ; options...); options...)
antiderivative_dict(Φ::Dictionary; options...) =
    antiderivative_dict(Φ, difforder(Φ; options...); options...)

# Insert the operator eltype if it is not given
differentiation(dict::Dictionary, args...; options...) =
    differentiation(operatoreltype(dict), dict, args...; options...)
differentiation(src::Dictionary, dest::Dictionary, args...; options...) =
    differentiation(operatoreltype(src, dest), src, dest, args...; options...)

antidifferentiation(dict::Dictionary, args...; options...) =
    antidifferentiation(operatoreltype(dict), dict, args...; options...)
antidifferentiation(src::Dictionary, dest::Dictionary, args...; options...) =
    antidifferentiation(operatoreltype(src, dest), src, dest, args...; options...)



# Assign a default order if none is given
differentiation(::Type{T}, src::Dictionary; options...) where {T} =
    differentiation(T, src, difforder(src; options...); options...)
differentiation(::Type{T}, src::Dictionary, dest::Dictionary; options...) where {T} =
    differentiation(T, src, dest, difforder(src; options...); options...)

antidifferentiation(::Type{T}, src::Dictionary; options...) where {T} =
    antidifferentiation(T, src, difforder(src; options...); options...)
antidifferentiation(::Type{T}, src::Dictionary, dest::Dictionary; options...) where {T} =
    antidifferentiation(T, src, dest, difforder(src; options...); options...)


function differentiation(::Type{T}, src::Dictionary, order::Int; options...) where {T}
    if orderiszero(order)
        IdentityOperator{T}(src)
    else
        if hasderivative(src, order)
            dest = derivative_dict(src, order; options...)
            differentiation(T, src, dest, order; options...)
        else
            # If order==1 and the test above failed, there is an issue with the dictionary.
            order == 1 && error("Dictionary does not support differentiation.")
            # From here on, we assume that order > 1
            @assert order > 1
            dest1 = derivative_dict(src, 1; options...)
            D1 = differentiation(T, src, dest1, 1; options...)
            D2 = differentiation(T, dest1, order-1; options...)
            D2*D1
        end
    end
end

function differentiation(::Type{T}, src::Dictionary, order; options...) where {T}
    if orderiszero(order)
        IdentityOperator{T}(src)
    else
        @assert hasderivative(src, order)
        dest = derivative_dict(src, order; options...)
        differentiation(T, src, dest, order; options...)
    end
end


function antidifferentiation(::Type{T}, src::Dictionary, order::Int; options...) where {T}
    if orderiszero(order)
        IdentityOperator{T}(src)
    else
        if hasantiderivative(src, order)
            dest = antiderivative_dict(src, order; options...)
            antidifferentiation(T, src, dest, order; options...)
        else
            # If order==1 and the test above failed, there is an issue with the dictionary.
            order == 1 && error("Dictionary does not support antidifferentiation.")
            # From here on, we assume that order > 1
            @assert order > 1
            dest1 = antiderivative_dict(src, 1; options...)
            D1 = antidifferentiation(T, src, dest1, 1; options...)
            D2 = antidifferentiation(T, dest1, order-1; options...)
            D2*D1
        end
    end
end

function antidifferentiation(::Type{T}, src::Dictionary, order; options...) where {T}
    if orderiszero(order)
        IdentityOperator{T}(src)
    else
        @assert hasantiderivative(src, order)
        dest = antiderivative_dict(src, order; options...)
        antidifferentiation(T, src, dest, order; options...)
    end
end
