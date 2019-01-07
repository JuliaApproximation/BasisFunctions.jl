
########################
# Generic approximation
########################

default_approximation_operator = leastsquares_operator

"""
The approximation_operator function returns an operator that can be used to approximate
a function in the function set. This operator maps a grid to a set of coefficients.
"""
approximation_operator(dict; discrete = true, options...) =
  (discrete) ?
    discrete_approximation_operator(dict; options...) :
    continuous_approximation_operator(dict; options...)

function discrete_approximation_operator(dict::Dictionary; options...)
    if is_basis(dict) && has_interpolationgrid(dict)
        interpolation_operator(dict, interpolation_grid(dict); options...)
    else
        default_approximation_operator(dict; options...)
    end
end

continuous_approximation_operator(dict::Dictionary; options...) = inv(gramoperator(dict; options...))

# Automatically sample a function if an operator is applied to it
function (*)(op::DictionaryOperator, f::Function)
    op * project(src(op), f)
end

project(dict::GridBasis, f::Function; options...) = sample(dict, f)

approximate(dict::Dictionary, f::Function; options...) =
    Expansion(dict, approximation_operator(dict; options...) * f)

"""
The 2-argument approximation_operator exists to allow you to transform any
Dictionary coefficients to any other Dictionary coefficients, including
Discrete Grid Spaces.
If a transform exists, the transform is used.
"""
function approximation_operator(A::Dictionary, B::Dictionary, options...)
    if has_transform(A, B)
        return transform_operator(A, B; options...)
    else
        error("Don't know a transformation from ", A, " to ", B)
    end
end
