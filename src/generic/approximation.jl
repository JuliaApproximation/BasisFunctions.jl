# approximation.jl

########################
# Generic approximation
########################

default_approximation_operator = leastsquares_operator

"""
The approximation_operator function returns an operator that can be used to approximate
a function in the function set. This operator maps a grid to a set of coefficients.
"""
approximation_operator(s; discrete = true, options...) =
  discrete?
    discrete_approximation_operator(s; options...) :
    continuous_approximation_operator(s; options...)

function discrete_approximation_operator(s::Span; options...)
    if is_basis(set(s)) && has_grid(set(s))
        interpolation_operator(s, grid(s); options...)
    else
        default_approximation_operator(s; options...)
    end
end

continuous_approximation_operator(s::Span; options...) = DualGram(s)

# Automatically sample a function if an operator is applied to it with a
# source that is a grid space.
function (*)(op::AbstractOperator, f::Function; discrete=nothing, solver=nothing, cutoff=nothing, options...)
  op * project(src(op), f; options...)
end

# general project on functionset, using inner products, is in functionset.jl
project(s::DiscreteGridSpace, f::Function; options...) = sample(s, f)

approximate(s::Span, f::Function; options...) =
    SetExpansion(set(s), *(approximation_operator(s; options...),f; options...))

"""
The 2-argument approximation_operator exists to allow you to transform any
FunctionSet coefficients to any other FunctionSet coefficients, including
Discrete Grid Spaces.
If a transform exists, the transform is used.
"""
function approximation_operator(A::Span, B::Span, options...)
    if has_transform(A, B)
        return transform_operator(A, B; options...)
    else
        error("Don't know a transformation from ", A, " to ", B)
    end
end
