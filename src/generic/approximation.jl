# approximation.jl

########################
# Generic approximation
########################

default_approximation_operator = leastsquares_operator

"""
The approximation_operator function returns an operator that can be used to approximate
a function in the function set. This operator maps a grid to a set of coefficients.
"""
approximation_operator(b; discrete=true, options...) =
  discrete?
    discrete_approximation_operator(b; options...) :
    continuous_approximation_operator(b; options...)

function discrete_approximation_operator(b::FunctionSet; options...)
    if is_basis(b) && has_grid(b)
        interpolation_operator(b, grid(b); options...)
    else
        default_approximation_operator(b; options...)
    end
end

continuous_approximation_operator(b::FunctionSet; options...) = DualGram(b)

# Automatically sample a function if an operator is applied to it with a
# source that is a grid space.
function (*)(op::AbstractOperator, f::Function; discrete=nothing, solver=nothing, cutoff=nothing, options...)
  op * project(src(op), f; options...)
end

# general project on functionset, using inner products, is in functionset.jl
project(b::DiscreteGridSpace, f::Function; options...) = sample(b, f)

approximate(s, f::Function; options...) =
    SetExpansion(s, *(approximation_operator(s; options...),f; options...))
