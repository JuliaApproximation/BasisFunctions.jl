# approximation.jl

########################
# Generic approximation
########################

default_approximation_operator = leastsquares_operator

"""
The approximation_operator function returns an operator that can be used to approximate
a function in the function set. This operator maps a grid to a set of coefficients.
"""
function approximation_operator(b::FunctionSet; discrete = true, options...)
    if discrete
      if is_basis(b) && has_grid(b)
          interpolation_operator(b, grid(b); options...)
      else
          default_approximation_operator(b; options...)
      end
    else
      DualGram(b; options...)
    end
end



# Automatically sample a function if an operator is applied to it with a
# source that is a grid space.
function (*)(op::AbstractOperator, f::Function; options...)
  op * project(src(op), f; options...)
end
# general project on functionset, using inner products, is in functionset.jl
project(b::DiscreteGridSpace, f::Function, ELT = eltype(b); options...) = sample(grid(b), f, eltype(b))

approximate(s::FunctionSet, f::Function; discrete=true, options...) =
    SetExpansion(s, *(approximation_operator(s; discrete=discrete, options...),f; options...))
