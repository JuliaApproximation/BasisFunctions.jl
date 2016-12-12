# approximation.jl

########################
# Generic approximation
########################

default_approximation_operator = leastsquares_operator

"""
The approximation_operator function returns an operator that can be used to approximate
a function in the function set. This operator maps a grid to a set of coefficients.
"""
function approximation_operator(b::FunctionSet; options...)
    if is_basis(b) && has_grid(b)
        interpolation_operator(b, grid(b); options...)
    else
        default_approximation_operator(b; options...)
    end
end



# Automatically sample a function if an operator is applied to it with a
# source that has a grid
(*)(op::AbstractOperator, f::Function) = op * sample(grid(src(op)), f, eltype(src(op)))

approximate(s::FunctionSet, f::Function; options...) =
    SetExpansion(s, approximation_operator(s; options...) * f)
