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
(*)(op::AbstractOperator, f::Function) = op * sample(gridspace(src(op)), f)

function approximate(s::FunctionSet, f; options...)
    A = approximation_operator(s; options...)
    SetExpansion(s, A * sample(gridspace(src(A)), f))
end
