# augmented_set.jl

"""
A WeightedSet represents some function f(x) times an existing set.
"""
immutable WeightedSet{N,T} <: DerivedSet{N,T}
    superset    ::  FunctionSet{N,T}
    weightfun
end

weightfunction(set::WeightedSet) = set.weightfun

similar_set(set::WeightedSet, set2::FunctionSet) = WeightedSet(set2, weightfunction(set))

name(set::WeightedSet) = _name(set, superset(set), weightfunction(set))
_name(set::WeightedSet, superset, fun::Function) = "A weighted set based on " * name(superset)
_name(set::WeightedSet, superset, fun::AbstractFunction) = name(fun) * " * " * name(superset)

isreal(set::WeightedSet) = _isreal(set, superset(set), weightfunction(set))
_isreal(set::WeightedSet, superset, fun::AbstractFunction) = isreal(superset) && isreal(fun)
_isreal(set::WeightedSet, superset, fun::Function) = isreal(superset)

has_derivative(set::WeightedSet) = has_derivative(superset(set)) && has_derivative(weightfunction(set))

# We can not compute antiderivatives in general.
has_antiderivative(set::WeightedSet) = false

# Evaluating basis functions: we multiply by the function of the set
eval_element(set::WeightedSet, idx, x) = _eval_element(set, weightfunction(set), idx, x)
_eval_element(set::WeightedSet, w, idx, x) = w(x) * eval_element(superset(set), idx, x)

eval_expansion(set::WeightedSet, coefficients, x) = _eval_expansion(set, weightfunction(set), coefficients, x)

_eval_expansion(set::WeightedSet, w, coefficients, x) = w(x) * eval_expansion(superset(set), coefficients, x)

_eval_expansion(set::WeightedSet, w, coefficients, grid::AbstractGrid) = w.(grid) .* eval_expansion(superset(set), coefficients, grid)

# You can create an WeightedSet by multiplying a function with a set, using
# left multiplication.
# We support any Julia function:
(*)(f::Function, set::FunctionSet) = WeightedSet(set, f)
# and our own functors:
(*)(f::AbstractFunction, set::FunctionSet) = WeightedSet(set, f)


function transform_to_grid_post(src::WeightedSet, dest::DiscreteGridSpace, grid; options...)
    f = weightfunction(src)
    DiagonalOperator(dest, dest, [f(x) for x in grid])
end

transform_from_grid_pre(src::DiscreteGridSpace, dest::WeightedSet, grid; options...) =
	inv(transform_to_grid_post(dest, src, grid; options...))


function derivative_set(src::WeightedSet, order; options...)
    @assert order == 1

    s = superset(src)
    f = weightfunction(src)
    f_prime = derivative(f)
    s_prime = derivative_set(s, order)
    (f_prime * s) ⊕ (f * s_prime)
end

# Assume order = 1...
function differentiation_operator(s1::WeightedSet, s2::MultiSet, order; options...)
    @assert order == 1
    @assert s2 == derivative_set(s1, order)

    I = IdentityOperator(s1, element(s2, 1))
    D = differentiation_operator(superset(s1))
    DW = wrap_operator(s1, element(s2, 2), D)
    block_column_operator([I,DW])
end

function grid_evaluation_operator(set::WeightedSet, dgs::DiscreteGridSpace, grid::AbstractGrid; options...)
    super_e = grid_evaluation_operator(superset(set), dgs, grid; options...)
    f = weightfunction(set)
    T = eltype(dgs)
    D = DiagonalOperator(dgs, dgs, T[f(x) for x in grid])
    D * wrap_operator(set, dgs, super_e)
end
