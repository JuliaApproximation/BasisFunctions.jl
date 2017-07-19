# augmented_set.jl

"""
A WeightedSet represents some function f(x) times an existing set.
"""
struct WeightedSet{T} <: DerivedSet{T}
    superset    ::  FunctionSet{T}
    weightfun
end

const WeightedSet1d{T <: Number} = WeightedSet{T}
const WeightedSet2d{T <: Number} = WeightedSet{SVector{2,T}}

const WeightedSetSpan{A, F <: WeightedSet} = Span{A,F}

weightfunction(set::WeightedSet) = set.weightfun

weightfunction(s::WeightedSetSpan) = weightfunction(set(s))

similar_set(set1::WeightedSet, set2::FunctionSet) = WeightedSet(set2, weightfunction(set1))

name(set::WeightedSet) = _name(set, superset(set), weightfunction(set))
_name(set::WeightedSet, superset, fun::Function) = "A weighted set based on " * name(superset)
_name(set::WeightedSet, superset, fun::AbstractFunction) = name(fun) * " * " * name(superset)

isreal(set::WeightedSet) = _isreal(set, superset(set), weightfunction(set))
_isreal(set::WeightedSet, superset, fun::AbstractFunction) = isreal(superset) && isreal(fun)
_isreal(set::WeightedSet, superset, fun::Function) = isreal(superset)

has_derivative(set::WeightedSet) = has_derivative(superset(set)) && has_derivative(weightfunction(set))
is_orthonormal(set::WeightedSet) = false
is_orthogonal(set::WeightedSet) = false
# We can not compute antiderivatives in general.
has_antiderivative(set::WeightedSet) = false

# We have to distinguish between 1d and higher-dimensional grids, since we
# have to splat the arguments to the weightfunction
eval_weight_on_grid(w, grid::AbstractGrid1d) = [w(x) for x in grid]

function eval_weight_on_grid(w, grid::AbstractGrid)
    # Perhaps the implementation here could be simpler, but [w(x...) for x in grid]
    # does not seem to respect the size of the grid, only its length
    a = zeros(float_type(eltype(grid)), size(grid))
    for i in eachindex(grid)
        a[i] = w(grid[i]...)
    end
    a
end

# Evaluating basis functions: we multiply by the function of the set
eval_element(set::WeightedSet, idx, x) = _eval_element(set, weightfunction(set), idx, x)
_eval_element(set::WeightedSet1d, w, idx, x) = w(x) * eval_element(superset(set), idx, x)
_eval_element(set::WeightedSet, w, idx, x) = w(x...) * eval_element(superset(set), idx, x)

# Evaluate an expansion: same story
eval_expansion(set::WeightedSet, coefficients, x) = _eval_expansion(set, weightfunction(set), coefficients, x)
# temporary, to remove an ambiguity
eval_expansion(set::WeightedSet, coefficients, x::AbstractGrid) = _eval_expansion(set, weightfunction(set), coefficients, x)

_eval_expansion(set::WeightedSet1d, w, coefficients, x::Number) = w(x) * eval_expansion(superset(set), coefficients, x)
_eval_expansion(set::WeightedSet, w, coefficients, x) = w(x...) * eval_expansion(superset(set), coefficients, x)

_eval_expansion(set::WeightedSet, w, coefficients, grid::AbstractGrid) =
    eval_weight_on_grid(w, grid) .* eval_expansion(superset(set), coefficients, grid)


# You can create an WeightedSet by multiplying a function with a set, using
# left multiplication.
# We support any Julia function:
(*)(f::Function, set::FunctionSet) = WeightedSet(set, f)
# and our own functors:
(*)(f::AbstractFunction, set::FunctionSet) = WeightedSet(set, f)

(*)(f::Function, s::Span) = Span(f*set(s), coeftype(s))
(*)(f::AbstractFunction, s::Span) = Span(f*set(s), coeftype(s))

weightfun_scaling_operator(dgs::DiscreteGridSpace1d, weightfunction) =
    DiagonalOperator(dgs, dgs, coeftype(dgs)[weightfunction(x) for x in grid(dgs)])

weightfun_scaling_operator(dgs::DiscreteGridSpace, weightfunction) =
    DiagonalOperator(dgs, dgs, coeftype(dgs)[weightfunction(x...) for x in grid(dgs)])

transform_to_grid_post(src::WeightedSetSpan, dest::DiscreteGridSpace, grid; options...) =
    weightfun_scaling_operator(dest, weightfunction(src))

transform_from_grid_pre(src::DiscreteGridSpace, dest::WeightedSetSpan, grid; options...) =
	inv(transform_to_grid_post(dest, src, grid; options...))


function derivative_space(src::WeightedSetSpan, order; options...)
    @assert order == 1

    s = superspan(src)
    f = weightfunction(src)
    f_prime = derivative(f)
    s_prime = derivative_space(s, order)
    (f_prime * s) ⊕ (f * s_prime)
end

# Assume order = 1...
function differentiation_operator(s1::WeightedSetSpan, s2::MultiSetSpan, order; options...)
    @assert order == 1
    @assert s2 == derivative_space(s1, order)

    I = IdentityOperator(s1, element(s2, 1))
    D = differentiation_operator(superspan(s1))
    DW = wrap_operator(s1, element(s2, 2), D)
    block_column_operator([I,DW])
end

function grid_evaluation_operator(set::WeightedSetSpan, dgs::DiscreteGridSpace, grid::AbstractGrid; options...)
    super_e = grid_evaluation_operator(superspan(set), dgs, grid; options...)
    D = weightfun_scaling_operator(dgs, weightfunction(set))
    D * wrap_operator(set, dgs, super_e)
end
