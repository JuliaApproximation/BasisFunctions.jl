
"""
A `WeightedDict` represents some function f(x) times an existing dictionary.
"""
struct WeightedDict{S,T} <: DerivedDict{S,T}
    superdict   ::  Dictionary{S}
    weightfun
end

WeightedDict(superdict::Dictionary{S,T},weightfun) where {S,T} = WeightedDict{S,T}(superdict,weightfun)

WeightedDict(superdict::Dictionary{S},weightfun,T) where S = WeightedDict{S,T}(superdict,weightfun)

const WeightedDict1d{S <: Number,T} = WeightedDict{S,T}
const WeightedDict2d{S <: Number,T} = WeightedDict{SVector{2,S},T}
const WeightedDict3d{S <: Number,T} = WeightedDict{SVector{3,S},T}
const WeightedDict4d{S <: Number,T} = WeightedDict{SVector{4,S},T}



weightfunction(set::WeightedDict) = set.weightfun

similar_dictionary(set1::WeightedDict, set2::Dictionary) = WeightedDict(set2, weightfunction(set1))

name(set::WeightedDict) = "Weightfunction " * string(weightfunction(set))

## _name(set::WeightedDict, superdict, fun::Function) = "A weighted dict based on " * name(superdict)
## _name(set::WeightedDict, superdict, fun::AbstractFunction) = name(fun) * " * " * name(superdict)

isreal(set::WeightedDict) = _isreal(set, superdict(set), weightfunction(set))
_isreal(set::WeightedDict, superdict, fun::AbstractFunction) = isreal(superdict) && isreal(fun)
_isreal(set::WeightedDict, superdict, fun::Function) = isreal(superdict)

has_derivative(set::WeightedDict) = has_derivative(superdict(set)) && has_derivative(weightfunction(set))
is_orthonormal(set::WeightedDict) = false
is_orthogonal(set::WeightedDict) = false
# We can not compute antiderivatives in general.
has_antiderivative(set::WeightedDict) = false

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
unsafe_eval_element(set::WeightedDict, idx, x) = _unsafe_eval_element(set, weightfunction(set), idx, x)
_unsafe_eval_element(set::WeightedDict1d, w, idx, x) = w(x) * unsafe_eval_element(superdict(set), idx, x)
_unsafe_eval_element(set::WeightedDict, w, idx, x) = w(x...) * unsafe_eval_element(superdict(set), idx, x)

# Evaluate the derivative of 1d weighted sets
unsafe_eval_element_derivative(set::WeightedDict1d, idx, x) =
    eval_derivative(weightfunction(set), x) * unsafe_eval_element(superdict(set), idx, x) +
    weightfunction(set)(x) * unsafe_eval_element_derivative(superdict(set), idx, x)

# Evaluate an expansion: same story
eval_expansion(set::WeightedDict, coefficients, x) = _eval_expansion(set, weightfunction(set), coefficients, x)
# temporary, to remove an ambiguity
eval_expansion(set::WeightedDict, coefficients, x::AbstractGrid) = _eval_expansion(set, weightfunction(set), coefficients, x)

_eval_expansion(set::WeightedDict1d, w, coefficients, x::Number) = w(x) * eval_expansion(superdict(set), coefficients, x)
_eval_expansion(set::WeightedDict, w, coefficients, x) = w(x...) * eval_expansion(superdict(set), coefficients, x)

_eval_expansion(set::WeightedDict, w, coefficients, grid::AbstractGrid) =
    eval_weight_on_grid(w, grid) .* eval_expansion(superdict(set), coefficients, grid)


# You can create an WeightedDict by multiplying a function with a set, using
# left multiplication.
# We support any Julia function:
(*)(f::Function, set::Dictionary) = WeightedDict(set, f)
# and our own functors:
(*)(f::AbstractFunction, set::Dictionary) = WeightedDict(set, f)

weightfun_scaling_operator(dgs::GridBasis1d, weightfunction) =
    DiagonalOperator(dgs, dgs, coefficienttype(dgs)[weightfunction(x) for x in grid(dgs)])

weightfun_scaling_operator(dgs::GridBasis, weightfunction) =
    DiagonalOperator(dgs, dgs, coefficienttype(dgs)[weightfunction(x...) for x in grid(dgs)])

transform_to_grid_post(src::WeightedDict, dest::GridBasis, grid; options...) =
    weightfun_scaling_operator(dest, weightfunction(src)) * transform_to_grid_post(superdict(src), dest, grid; options...)

transform_from_grid_pre(src::GridBasis, dest::WeightedDict, grid; options...) =
	transform_from_grid_pre(src, superdict(dest), grid; options...) * inv(weightfun_scaling_operator(src, weightfunction(dest)))


function derivative_dict(src::WeightedDict, order; options...)
    @assert order == 1

    s = superdict(src)
    f = weightfunction(src)
    f_prime = derivative(f)
    s_prime = derivative_dict(s, order)
    (f_prime * s) ⊕ (f * s_prime)
end

# Assume order = 1...
function differentiation_operator(s1::WeightedDict, s2::MultiDict, order; options...)
    @assert order == 1
    @assert s2 == derivative_dict(s1, order)

    I = IdentityOperator(s1, element(s2, 1))
    D = differentiation_operator(superdict(s1))
    DW = wrap_operator(s1, element(s2, 2), D)
    block_column_operator([I,DW])
end

function grid_evaluation_operator(set::WeightedDict, dgs::GridBasis, grid::AbstractGrid; options...)
    super_e = grid_evaluation_operator(superdict(set), dgs, grid; options...)
    D = weightfun_scaling_operator(dgs, weightfunction(set))
    D * wrap_operator(set, dgs, super_e)
end

function grid_evaluation_operator(set::WeightedDict, dgs::GridBasis, grid::AbstractSubGrid; options...)
    super_e = grid_evaluation_operator(superdict(set), dgs, grid; options...)
    D = weightfun_scaling_operator(dgs, weightfunction(set))
    D * wrap_operator(set, dgs, super_e)
end

symbol(s::WeightedDict) = "ω"
