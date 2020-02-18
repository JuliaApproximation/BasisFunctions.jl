
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



weightfunction(dict::WeightedDict) = dict.weightfun

similardictionary(dict1::WeightedDict, dict2::Dictionary) = WeightedDict(dict2, weightfunction(dict1))

name(dict::WeightedDict) = "Weighted " * name(superdict(dict))

isreal(dict::WeightedDict) = _isreal(dict, superdict(dict), weightfunction(dict))
_isreal(dict::WeightedDict, superdict, fun::Function) = isreal(superdict)

hasderivative(dict::WeightedDict) = hasderivative(superdict(dict))
isorthonormal(dict::WeightedDict) = false
isorthogonal(dict::WeightedDict) = false
# We can not compute antiderivatives in general.
hasantiderivative(dict::WeightedDict) = false

hasmeasure(dict::WeightedDict) = false

# We have to distinguish between 1d and higher-dimensional grids, since we
# have to splat the arguments to the weightfunction
eval_weight_on_grid(w, grid::AbstractGrid1d) = [w(x) for x in grid]

function eval_weight_on_grid(w, grid::AbstractGrid)
    # Perhaps the implementation here could be simpler, but [w(x...) for x in grid]
    # does not seem to respect the size of the grid, only its length
    a = zeros(prectype(grid), size(grid))
    for i in eachindex(grid)
        a[i] = w(grid[i]...)
    end
    a
end

# Evaluating basis functions: we multiply by the function of the dict
unsafe_eval_element(dict::WeightedDict, idx, x) = _unsafe_eval_element(dict, weightfunction(dict), idx, x)
_unsafe_eval_element(dict::WeightedDict1d, w, idx, x) = w(x) * unsafe_eval_element(superdict(dict), idx, x)
_unsafe_eval_element(dict::WeightedDict, w, idx, x) = w(x...) * unsafe_eval_element(superdict(dict), idx, x)

# Evaluate the derivative of 1d weighted sets
unsafe_eval_element_derivative(dict::WeightedDict1d, idx, x) =
    derivative(weightfunction(dict), x) * unsafe_eval_element(superdict(dict), idx, x) +
    weightfunction(dict)(x) * unsafe_eval_element_derivative(superdict(dict), idx, x)

# Special case for log, since it is not precise at 0.
derivative(::typeof(log)) = x->1/x
derivative(::typeof(log), x::T) where T<:Number= convert(T,1)/x

# Evaluate an expansion: same story
eval_expansion(dict::WeightedDict, coefficients, x) = _eval_expansion(dict, weightfunction(dict), coefficients, x)
# temporary, to remove an ambiguity
eval_expansion(dict::WeightedDict, coefficients, x::AbstractGrid) = _eval_expansion(dict, weightfunction(dict), coefficients, x)

_eval_expansion(dict::WeightedDict1d, w, coefficients, x::Number) = w(x) * eval_expansion(superdict(dict), coefficients, x)
_eval_expansion(dict::WeightedDict, w, coefficients, x) = w(x...) * eval_expansion(superdict(dict), coefficients, x)

_eval_expansion(dict::WeightedDict, w, coefficients, grid::AbstractGrid) =
    eval_weight_on_grid(w, grid) .* eval_expansion(superdict(dict), coefficients, grid)


# You can create a WeightedDict by multiplying a function with a dict, using
# left multiplication:
(*)(f::Function, dict::Dictionary) = WeightedDict(dict, f)

weightfun_scaling_operator(::Type{T}, gb::GridBasis, weightfunction) where {T} =
	weightfun_scaling_operator(T, gb, weightfunction, grid(gb))

weightfun_scaling_operator(::Type{T}, gb, weightfunction, grid::AbstractGrid1d) where {T} =
    DiagonalOperator(gb, gb, map(T,map(weightfunction, grid)))

function weightfun_scaling_operator(::Type{T}, gb, weightfunction, grid::AbstractGrid) where {T}
	A = map(T,map(x->weightfunction(x...), grid))
    DiagonalOperator(gb, gb, A[:])
end

function weightfun_scaling_operator(::Type{T}, gb, weightfunction, grid::ProductGrid) where {T}
	A = map(T, map(x->weightfunction(x...), grid))
    DiagonalOperator(gb, gb, A[:])
end


transform_to_grid(T, src::WeightedDict, dest::GridBasis, grid; options...) =
    weightfun_scaling_operator(T, dest, weightfunction(src)) * wrap_operator(src, dest, transform_to_grid(T, superdict(src), dest, grid; options...))

transform_from_grid(T, src::GridBasis, dest::WeightedDict, grid; options...) =
	wrap_operator(src, dest, transform_from_grid(T, src, superdict(dest), grid; options...)) * inv(weightfun_scaling_operator(T, src, weightfunction(dest)))



function derivative_dict(src::WeightedDict, order; options...)
	if order == 0
		src
	else
    	@assert order == 1
	    s = superdict(src)
	    f = weightfunction(src)
	    f_prime = derivative(f)
	    s_prime = derivative_dict(s, order)
	    (f_prime * s) ⊕ (f * s_prime)
	end
end

# Assume order = 1...
function differentiation(::Type{T}, src::WeightedDict, dest::MultiDict, order; options...) where {T}
    @assert order == 1
    @assert dest == derivative_dict(src, order)

    I = IdentityOperator{T}(src, element(dest, 1))
    D = differentiation(T, superdict(src), order)
    DW = wrap_operator(src, element(dest, 2), D)
    block_column_operator([I,DW])
end

function evaluation(::Type{T}, dict::WeightedDict, gb::GridBasis, grid::AbstractGrid; options...) where {T}
    A = evaluation(T, superdict(dict), gb, grid; options...)
    D = weightfun_scaling_operator(T, gb, weightfunction(dict))
    D * wrap_operator(dict, gb, A)
end

## Printing

string(dict::WeightedDict) = name(dict) * ", weighted by " * string(weightfunction(dict))

modifiersymbol(dict::WeightedDict) = PrettyPrintSymbol{:ω}(weightfunction(dict))
name(s::PrettyPrintSymbol{:ω}) = "Weight function: " * string(s.object)
