
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

==(dict1::WeightedDict,dict2::WeightedDict) =
    superdict(dict1)==superdict(dict2) && weightfunction(dict1)==weightfunction(dict2)

weightfunction(dict::WeightedDict) = dict.weightfun

similardictionary(dict1::WeightedDict, dict2::Dictionary) = WeightedDict(dict2, weightfunction(dict1))

isreal(dict::WeightedDict) = _isreal(dict, superdict(dict), weightfunction(dict))
_isreal(dict::WeightedDict, superdict, fun::Function) = isreal(superdict)

hasderivative(dict::WeightedDict) = hasderivative(superdict(dict))
isorthonormal(dict::WeightedDict) = false
isorthogonal(dict::WeightedDict) = false
# We can not compute antiderivatives in general.
hasantiderivative(dict::WeightedDict) = false

hasmeasure(dict::WeightedDict) = false

mapped_dict(d::WeightedDict, map) = MappedDict(d, map)

# We have to distinguish between 1d and higher-dimensional grids, since we
# have to splat the arguments to the weightfunction
eval_weight_on_grid(w, grid::GridArrays.Grid1dLike) = [w(x) for x in grid]

function eval_weight_on_grid(w, grid)
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
function unsafe_eval_element_derivative(dict::WeightedDict1d, idx, x, order)
	@assert order == 1
    diff(weightfunction(dict), x) * unsafe_eval_element(superdict(dict), idx, x) +
		weightfunction(dict)(x) * unsafe_eval_element_derivative(superdict(dict), idx, x, order)
end

# Special case for log, since it is not precise at 0.
diff(::typeof(log)) = x->1/x
diff(::typeof(log), x::T) where T<:Number= one(T)/x
diff(::typeof(cos)) = x -> -sin(x)
diff(::typeof(cos), x) = -sin(x)

# Evaluate an expansion: same story
eval_expansion(dict::WeightedDict, coefficients, x) = _eval_expansion(dict, weightfunction(dict), coefficients, x)
_eval_expansion(dict::WeightedDict1d, w, coefficients, x::Number) = w(x) * eval_expansion(superdict(dict), coefficients, x)
_eval_expansion(dict::WeightedDict, w, coefficients, x) = w(x...) * eval_expansion(superdict(dict), coefficients, x)


# You can create a WeightedDict by multiplying a function with a dict, using
# left multiplication:
(*)(f::Function, dict::Dictionary) = WeightedDict(dict, f)

weightfun_scaling_operator(::Type{T}, gb::GridBasis, weightfunction) where {T} =
	weightfun_scaling_operator(T, gb, weightfunction, grid(gb))

weightfun_scaling_operator(::Type{T}, gb, weightfunction, grid::GridArrays.Grid1dLike) where {T} =
    DiagonalOperator(gb, gb, map(T,map(weightfunction, grid)))

function weightfun_scaling_operator(::Type{T}, gb, weightfunction, grid) where {T}
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
	elseif order == 1
	    s = superdict(src)
	    f = weightfunction(src)
	    f_prime = diff(f)
	    s_prime = derivative_dict(s, order)
	    (f_prime * s) ⊕ (f * s_prime)
    else
        @assert order>1
        s = superdict(src)
        f = weightfunction(src)
        dfs = Any[]
        push!(dfs,f)
        for k in 1:order
            push!(dfs,diff(dfs[k]))
        end
        ⊕([dfs[k+1]*(derivative_dict(s, order-k))  for k in order:-1:0]...)
    end
end

# Assume order = 1...
function differentiation(::Type{T}, src::WeightedDict, dest::MultiDict, order; options...) where {T}
    @assert size(dest) == size(derivative_dict(src, order))
    if order==0
        return IdentityOperator{T}(src, component(dest, 1))
    end
    if order == 1
        I = IdentityOperator{T}(src, component(dest, 1))
        D = differentiation(T, superdict(src), order)
        DW = wrap_operator(src, component(dest, 2), D)
        block_column_operator([I,DW])
    elseif order > 1
        Ds = [wrap_operator(src,e,binomial(order,k)*differentiation(T, superdict(src), k))
            for (e,k) in zip(components(dest),0:order)]
        block_column_operator(Ds)
    else
        error("differentiation of order $order not implemented")
    end
end

function evaluation(::Type{T}, dict::WeightedDict, gb::GridBasis, grid; options...) where {T}
    A = evaluation(T, superdict(dict), gb, grid; options...)
    D = weightfun_scaling_operator(T, gb, weightfunction(dict))
    D * wrap_operator(dict, gb, A)
end

## Printing

Display.object_parentheses(d::WeightedDict) = true
Display.stencil_parentheses(d::WeightedDict) = true
Display.displaystencil(d::WeightedDict) = _stencil(d, superdict(d), weightfunction(d))
_stencil(d::WeightedDict, dict, weight) = [weight, " * ", dict]
