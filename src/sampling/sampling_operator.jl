# sampling_operator.jl

"""
A `GridSamplingOperator` is an operator that maps a function to its samples.
"""
struct GridSamplingOperator <: GenericOperator
    src     ::  AbstractFunctionSpace
    dest    ::  DiscreteGridSpace

	# An inner constructor to enforce that the spaces match
	GridSamplingOperator(src::AbstractFunctionSpace{S,T}, dest::DiscreteGridSpace{A,S,T}) where {A,S,T} =
		new(src, dest)
end

GridSamplingOperator(src::FunctionSpace{S,T}, grid::AbstractGrid{S}) where {S,T} =
	GridSamplingOperator(src, gridspace(grid, T))

GridSamplingOperator(gridspace::DiscreteGridSpace{A,S,T}) where {A,S,T} =
	GridSamplingOperator(FunctionSpace{S,T}(), gridspace)

src(op::GridSamplingOperator) = op.src
dest(op::GridSamplingOperator) = op.dest

gridspace(op::GridSamplingOperator) = dest(op)

grid(op::GridSamplingOperator) = grid(gridspace(op))

apply(op::GridSamplingOperator, f) = sample(grid(op), f, coeftype(gridspace(op)))
apply!(result, op::GridSamplingOperator, f) = sample!(result, grid(op), f)

"Sample the function f on the given grid."
sample(g::AbstractGrid, f, T = float_type(eltype(g))) = sample!(zeros(T, size(g)), g, f)

(*)(op::GridSamplingOperator, f) = apply(op, f)

broadcast(f::Function, grid::AbstractGrid) = sample(grid, f)


# We don't want to assume that f can be called with a vector argument.
# In order to avoid the overhead of splatting, we capture a number of special cases
call_function_with_vector(f, x::Number) = f(x)
call_function_with_vector(f, x::SVector{1}) = f(x[1])
call_function_with_vector(f, x::SVector{2}) = f(x[1], x[2])
call_function_with_vector(f, x::SVector{3}) = f(x[1], x[2], x[3])
call_function_with_vector(f, x::SVector{4}) = f(x[1], x[2], x[3], x[4])
call_function_with_vector(f, x::SVector{N}) where {N} = f(x...)
call_function_with_vector(f, x::AbstractVector) = f(x...)

function sample!(result, g::AbstractGrid, f)
	for i in eachindex(g)
		result[i] = call_function_with_vector(f, g[i])
	end
	result
end

(*)(S::GridSamplingOperator, D::Dictionary) = apply(S, D)

apply(S::GridSamplingOperator, D::Dictionary) = evaluation_operator(Span(D, coeftype(gridspace(S))), grid(S))
