
abstract type AbstractSamplingOperator <: AbstractOperator
end

dest_space(op::AbstractSamplingOperator) = Span(dest(op))

gridbasis(op::AbstractSamplingOperator) = dest(op)

grid(op::AbstractSamplingOperator) = grid(gridbasis(op))

apply(op::AbstractSamplingOperator, f::AbstractVector) = (@assert length(f)==size(op,2); f)


"""
A `GridSamplingOperator` is an operator that maps a function to its samples.

The operator optionally applies a scaling to the result. This may be, e.g.,
multiplication by a scalar number, or pointwise multiplication by a vector.
"""
struct GridSamplingOperator <: AbstractSamplingOperator
    src     ::  AbstractFunctionSpace
    dest    ::  GridBasis
    scaling
end

GridSamplingOperator(grid::AbstractGrid{S}, ::Type{T} = subeltype(S), args...; options...) where {S,T} =
    GridSamplingOperator(FunctionSpace{S,T}(), gridbasis(grid, T), args...; options...)

GridSamplingOperator(gridbasis::GridBasis{S,T}, args...; options...) where {S,T} =
	GridSamplingOperator(FunctionSpace{eltype(grid(gridbasis)),T}(), gridbasis, args...; options...)

GridSamplingOperator(src::AbstractFunctionSpace, dest::GridBasis; scaling = nothing) =
    GridSamplingOperator(src, dest, scaling)

dest(op::GridSamplingOperator) = op.dest

src_space(op::GridSamplingOperator) = op.src

apply!(result, op::GridSamplingOperator, f) = sample!(result, grid(op), f, op.scaling)

"Sample the function f on the given grid."
sample(g::AbstractGrid, f, T = float_type(eltype(g)), scaling=one(T)) = sample!(zeros(T, size(g)), g, f, scaling)

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

function sample!(result, grid, f, scaling::Number)
    for i in eachindex(grid)
		result[i] = scaling*call_function_with_vector(f, grid[i])
	end
	result
end

function sample!(result, grid, f, scaling::Nothing)
    for i in eachindex(grid)
		result[i] = call_function_with_vector(f, grid[i])
	end
	result
end

function sample!(result, grid, f, scaling::AbstractArray)
    for i in eachindex(grid)
		result[i] = scaling[i] * call_function_with_vector(f, grid[i])
	end
	result
end


apply(op::GridSamplingOperator, dict::Dictionary; options...) =
    _apply(op.scaling, op, dict; options...)
_apply(scaling::Nothing, op::GridSamplingOperator, dict; options...) =
    evaluation_operator(dict, grid(op); options...)
_apply(scaling::Number, op::GridSamplingOperator, dict; options...) =
    scaling*evaluation_operator(dict, grid(op); options...)
_apply(scaling::AbstractArray, op::GridSamplingOperator, dict; options...) =
    DiagonalOperator(dest(op), dest(op), scaling)*evaluation_operator(dict, grid(op); options...)

*(op::GridSamplingOperator, scalar::Number) = times_by(op.scaling, op, scalar)
*(scalar::Number, op::GridSamplingOperator) = times_by(op.scaling, op, scalar)

times_by(scaling::Nothing, op::GridSamplingOperator, scalar::Number) =
    GridSamplingOperator(op.src, op.dest, scalar)

times_by(scaling, op::GridSamplingOperator, scalar::Number) =
    GridSamplingOperator(op.src, op.dest, scalar*scaling)
