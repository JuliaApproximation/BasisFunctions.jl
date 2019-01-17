
abstract type SamplingOperator <: AbstractOperator
end

dest_space(op::SamplingOperator) = Span(dest(op))

function apply(op::SamplingOperator, f::AbstractVector)
	@warn "This is a funny function to use"
	@assert length(f) == size(op,2)
	f
end


"""
A `GridSampling` is an operator that maps a function to its samples
in a grid.
"""
struct GridSampling <: SamplingOperator
    src     ::  FunctionSpace
    dest    ::  GridBasis
end

GridSampling(grid::AbstractGrid{S}, ::Type{T} = subeltype(S)) where {S,T} =
    GridSampling(GenericFunctionSpace{S,T}(), GridBasis{T}(grid))

GridSampling(gridbasis::GridBasis{S,T}) where {S,T} =
	GridSampling(GenericFunctionSpace{eltype(grid(gridbasis)),T}(), gridbasis)

grid(op::GridSampling) = grid(dest(op))

(op::GridSampling)(f) = apply(op, f)

dest(op::GridSampling) = op.dest

src_space(op::GridSampling) = op.src

apply!(result, op::GridSampling, f) = sample!(result, grid(op), f)

"Sample the function f on the given grid."
sample(g::AbstractGrid, f, T = float_type(eltype(g))) = sample!(zeros(T, size(g)), g, f)

broadcast(f::Function, grid::AbstractGrid) = sample(grid, f)


function tensorproduct(op1::GridSampling, op2::GridSampling)
	T = promote_type(coefficienttype(dest(op1)),coefficienttype(dest(op2)))
	grid1 = grid(op1)
	grid2 = grid(op2)
	GridSampling(grid1 × grid2, T)
end

function tensorproduct(op1::GridSampling, op2::AbstractOperator)
	@assert dest(op2) isa GridBasis
	@assert numelements(op2) == 2
	tensorproduct(IdentityOperator(dest(op1)), element(op2,2)) * tensorproduct(op1, element(op2,1))
end

function tensorproduct(op1::AbstractOperator, op2::AbstractOperator)
	@assert dest(op1) isa GridBasis
	@assert dest(op2) isa GridBasis
	@assert numelements(op1) == 2
	@assert numelements(op2) == 2
	T = promote_type(coefficienttype(dest(op1)),coefficienttype(dest(op2)))
	grid1 = grid(dest(op1))
	grid2 = grid(dest(op2))
	g = grid1 × grid2
	tensorproduct(element(op1,2),element(op2,2)) * tensorproduct(element(op1,1),element(op2,1))
end

# We don't want to assume that f can be called with a vector argument.
# In order to avoid the overhead of splatting, we capture a number of special cases
call_function_with_vector(f, x::Number) = f(x)
call_function_with_vector(f, x::SVector{1}) = f(x[1])
call_function_with_vector(f, x::SVector{2}) = f(x[1], x[2])
call_function_with_vector(f, x::SVector{3}) = f(x[1], x[2], x[3])
call_function_with_vector(f, x::SVector{4}) = f(x[1], x[2], x[3], x[4])
call_function_with_vector(f, x::SVector{N}) where {N} = f(x...)
call_function_with_vector(f, x::AbstractVector) = f(x...)

function sample!(result, grid, f)
    for i in eachindex(grid)
		result[i] = call_function_with_vector(f, grid[i])
	end
	result
end

apply(op::GridSampling, dict::Dictionary) = evaluation_operator(dict, grid(op))


strings(op::GridSampling) = ("Sampling in a discrete grid with coefficient type $(coefficienttype(dest(op)))",strings(grid(op)))


"""
A `ProjectionSampling` is an operator that maps a function to its inner products
with a projection basis.
"""
struct ProjectionSampling <: SamplingOperator
    dict	::  Dictionary
	measure	::	Measure
	space	::	FunctionSpace
end

space(dict::Dictionary) = space(measure(dict))

ProjectionSampling(dict::Dictionary) = ProjectionSampling(dict, measure(dict))

ProjectionSampling(dict::Dictionary, measure::Measure) = ProjectionSampling(dict, measure, space(measure))

ProjectionSampling(dict::Dictionary, space::FunctionSpace) = ProjectionSampling(dict, measure(space), space)

dictionary(op::ProjectionSampling) = op.dict

dest(op::ProjectionSampling) = dictionary(op)

src_space(op::ProjectionSampling) = op.space

measure(op::ProjectionSampling) = op.measure

apply(op::ProjectionSampling, dict::Dictionary) =
	apply_projection(op, dict, dictionary(op), measure(dict), measure(op))

apply_projection(op, dict::D, projectiondict::D, measure::M, projectionmeasure::M) where {D <: Dictionary,M<:Measure} =
	gramoperator(dict)

apply_projection(op, dict, projectiondict, measure, projectionmeasure) = mixedgramoperator(projectiondict, dict, projectionmeasure)

apply!(result, op::ProjectionSampling, f) = project!(result, f, dictionary(op), measure(op))

function project!(result, f, dict::Dictionary, measure::Measure)
    for i in eachindex(result)
		result[i] = innerproduct(dict[i], f, measure)
	end
	result
end

strings(op::ProjectionSampling) = ("Projection operator onto a dictionary", strings(measure(op)), strings(dictionary(op)))
