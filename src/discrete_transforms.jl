# discrete_transforms.jl


# A function set can implement the apply! method of a suitable TransformOperator for any known transform.
# Example: a discrete transform from a set of samples on a grid to a set of expansion coefficients.
immutable TransformOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

# The default transform from src to dest is a TransformOperator. This may be overridden for specific source and destinations.
transform_operator(src, dest) = TransformOperator(src, dest)

# Convenience functions: automatically convert a grid to a DiscreteGridSpace
transform_operator(src::AbstractGrid, dest::AbstractFunctionSet) = transform_operator(DiscreteGridSpace(src), dest)
transform_operator(src::AbstractFunctionSet, dest::AbstractGrid) = transform_operator(src, DiscreteGridSpace(dest))

ctranspose(op::TransformOperator) = TransformOperator(dest(op), src(op))

## The transform is invariant under a linear map.
#apply!(op::TransformOperator, src::LinearMappedSet, dest::LinearMappedSet, coef_dest, coef_src) =
#    apply!(op, set(src), set(dest), coef_dest, coef_src)



# Compute the interpolation matrix of the given basis on the given grid.
function interpolation_matrix(b::AbstractBasis, g::AbstractGrid, T = eltype(b))
    a = Array(T, length(g), length(b))
    interpolation_matrix!(b, g, a)
    a
end

function interpolation_matrix!{N,T}(b::AbstractBasis{N,T}, g::AbstractGrid{N,T}, a::AbstractArray)
	n = size(a,1)
	m = size(a,2)
	@assert n == length(g)
	@assert m == length(b)

    x_i = Array(T,N)
	for j = 1:m
		for i = 1:n
			a[i,j] = call(b, j, x_i...)
		end
	end
end


function interpolation_matrix!{T}(b::AbstractBasis1d{T}, g::AbstractGrid1d{T}, a::AbstractArray)
    n = size(a,1)
    m = size(a,2)
    @assert n == length(g)
    @assert m == length(b)

    for j = 1:m
        for i = 1:n
            a[i,j] = call(b, j, g[i])
        end
    end
end


interpolation_operator(b::AbstractBasis) = SolverOperator(grid(b), b, qrfact(interpolation_matrix(b, grid(b))))

# Evaluation works for any set that has a grid(set) associated with it.
evaluation_operator(s::AbstractFunctionSet) = MatrixOperator(s, grid(s), interpolation_matrix(s, grid(s)))


# The default approximation for a basis is interpolation
approximation_operator(b::AbstractBasis) = interpolation_operator(b)




