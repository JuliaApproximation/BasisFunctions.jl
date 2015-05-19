# discrete_transforms.jl

abstract AbstractDiscreteTransform{SRC,DEST} <: AbstractOperator{SRC,DEST}


immutable DenseInterpolation{ELT,SRC,DEST} <: AbstractDiscreteTransform{SRC,DEST}
	src		::	SRC
	dest	::	DEST
    matrix  ::  Array{ELT,2}

    DenseInterpolation(src, dest) = new(src, dest, interpolation_matrix(dest, grid(src), ELT))
end

DenseInterpolation{SRC,DEST}(src::SRC, dest::DEST) = DenseInterpolation{mixed_eltype(eltype(src(op)), eltype(dest(op))),SRC,DEST}(src, dest)

matrix(op::DenseInterpolation) = op.matrix

apply!(op::DenseInterpolation, coef_dest, coef_src) = (coef_dest[:] = op.matrix * coef_src)


function interpolation_matrix(b::AbstractBasis, g::AbstractGrid, T = eltype(b))
    A = Array(T, length(g), length(b))
    interpolation_matrix!(b, g, A)
    A
end

function interpolation_matrix!{N,T}(b::AbstractBasis{N,T}, g::AbstractGrid{N,T}, a::Array)
	n = size(a,1)
	m = size(a,2)
	@assert n == length(g)
	@assert m == length(b)

    x_i = Array(T,N)
	for j = 1:m
		for i = 1:n
			getindex!(g, x_i, i)
			a[i,j] = call(b, j, x_i...)
		end
	end
end





