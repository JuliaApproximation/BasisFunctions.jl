# timedomain.jl

immutable TimeDomain{G <: AbstractGrid,ELT,N,T} <: AbstractBasis{N,T}
	grid		::	G

	TimeDomain(grid::AbstractGrid{N,T}) = new(grid)
end

typealias TimeDomain1d{G,ELT,T} TimeDomain{G,ELT,1,T}

typealias TimeDomainNd{G,ELT,N,T} TensorProductBasis{TimeDomain1d{G,ELT,T},G,N,T}


TimeDomain{N,T}(grid::AbstractGrid{N,T}) = TimeDomain{typeof(grid),T,N,T}(grid)

TimeDomain{N,T,ELT <: Number}(grid::AbstractGrid{N,T}, ::Type{ELT}) = TimeDomain{typeof(grid),ELT,N,T}(grid)

eltype{G,ELT}(::TimeDomain{G,ELT}) = ELT
eltype{G,ELT}(::Type{TimeDomain{G,ELT}}) = ELT
eltype{B <: TimeDomain}(::Type{B}) = eltype(super(B))

length(b::TimeDomain) = length(b.grid)

size(b::TimeDomain) = size(b.grid)

grid(b::TimeDomain) = b.grid

