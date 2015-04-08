module BasisFunctions


import Base: length, size, start, next, done, ind2sub, sub2ind, eachindex, checkbounds, range

import Base: getindex, getindex!, setindex!, eltype

import Base: isreal, iseven, isodd

# from basis_types.jl
export AbstractFunctionSet, AbstractFrame, AbstractBasis, AbstractBasis1d, SetFunction

#from expansions.jl
export SetExpansion, TensorProductExpansion

# from tensorproduct.jl
export TensorProductBasis, tensorproduct

# from grid.jl
export AbstractGrid, AbstractGrid1d, AbstractEquispacedGrid, EquispacedGrid, PeriodicEquispacedGrid, TensorProductGrid

export left, right, support

export call, call!

export natural_grid

export numtype

#export transform, transform!, itransform, itransform!, transform_matrix, transform_matrix!, interpolation_matrix, interpolation_matrix!

#export differentiate, differentiate!

# from extensions.jl
export ZeroPadding, Restriction, TimeDomain, AbstractOperator, EquispacedSubGrid, TensorProductGrid, AbstractIntervalGrid

# from fourierbasis.jl
export FourierBasis, FourierBasisEven, FourierBasisOdd, FourierBasisNd, FastFourierTransform, InverseFastFourierTransform

# from plots.jl
export plot

using Base.Cartesian


typealias True Val{true}
typealias False Val{false}


complexify{T <: Real}(::Type{T}) = Complex{T}

complexify{T <: Real}(::Type{Complex{T}}) = Complex{T}


include("grid.jl")

include("basis_types.jl")

include("tensorproductbasis.jl")

include("expansions.jl")

include("operator.jl")

include("discrete_transforms.jl")

include("extensions.jl")

include("differentiation.jl")

include("timedomain.jl")

include("fourierbasis.jl")

include("splinebasis.jl")

include("waveletbasis.jl")

include("polynomialbasis.jl")

include("plots.jl")

end # module


