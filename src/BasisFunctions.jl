module BasisFunctions


import Base: length, size, start, next, done, ind2sub, sub2ind, eachindex, checkbounds, range

import Base: getindex, getindex!, setindex!, eltype

import Base: isreal, iseven, isodd

import Base: transpose

# from grid.jl
export AbstractGrid, AbstractGrid1d, AbstractEquispacedGrid, EquispacedGrid, PeriodicEquispacedGrid, TensorProductGrid, AbstractIntervalGrid

# from basis_types.jl
export AbstractFunctionSet, AbstractFrame, AbstractBasis, AbstractBasis1d, SetFunction
export numtype, natural_grid, left, right, support, call, call!

# from tensorproductbasis.jl
export TensorProductBasis, tensorproduct

#from expansions.jl
export SetExpansion, TensorProductExpansion

# from operator.jl
export AbstractOperator, CompositeOperator, OperatorTranspose, transpose, operator, src, dest

# from timedomain.jl
export TimeDomain

# from extensions.jl
export ZeroPadding, Restriction

# from fourierbasis.jl
export FourierBasis, FourierBasisEven, FourierBasisOdd, FourierBasisNd, FastFourierTransform, InverseFastFourierTransform

# from plots.jl
export plot


using Base.Cartesian


typealias True Val{true}
typealias False Val{false}


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


