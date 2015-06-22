module BasisFunctions

# We may import ApproxFun to use its implementation of FFT for BigFloat's
# import ApproxFun

import Base: length, size, start, next, done, ind2sub, sub2ind, eachindex, range

import Base: getindex, setindex!, eltype

import Base: isreal, iseven, isodd

import Base: ctranspose, transpose

import Base: show, showcompact

# from grid.jl
export AbstractGrid, AbstractGrid1d, AbstractEquispacedGrid, EquispacedGrid, PeriodicEquispacedGrid,
        TensorProductGrid, AbstractIntervalGrid, eachelement
export dim, left,right

# from functionset.jl
export AbstractFunctionSet, AbstractFrame, AbstractBasis, AbstractBasis1d
export numtype, grid, left, right, support, call, call!
export name
export transform_operator, differentiation_operator

# from setfunction.jl
export SetFunction

# from tensorproductbasis.jl
export TensorProductSet, TensorProductBasis, tensorproduct, idx2frequency, frequency2idx

#from expansions.jl
export SetExpansion, TensorProductExpansion, coefficients, set

# from operator.jl
export AbstractOperator, CompositeOperator, OperatorTranspose, ctranspose, operator, src, dest,
    IdentityOperator, ScalingOperator, DenseOperator, AffineMap

# from timedomain.jl
export TimeDomain, TimeDomain1d, TimeDomainNd

# from extensions.jl
export Extension, Restriction

# from fourierbasis.jl
export FourierBasis, FourierBasisEven, FourierBasisOdd, FourierBasisNd, 
    FastFourierTransform, InverseFastFourierTransform,
    FastFourierTransformFFTW, InverseFastFourierTransformFFTW

# from plots.jl
export plot


using Base.Cartesian


# Convenience definitions for the implementation of traits
typealias True Val{true}
typealias False Val{false}

(&)(::Type{True},  ::Type{True} ) = True
(&)(::Type{False}, ::Type{True} ) = False
(&)(::Type{True},  ::Type{False}) = False
(&)(::Type{False}, ::Type{False}) = False

(&)(::True,  ::True ) = True()
(&)(::False, ::True ) = False()
(&)(::True,  ::False) = False()
(&)(::False, ::False) = False()

(|)(::Type{True},  ::Type{True} ) = True
(|)(::Type{False}, ::Type{True} ) = True
(|)(::Type{True},  ::Type{False}) = True
(|)(::Type{False}, ::Type{False}) = False

(|)(::True,  ::True ) = True()
(|)(::False, ::True ) = True()
(|)(::True,  ::False) = True()
(|)(::False, ::False) = False()


include("grid.jl")

include("functionset.jl")

include("setfunction.jl")

include("tensorproductbasis.jl")

include("expansions.jl")

include("euclidean.jl")

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


