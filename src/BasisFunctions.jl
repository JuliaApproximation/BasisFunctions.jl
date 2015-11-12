module BasisFunctions

# We may import ApproxFun to use its implementation of FFT for BigFloat's
# import ApproxFun

using ArrayViews

import Base: +, *, /, ==, |, &, -, \

import Base: length, size, start, next, done, ind2sub, sub2ind, eachindex, range

import Base: getindex, setindex!, eltype

import Base: isreal, iseven, isodd

import Base: ctranspose, transpose

import Base: show, showcompact

# from grid.jl
export AbstractGrid, AbstractGrid1d, AbstractEquispacedGrid, EquispacedGrid, PeriodicEquispacedGrid,
        TensorProductGrid, AbstractIntervalGrid, eachelement, stepsize
export dim, left,right, range

# from functionset.jl
export AbstractFunctionSet, AbstractFrame, AbstractBasis, AbstractBasis1d
export numtype, grid, left, right, support, call, call!
export name
export transform_operator, differentiation_operator, approximation_operator

# from setfunction.jl
export SetFunction, index

# from tensorproductset.jl
export TensorProductSet, tensorproduct, ⊗

# from mappedsets.jl
export map, imap, map_linear, imap_linear

#from expansions.jl
export SetExpansion, TensorProductExpansion, coefficients, set

# from operator.jl
export AbstractOperator, CompositeOperator, OperatorTranspose, ctranspose, operator, src, dest,
    IdentityOperator, ScalingOperator, DenseOperator, MatrixOperator, apply!

# from generic_operator.jl
export extension_operator, restriction_operator, interpolation_operator, 
    approximation_operator, transform_operator, differentiation_operator,
    evaluation_operator,
    Extension, Restriction, Differentiation, TransformOperator

# from tensorproductoperator.jl
export TensorProductOperator

# from functional.jl
export AbstractFunctional

# from discretegridspace.jl
export DiscreteGridSpace, DiscreteGridSpace1d, DiscreteGridSpaceNd, left, right

# from approximation.jl
export approximate


# from fourierbasis.jl
export FourierBasis, FourierBasisEven, FourierBasisOdd, FourierBasisNd, 
    FastFourierTransform, InverseFastFourierTransform,
    FastFourierTransformFFTW, InverseFastFourierTransformFFTW,
    frequency2idx, idx2frequency

# from chebyshevbasis.jl
export ChebyshevBasis, 
    FastChebyshevTransform, InverseFastChebyshevTransform,
    FastChebyshevTransformFFTW, InverseFastChebyshevTransformFFTW

# from plots.jl
export plot

# from bf_polynomials.jl and friends
export LegendreBasis, JacobiBasis, LaguerreBasis, HermiteBasis, MonomialBasis


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

include("tensorproductset.jl")

include("mappedsets.jl")

include("expansions.jl")

include("euclidean.jl")

include("operator.jl")

include("tensorproductoperator.jl")

include("functional.jl")

include("generic_operators.jl")

include("discretegridspace.jl")

include("approximation.jl")

include("bf_fourier.jl")

include("bf_splines.jl")

include("bf_wavelets.jl")

include("bf_polynomials.jl")

include("poly_chebyshev.jl")

include("poly_legendre.jl")

include("poly_jacobi.jl")

include("poly_laguerre.jl")

include("poly_hermite.jl")

include("plots.jl")

end # module


