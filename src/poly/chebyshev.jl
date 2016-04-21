# chebyshev.jl


############################################
# Chebyshev polynomials of the first kind
############################################


"""
A basis of Chebyshev polynomials of the first kind on the interval [-1,1].
"""
immutable ChebyshevBasis{T} <: OPS{T}
    n			::	Int

    ChebyshevBasis(n) = new(n)
end

typealias ChebyshevBasisFirstKind{T} ChebyshevBasis{T}


name(b::ChebyshevBasis) = "Chebyshev series (first kind)"


ChebyshevBasis{T}(n, ::Type{T} = Float64) = ChebyshevBasis{T}(n)

ChebyshevBasis{T}(n, a, b, ::Type{T} = promote_type(typeof(a),typeof(b))) = rescale( ChebyshevBasis(n,floatify(T)), a, b)

instantiate{T}(::Type{ChebyshevBasis}, n, ::Type{T}) = ChebyshevBasis{T}(n)

promote_eltype{T,S}(b::ChebyshevBasis{T}, ::Type{S}) = ChebyshevBasis{promote_type(T,S)}(b.n)

resize(b::ChebyshevBasis, n) = ChebyshevBasis(n, eltype(b))


has_grid(b::ChebyshevBasis) = true
has_derivative(b::ChebyshevBasis) = true
has_antiderivative(b::ChebyshevBasis) = true
has_transform{G <: ChebyshevIIGrid}(b::ChebyshevBasis, d::DiscreteGridSpace{G}) = true


left(b::ChebyshevBasis) = -1
left(b::ChebyshevBasis, idx) = left(b)

right(b::ChebyshevBasis) = 1
right(b::ChebyshevBasis, idx) = right(b)

grid{T}(b::ChebyshevBasis{T}) = ChebyshevIIGrid(b.n,numtype(b))

# The weight function
weight{T}(b::ChebyshevBasis{T}, x) = 1/sqrt(1-T(x)^2)

# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_α(b::ChebyshevBasis) = -1//2
jacobi_β(b::ChebyshevBasis) = -1//2



# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevBasis, n::Int) = n==0 ? 1 : 2

rec_Bn(b::ChebyshevBasis, n::Int) = 0

rec_Cn(b::ChebyshevBasis, n::Int) = 1


# Map the point x in [a,b] to the corresponding point in [-1,1]

call_element(b::ChebyshevBasis, idx::Int, x) = cos((idx-1)*acos(x))

function apply!{T}(op::Differentiation, dest::ChebyshevBasis{T}, src::ChebyshevBasis{T}, result, coef)
    #	@assert period(dest)==period(src)
    n = length(src)
    tempc = coef[:]
    tempr = coef[:]
    for o = 1:order(op)
        tempr = zeros(T,n)
        # 'even' summation
        s = 0
        for i=(n-1):-2:2
            s = s+2*i*tempc[i+1]
            tempr[i] = s
        end
        # 'uneven' summation
        s = 0
        for i=(n-2):-2:2
            s = s+2*i*tempc[i+1]
            tempr[i] = s
        end
        # first row
        s = 0
        for i=2:2:n
            s = s+(i-1)*tempc[i]
        end
        tempr[1]=s
        tempc = tempr
    end
    result[1:n-order(op)] = tempr[1:n-order(op)]
    result
end

function apply!{T}(op::AntiDifferentiation, dest::ChebyshevBasis{T}, src::ChebyshevBasis{T}, result, coef)
    #	@assert period(dest)==period(src)
    tempc = zeros(T,length(result))
    tempc[1:length(src)] = coef[1:length(src)]
    tempr = zeros(T,length(result))
    tempr[1:length(src)] = coef[1:length(src)]
    for o = 1:order(op)
        n = length(src)+o
        tempr = zeros(T,n)
        tempr[2]+=tempc[1]
        tempr[3]=tempc[2]/4
        tempr[1]+=tempc[2]/4
        for i = 3:n-1
            tempr[i-1]-=tempc[i]/(2*(i-2))
            tempr[i+1]+=tempc[i]/(2*(i))
            tempr[1]+=real(-1im^i)*tempc[i]*(2*i-2)/(2*i*(i-2))
        end
        tempc = tempr
    end
    result[:]=tempr[:]
    result
end


# TODO: restrict the grid of grid space here
transform_operator(src::DiscreteGridSpace, dest::ChebyshevBasis; options...) =
	_forward_chebyshev_operator(src, dest, eltype(src,dest); options...)

_forward_chebyshev_operator(src, dest, ::Union{Type{Float64},Type{Complex{Float64}}}; options...) =
	FastChebyshevTransformFFTW(src, dest; options...)

_forward_chebyshev_operator{T <: Number}(src, dest, ::Type{T}; options...) =
	FastChebyshevTransform(src, dest)



transform_operator(src::ChebyshevBasis, dest::DiscreteGridSpace; options...) =
	_backward_chebyshev_operator(src, dest, eltype(src,dest); options...)

_backward_chebyshev_operator(src, dest, ::Type{Float64}; options...) =
	InverseFastChebyshevTransformFFTW(src, dest; options...)

_backward_chebyshev_operator(src, dest, ::Type{Complex{Float64}}; options...) =
	InverseFastChebyshevTransformFFTW(src, dest; options...)

_backward_chebyshev_operator{T <: Number}(src, dest, ::Type{T}; options...) =
	InverseFastChebyshevTransform(src, dest)


# Catch 2D and 3D fft's automatically
transform_operator_tensor(src, dest,
    src_set1::DiscreteGridSpace, src_set2::DiscreteGridSpace,
    dest_set1::ChebyshevBasis, dest_set2::ChebyshevBasis; options...) =
        _forward_chebyshev_operator(src, dest, eltype(src, dest); options...)

transform_operator_tensor(src, dest,
    src_set1::ChebyshevBasis, src_set2::ChebyshevBasis,
    dest_set1::DiscreteGridSpace, dest_set2::DiscreteGridSpace; options...) =
        _backward_chebyshev_operator(src, dest, eltype(src, dest); options...)

transform_operator_tensor(src, dest,
    src_set1::DiscreteGridSpace, src_set2::DiscreteGridSpace, src_set3::DiscreteGridSpace,
    dest_set1::ChebyshevBasis, dest_set2::ChebyshevBasis, dest_set3::ChebyshevBasis; options...) =
        _forward_chebyshev_operator(src, dest, eltype(src, dest); options...)

transform_operator_tensor(src, dest,
    src_set1::ChebyshevBasis, src_set2::ChebyshevBasis, src_set3::ChebyshevBasis,
    dest_set1::DiscreteGridSpace, dest_set2::DiscreteGridSpace, dest_set3::DiscreteGridSpace; options...) =
        _backward_chebyshev_operator(src, dest, eltype(src, dest); options...)


function transform_normalization_operator(src::ChebyshevBasis; options...)
    ELT = eltype(src)
    scaling = ScalingOperator(src, 1/sqrt(ELT(length(src)/2)))
    coefscaling = CoefficientScalingOperator(src, 1, 1/sqrt(ELT(2)))
    flip = UnevenSignFlipOperator(src)
	scaling * coefscaling * flip
end


# immutable ChebyshevNormalization{ELT} <: AbstractOperator{ELT}
#     src     :: FunctionSet
# end
#
# ChebyshevNormalization(src::FunctionSet) = ChebyshevNormalization{eltype(src)}(src)
#
# dest(op::ChebyshevNormalization) = src(op)
#
#
# function apply_inplace!{ELT}(op::ChebyshevNormalization{ELT}, coef_srcdest)
# 	L = length(op.src)
# 	T = numtype(src)
# 	s = 1/sqrt(ELT(length(coef_srcdest))/2)
#     coef_srcdest[1] /= sqrt(T(2))
# end
#


############################################
# Chebyshev polynomials of the second kind
############################################

"A basis of Chebyshev polynomials of the second kind (on the interval [-1,1])."
immutable ChebyshevBasisSecondKind{T} <: OPS{T}
    n			::	Int
end

ChebyshevBasisSecondKind{T}(n, ::Type{T} = Float64) = ChebyshevBasisSecondKind{T}(n)

instantiate{T}(::Type{ChebyshevBasisSecondKind}, n, ::Type{T}) = ChebyshevBasisSecondKind{T}(n)

promote_eltype{T,S}(b::ChebyshevBasisSecondKind{T}, ::Type{S}) = ChebyshevBasisSecondKind{promote_type(T,S)}(b.n)

resize(b::ChebyshevBasisSecondKind, n) = ChebyshevBasisSecondKind(n, eltype(b))

name(b::ChebyshevBasisSecondKind) = "Chebyshev series (second kind)"


left{T}(b::ChebyshevBasisSecondKind{T}) = -one(T)
left{T}(b::ChebyshevBasisSecondKind{T}, idx) = left(b)

right{T}(b::ChebyshevBasisSecondKind{T}) = one(T)
right{T}(b::ChebyshevBasisSecondKind{T}, idx) = right(b)

grid{T}(b::ChebyshevBasisSecondKind{T}) = ChebyshevIIGrid{T}(b.n)


# The weight function
weight{T}(b::ChebyshevBasisSecondKind{T}, x) = sqrt(1-T(x)^2)

# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_α(b::ChebyshevBasisSecondKind) = 1//2
jacobi_β(b::ChebyshevBasisSecondKind) = 1//2


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevBasisSecondKind, n::Int) = 2

rec_Bn(b::ChebyshevBasisSecondKind, n::Int) = 0

rec_Cn(b::ChebyshevBasisSecondKind, n::Int) = 1
