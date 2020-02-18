
# We wrap the discrete Fourier and cosine transforms in an operator.
# For Float64 and Complex{Float64} we can use FFTW. The plans that FFTW computes
# support multiplication, and hence we can store them in a MultiplicationOperator.
#
# For other types (like BigFloat) we have to resort to a different implementation.
# We make specific operator types that call fft and dct from FastTransforms.jl
# (dct moved there from this code).
#
# This situation will improve once the pure-julia implementation of FFT lands (#6193).

# The types T for which we can use FFTW's fft routines (with element type Complex{T})
FFTW_TYPES = Union{Float32,Float64}

# The types for which we can use FFTW's dct routines
DCT_FFTW_TYPES = Union{Float32,Float64,Complex{Float32},Complex{Float64}}


#############################
# The Fast Fourier transform
#############################

inverse_fourier_operator(src, dest; options...) =
    inverse_fourier_operator(src, dest, operatoreltype(src,dest); options...)
forward_fourier_operator(src, dest; options...) =
    forward_fourier_operator(src, dest, operatoreltype(src,dest); options...)


# For any type but the FFTW types
inverse_fourier_operator(src, dest, ::Type{Complex{T}}; options...) where {T} =
    length(dest) * FunctionOperator{T}(src, dest, ifft)
forward_fourier_operator(src, dest, ::Type{Complex{T}}; options...) where {T} =
    inv(inverse_fourier_operator(dest, src, Complex{T}; options...))


inverse_fourier_operator(src, dest, ::Type{Complex{T}};
            fftwflags = FFTW.MEASURE, options...) where {T <: FFTW_TYPES} =
    bfftw_operator(src, dest, Complex{T}; options...)

function forward_fourier_operator(src, dest, ::Type{Complex{T}};
            fftwflags = FFTW.MEASURE, options...) where {T <: FFTW_TYPES}
    t_op = fftw_operator(src, dest, Complex{T}; options...)
    1/convert(T, length(src)) * t_op
end


function bfftw_operator(src, dest, ::Type{T}; fftwflags = FFTW.MEASURE, options...) where {T}
    dims = 1:dimension(src)
    plan = plan_bfft!(zeros(T, dest), dims; flags = fftwflags)
    MultiplicationOperator{T}(src, dest, plan; inplace = true)
end

function fftw_operator(src, dest, ::Type{T}; fftwflags = FFTW.MEASURE, options...) where {T}
    dims = 1:dimension(src)
    plan = plan_fft!(zeros(T, dest), dims; flags = fftwflags)
    MultiplicationOperator{T}(src, dest, plan; inplace = true)
end



# We have to know the precise type of the FFT plans in order to intercept calls to
# adjoint and inv.
FFTPLAN{T,N} = FFTW.cFFTWPlan{T,-1,true,N}
IFFTPLAN{T,N,S} = FFTW.cFFTWPlan{T,1,true,N}

# We define adjoints ourselves, since FFTW doesn't
adjoint_multiplication(op::MultiplicationOperator, object::FFTPLAN) =
    bfftw_operator(dest(op), src(op), eltype(object); fftwflags=object.flags)

adjoint_multiplication(op::MultiplicationOperator, object::IFFTPLAN) =
    fftw_operator(dest(op), src(op), eltype(object); fftwflags=object.flags)

# Explicitly return an inverse in order to avoid creating a scaled FFT plan.
inv_multiplication(op::MultiplicationOperator, object::FFTPLAN) =
    1/convert(eltype(object), length(src(op))) * bfftw_operator(dest(op), src(op), eltype(object); fftwflags=object.flags)

inv_multiplication(op::MultiplicationOperator, object::IFFTPLAN) =
    1/convert(eltype(object), length(src(op))) * fftw_operator(dest(op), src(op), eltype(object); fftwflags=object.flags)


adjoint_function(op::FunctionOperator, fun::typeof(fft)) =
    FunctionOperator(dest(op), src(op), ifft) * ScalingOperator(dest(op), length(dest(op)))

adjoint_function(op::FunctionOperator, fun::typeof(ifft)) =
    FunctionOperator(dest(op), src(op), fft) * ScalingOperator(dest(op), 1/convert(eltype(op), length(dest(op))))

# The two functions below are used when computing the inv of a
# FunctionOperator that uses fft/ifft:
inv(fun::typeof(fft)) = ifft
inv(fun::typeof(ifft)) = fft




##################################
# The Discrete Cosine transform
##################################

# We invoke the standard type II DCT and less standard type I DCT here.
# The type II DCT is used for the discrete Chebyshev transform from Chebyshev roots
# The type I DCT is used for the discrete Chebyshev transform from Chebyshev extrema
# See Strang's paper for the different versions:
# http://www-math.mit.edu/~gs/papers/dct.pdf

inverse_chebyshev_operator(src, dest; options...) =
    inverse_chebyshev_operator(src, dest, operatoreltype(src,dest); options...)
forward_chebyshev_operator(src, dest; options...) =
    forward_chebyshev_operator(src, dest, operatoreltype(src,dest); options...)

inverse_chebyshevI_operator(src, dest; options...) =
    inverse_chebyshevI_operator(src, dest, operatoreltype(src,dest); options...)
forward_chebyshevI_operator(src, dest; options...) =
    forward_chebyshevI_operator(src, dest, operatoreltype(src,dest); options...)

# The generic routines for the DCTII. They rely on dct and idct being available for the
# types of coefficients.
inverse_chebyshev_operator(src, dest, ::Type{T}; options...) where {T} =
    FunctionOperator{T}(src, dest, idct)
forward_chebyshev_operator(src, dest, ::Type{T}; options...) where {T} =
    FunctionOperator{T}(src, dest, dct)

# The generic routines for the DCTI. They they are not yet available
inverse_chebyshevI_operator(src, dest, ::Type{T}; options...) where {T} =
    error("The inverse DCTI-type transform is not available for type ", eltype(src))
forward_chebyshevI_operator(src, dest, ::Type{T}; options...) where {T} =
    error("The DCTI-type transform is not available for type ", eltype(src))


# For the FFTW routines, we use a MultiplicationOperator with the DCT plans.
inverse_chebyshev_operator(src, dest, ::Type{T};
            fftwflags = FFTW.MEASURE, options...) where {T <: DCT_FFTW_TYPES} =
    idctw_operator(src, dest, T; fftwflags = fftwflags, options...)

forward_chebyshev_operator(src, dest, ::Type{T};
            fftwflags = FFTW.MEASURE, options...) where {T <: DCT_FFTW_TYPES} =
    dctw_operator(src, dest, T; fftwflags = fftwflags, options...)


function idctw_operator(src, dest, ::Type{T}; fftwflags = FFTW.MEASURE, options...) where {T}
    dims = 1:dimension(src)
    plan = plan_idct!(zeros(T, dest), dims; flags = fftwflags)
    MultiplicationOperator{T}(src, dest, plan; inplace = true)
end

function dctw_operator(src, dest, ::Type{T}; fftwflags = FFTW.MEASURE, options...) where {T}
    dims = 1:dimension(src)
    plan = plan_dct!(zeros(T, dest), dims; flags = fftwflags)
    MultiplicationOperator{T}(src, dest, plan; inplace = true)
end

inverse_chebyshevI_operator(src, dest, ::Type{T};
            fftwflags = FFTW.MEASURE, options...) where {T <: DCT_FFTW_TYPES} =
    dctIw_operator(src, dest, T; fftwflags = fftwflags, options...)

forward_chebyshevI_operator(src, dest, ::Type{T};
            fftwflags = FFTW.MEASURE, options...) where {T <: DCT_FFTW_TYPES} =
    dctIw_operator(dest, src, T; fftwflags = fftwflags, options...)


function dctIw_operator(src, dest, ::Type{T}; fftwflags = FFTW.MEASURE, options...) where {T}
    dims = 1:dimension(src)
    plan = FFTW.plan_r2r!(zeros(T, src), FFTW.REDFT00, dims; flags = fftwflags)
    MultiplicationOperator{T}(src, dest, plan; inplace = true)
end


# We have to know the precise type of the DCT plans in order to intercept calls to
# adjoint and inv.
DCTII_PLAN{T} = FFTW.DCTPlan{T,5,true}
INV_DCTII_PLAN{T} = FFTW.DCTPlan{T,4,true}
DCTI_PLAN{T} = FFTW.r2rFFTWPlan{T,(3,),true,1}

const DCTPLANS = Union{DCTII_PLAN,INV_DCTII_PLAN,DCTI_PLAN}

inv_multiplication(op::MultiplicationOperator, object::DCTII_PLAN) =
    idctw_operator(dest(op), src(op), eltype(object))

inv_multiplication(op::MultiplicationOperator, object::INV_DCTII_PLAN) =
    dctw_operator(dest(op), src(op), eltype(object))

inv_multiplication(op::MultiplicationOperator, object::DCTI_PLAN) =
    1/(2*convert(eltype(object), size(op,1))-2) * dctIw_operator(dest(op), src(op), eltype(object))

adjoint_multiplication(op::MultiplicationOperator, object::INV_DCTII_PLAN) =
    inv_multiplication(op, object)

adjoint_multiplication(op::MultiplicationOperator, object::DCTII_PLAN) =
    inv_multiplication(op, object)

function adjoint_multiplication(op::MultiplicationOperator, object::DCTI_PLAN)
    T = eltype(object)
    diag = ones(T, size(op,1))
    diag[1] = 2
    diag[end] = 2
    D1 = DiagonalOperator(dest(op), dest(op), diag)
    D2 = DiagonalOperator(src(op), src(op), inv.(diag))
    D2 * op * D1
end


# The four functions below are used when computing the adjoint or inv of a
# FunctionOperator that uses dct/idct:
inv(fun::typeof(dct)) = idct
inv(fun::typeof(idct)) = dct

adjoint(fun::typeof(dct)) = idct
adjoint(fun::typeof(idct)) = dct
