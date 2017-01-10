# fouriertransforms.jl

# We wrap the discrete Fourier and cosine transforms in an operator.
# For Float64 and Complex{Float64} we can use FFTW. The plans that FFTW computes
# support multiplication, and hence we can store them in a MultiplicationOperator.
#
# For other types (like BigFloat) we have to resort to a different implementation.
# We make specific operator types that call fft and dct from FastTransforms.jl
# (dct moved there from this code).
#
# This situation will improve once the pure-julia implementation of FFT lands (#6193).


#############################
# The Fast Fourier transform
#############################

function fftw_scaling_operator(set::FunctionSet)
    scalefactor = 1/sqrt(convert(eltype(set), length(set)))
    ScalingOperator(set, set, scalefactor)
end

function fftw_operator(src::FunctionSet, dest::FunctionSet, dims, fftwflags)
    T = eltype(dest)
    plan = plan_fft!(zeros(T, dest), dims; flags = fftwflags)
    MultiplicationOperator(src, dest, plan; inplace = true)
end

function ifftw_operator(src::FunctionSet, dest::FunctionSet, dims, fftwflags)
    T = eltype(dest)
    plan = plan_bfft!(zeros(T, src), dims; flags = fftwflags)
    MultiplicationOperator(src, dest, plan; inplace = true)
end

# We use the fft routine provided by FFTW, but scale the result by 1/sqrt(N)
# in order to have a unitary transform. Additional scaling is done in the _pre and
# _post routines.
function FastFourierTransformFFTW(src::FunctionSet, dest::FunctionSet,
    dims = 1:ndims(dest); fftwflags = FFTW.MEASURE, options...)

    t_op = fftw_operator(src, dest, dims, fftwflags)
    # TODO: this scaling can't be correct if dims is not equal to 1:ndims(dest)
    s_op = fftw_scaling_operator(dest)
    s_op * t_op
end

# Note that we choose to use bfft, an unscaled inverse fft.
function InverseFastFourierTransformFFTW(src, dest, dims = 1:ndims(src); fftwflags = FFTW.MEASURE, options...)
    t_op = ifftw_operator(src, dest, dims, fftwflags)
    # TODO: this scaling can't be correct if dims is not equal to 1:ndims(dest)
    s_op = fftw_scaling_operator(dest)
    s_op * t_op
end

# We have to know the precise type of the FFT plans in order to intercept calls to
# dimension_operator. These are important to catch, since there are specific FFT-plans
# that work along one dimension and they are more efficient than our own generic implementation.
typealias FFTPLAN{T,N} Base.DFT.FFTW.cFFTWPlan{T,-1,true,N}
typealias IFFTPLAN{T,N} Base.DFT.FFTW.cFFTWPlan{T,1,true,N}

dimension_operator_multiplication(src::FunctionSet, dest::FunctionSet, op::MultiplicationOperator,
    dim, object::FFTPLAN; options...) =
        FastFourierTransformFFTW(src, dest, dim:dim; options...)

dimension_operator_multiplication(src::FunctionSet, dest::FunctionSet, op::MultiplicationOperator,
    dim, object::IFFTPLAN; options...) =
        InverseFastFourierTransformFFTW(src, dest, dim:dim; options...)


ctranspose_multiplication(op::MultiplicationOperator, object::FFTPLAN) =
    ifftw_operator(dest(op), src(op), 1:ndims(dest(op)), object.flags)

ctranspose_multiplication(op::MultiplicationOperator, object::IFFTPLAN) =
    fftw_operator(dest(op), src(op), 1:ndims(dest(op)), object.flags)

inv_multiplication(op::MultiplicationOperator, object::FFTPLAN) =
    ctranspose_multiplication(op, object) * ScalingOperator(dest(op), 1/convert(eltype(op), length(dest(op))))

inv_multiplication(op::MultiplicationOperator, object::IFFTPLAN) =
    ctranspose_multiplication(op, object) * ScalingOperator(dest(op), 1/convert(eltype(op), length(dest(op))))


ctranspose_function(op::FunctionOperator, fun::typeof(fft)) =
    FunctionOperator(dest(op), src(op), ifft) * ScalingOperator(dest(op), dest(op), length(dest(op)))

ctranspose_function(op::FunctionOperator, fun::typeof(ifft)) =
    FunctionOperator(dest(op), src(op), fft) * ScalingOperator(dest(op), dest(op), 1/convert(eltype(op), length(dest(op))))

# The two functions below are used when computing the inv of a
# FunctionOperator that uses fft/ifft:
inv(fun::typeof(fft)) = ifft
inv(fun::typeof(ifft)) = fft


# Now the generic implementation, based on using fft and ifft

function FastFourierTransform(src, dest)
    ELT = op_eltype(src, dest)
    scalefactor = 1/sqrt(ELT(length(src)))
    s_op = ScalingOperator(dest, dest, scalefactor)
    t_op = FunctionOperator(src, dest, fft)

    s_op * t_op
end

function InverseFastFourierTransform(src, dest)
    ELT = op_eltype(src, dest)
    scalefactor = sqrt(ELT(length(src)))
    s_op = ScalingOperator(dest, dest, scalefactor)
    t_op = FunctionOperator(src, dest, ifft)

    s_op * t_op
end



##################################
# The Discrete Cosine transform
##################################

# We implement the standard type II DCT and less standard type I DCT here.
# The type II DCT is used for the discrete Chebyshev transform from Chebyshev roots
# The type I DCT is used for the discrete Chebyshev transforn from Chebyshev extrema
# See Strang's paper for the different versions:
# http://www-math.mit.edu/~gs/papers/dct.pdf

# First: the FFTW routines. We can use MultiplicationOperator with the DCT plans.

for (transform, plan) in
    ((:FastChebyshevTransformFFTW, :plan_dct!),
    (:InverseFastChebyshevTransformFFTW, :plan_idct!))
  @eval begin
    function $transform(src, dest, dims = 1:ndims(dest); fftwflags = FFTW.MEASURE, options...)
        ELT = op_eltype(src, dest)
        plan = FFTW.$plan(zeros(ELT, dest), dims; flags = fftwflags)
        MultiplicationOperator(src, dest, plan; inplace=true)
    end
  end
end

function FastChebyshevITransformFFTW(src, dest, dims = 1:ndims(src); fftwflags = FFTW.MEASURE, options...)
    ELT = op_eltype(src, dest)
    plan = FFTW.plan_r2r!(zeros(ELT, src), FFTW.REDFT00, dims; flags = fftwflags)
    MultiplicationOperator(src, dest, plan; inplace=true)
end

# We have to know the precise type of the DCT plan in order to intercept calls to
# dimension_operator. These are important to catch, since there are specific DCT-plans
# that work along one dimension and they are more efficient than our own generic implementation.
typealias DCTPLAN{T} Base.DFT.FFTW.DCTPlan{T,5,true}
typealias IDCTPLAN{T} Base.DFT.FFTW.DCTPlan{T,4,true}
typealias DCTIPLAN{T} Base.DFT.FFTW.r2rFFTWPlan{T,(3,),false,1}

for (plan, transform, invtransform) in (
      (:DCTPLAN, :FastChebyshevTransformFFTW, :InverseFastChebyshevTransform),
      (:IDCTPLAN, :InverseFastChebyshevTransformFFTW, :FastChebyshevTransform),
      (:DCTIPLAN, :FastChebyshevITransformFFTW, :FastChebyshevITransformFFTW))
  @eval begin
    dimension_operator_multiplication(src::FunctionSet, dest::FunctionSet, op::MultiplicationOperator,
        dim, object::$plan; options...) =
            $transform(src, dest, dim:dim; options...)
    ctranspose_multiplication(op::MultiplicationOperator, object::$plan) =
        $invtransform(dest(op), src(op))
    inv_multiplication(op::MultiplicationOperator, object::$plan) =
        ctranspose_multiplication(op, object)
  end
end

# The four functions below are used when computing the ctranspose or inv of a
# FunctionOperator that uses dct/idct:
inv(fun::typeof(dct)) = idct
inv(fun::typeof(idct)) = dct
ctranspose(fun::typeof(dct)) = idct
ctranspose(fun::typeof(idct)) = dct

# The generic routines for the DCTII. They rely on dct and idct being available for the
# types of coefficients.
FastChebyshevTransform(src, dest) = FunctionOperator(src, dest, dct)
InverseFastChebyshevTransform(src, dest) = FunctionOperator(src, dest, idct)

# The generic routines for the DCTI. They they are not yet available
FastChebyshevITransform(src, dest) = error("The FastChebyshevITransform is not available for the type ", eltype(src))
InverseFastChebyshevITransform(src, dest) = error("The InverseFastChebyshevITransform is not available for the type ", eltype(src))
