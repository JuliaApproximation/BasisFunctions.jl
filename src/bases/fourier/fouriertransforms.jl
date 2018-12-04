
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

function fftw_scaling_operator(dict::Dictionary)
    T = coefficienttype(dict)
    scalefactor = 1/(convert(T, length(dict)))
    ScalingOperator{T}(dict, dict, scalefactor)
end

for (op, plan_, f) in ((:fftw_operator, :plan_fft!, :fft ),
                            (:ifftw_operator, :plan_bfft!, :bfft))
    # fftw_operator and ifftw_operator take a different route depending on the eltype of dest
    @eval $op(src::Dictionary, dest::Dictionary, dims, fftwflags; T=coefficienttype(dest)) =
        $op(src, dest, T, dims, fftwflags)
    # In the default case apply fft or ifft
    @eval $op(src::Dictionary, dest::Dictionary, ::Type{Complex{T}}, dims, fftwflags) where {T} =
        FunctionOperator{Complex{T}}(src, dest, $f)
    # When possible apply the fast FFTW operator
    for T in (:(Complex{Float32}), :(Complex{Float64}))
    	@eval function $op(src::Dictionary, dest::Dictionary, ::Type{$(T)}, dims, fftwflags)
            plan = $plan_(zeros($(T), dest), dims; flags = fftwflags)
            MultiplicationOperator{$(T)}(src, dest, plan; inplace = true)
        end
    end
end

for (transform, FastTransform, FFTWTransform, fun, op, scalefactor) in ((:forward_fourier_operator, :FastFourierTransform, :FastFourierTransformFFTW, :fft, :fftw_operator, :fft_scalefactor),
                                   (:backward_fourier_operator, :InverseFastFourierTransform, :InverseFastFourierTransformFFTW, :ifft, :ifftw_operator, :ifft_scalefactor))
    # These are the generic fallbacks
    @eval $transform(src, dest, ::Type{Complex{T}}; options...) where {T} =
        $FastTransform(src, dest; T=Complex{T})
    # But for some types we can use FFTW
    for T in (:(Complex{Float32}), :(Complex{Float64}))
        @eval function $transform(src, dest, ::Type{$(T)}; options...)
            # TODO: this scaling can't be correct if dims is not equal to 1:dimension(dest)
            s_op = ScalingOperator{$(T)}(dest, dest, $scalefactor(src,$(T)))
            t_op = $FFTWTransform(src, dest; T=$(T), options...)
            s_op * t_op
  	end
    end

    # Now the generic implementation, based on using fft and ifft
    @eval function $FastTransform(src, dest; T = op_eltype(src,dest))
        s_op = ScalingOperator{T}(dest, dest, $scalefactor(src, T))
        t_op = FunctionOperator{T}(src, dest, $fun)

        s_op*t_op
    end

    # We use the fft routine provided by FFTW, but scale the result by 1/sqrt(N)
    # in order to have a unitary transform. Additional scaling is done in the _pre and
    # _post routines.
    # Note that we choose to use bfft, an unscaled inverse fft.
    @eval function $FFTWTransform(src::Dictionary, dest::Dictionary,
        dims = 1:dimension(src); fftwflags = FFTW.MEASURE, options...)

        t_op = $op(src, dest, dims, fftwflags)
        t_op
    end
end

fft_scalefactor(src, ::Type{T}) where {T} = 1/convert(T, length(src))

ifft_scalefactor(src, ::Type{T}) where {T} = convert(T,  length(src))

ifft_scalefactor(src, ::Type{Complex{Float64}}) = 1

ifft_scalefactor(src, ::Type{Complex{Float32}}) = 1


# We have to know the precise type of the FFT plans in order to intercept calls to
# dimension_operator. These are important to catch, since there are specific FFT-plans
# that work along one dimension and they are more efficient than our own generic implementation.
FFTPLAN{T,N} = FFTW.cFFTWPlan{T,-1,true,N}
IFFTPLAN{T,N,S} = FFTW.cFFTWPlan{T,1,true,N}

#IFFTPLAN{T,N,S} = Base.DFT.ScaledPlan{T,FFTW.cFFTWPlan{T,1,true,N},S}

dimension_operator_multiplication(src::Dictionary, dest::Dictionary, op::MultiplicationOperator,
    dim, object::FFTPLAN; options...) =
        FastFourierTransformFFTW(src, dest, dim:dim; options...)

dimension_operator_multiplication(src::Dictionary, dest::Dictionary, op::MultiplicationOperator,
    dim, object::IFFTPLAN; options...) =
        InverseFastFourierTransformFFTW(src, dest, dim:dim; options...)


adjoint_multiplication(op::MultiplicationOperator, object::FFTPLAN) =
    ifftw_operator(dest(op), src(op), 1:dimension(dest(op)), object.flags)

adjoint_multiplication(op::MultiplicationOperator, object::IFFTPLAN) =
    fftw_operator(dest(op), src(op), 1:dimension(dest(op)),object.flags)

inv_multiplication(op::MultiplicationOperator, object::FFTPLAN) =
#        adjoint_multiplication(op, object)
adjoint_multiplication(op, object) * ScalingOperator(dest(op), 1/convert(eltype(op), length(dest(op))))

inv_multiplication(op::MultiplicationOperator, object::IFFTPLAN) =
#        adjoint_multiplication(op, object)
    adjoint_multiplication(op, object) * ScalingOperator(dest(op), 1/convert(eltype(op), length(dest(op))))


adjoint_function(op::FunctionOperator, fun::typeof(fft)) =
    FunctionOperator(dest(op), src(op), ifft) * ScalingOperator(dest(op), dest(op), length(dest(op)))

adjoint_function(op::FunctionOperator, fun::typeof(ifft)) =
    FunctionOperator(dest(op), src(op), fft) * ScalingOperator(dest(op), dest(op), 1/convert(eltype(op), length(dest(op))))

# The two functions below are used when computing the inv of a
# FunctionOperator that uses fft/ifft:
inv(fun::typeof(fft)) = ifft
inv(fun::typeof(ifft)) = fft






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
        function $transform(src, dest, dims = 1:dimension(dest);
                    fftwflags = FFTW.MEASURE, T = op_eltype(src,dest), options...)
            plan = FFTW.$plan(zeros(T, src), dims; flags = fftwflags)
            MultiplicationOperator(src, dest, plan; inplace=true)
        end
    end
end

function FastChebyshevITransformFFTW(src, dest, dims = 1:dimension(src);
        fftwflags = FFTW.MEASURE, T = op_eltype(src, dest), options...)
    plan = FFTW.plan_r2r!(zeros(T, src), FFTW.REDFT00, dims; flags = fftwflags)
    MultiplicationOperator(src, dest, plan; inplace=true)
end

# We have to know the precise type of the DCT plan in order to intercept calls to
# dimension_operator. These are important to catch, since there are specific DCT-plans
# that work along one dimension and they are more efficient than our own generic implementation.
DCTPLAN{T} = FFTW.DCTPlan{T,5,true}
IDCTPLAN{T} = FFTW.DCTPlan{T,4,true}
DCTIPLAN{T} = FFTW.r2rFFTWPlan{T,(3,),false,1}

for (plan, transform, invtransform) in (
      (:DCTPLAN, :FastChebyshevTransformFFTW, :InverseFastChebyshevTransformFFTW),
      (:IDCTPLAN, :InverseFastChebyshevTransformFFTW, :FastChebyshevTransformFFTW),
      (:DCTIPLAN, :FastChebyshevITransformFFTW, :FastChebyshevITransformFFTW))
    @eval begin
        dimension_operator_multiplication(src::Dictionary, dest::Dictionary, op::MultiplicationOperator,
            dim, object::$plan; options...) =
                $transform(src, dest, dim:dim; options...)

        adjoint_multiplication(op::MultiplicationOperator, object::$plan) =
            $invtransform(dest(op), src(op))

        inv_multiplication(op::MultiplicationOperator, object::$plan) =
            adjoint_multiplication(op, object)
    end
end

# The four functions below are used when computing the adjoint or inv of a
# FunctionOperator that uses dct/idct:
inv(fun::typeof(dct)) = idct
inv(fun::typeof(idct)) = dct

adjoint(fun::typeof(dct)) = idct
adjoint(fun::typeof(idct)) = dct

# The generic routines for the DCTII. They rely on dct and idct being available for the
# types of coefficients.
FastChebyshevTransform(src, dest; T = op_eltype(src, dest), options...) = FunctionOperator{T}(src, dest, dct)
InverseFastChebyshevTransform(src, dest; T = op_eltype(src,dest), options...) = FunctionOperator{T}(src, dest, idct)

# The generic routines for the DCTI. They they are not yet available
FastChebyshevITransform(src, dest) = error("The FastChebyshevITransform is not available for the type ", eltype(src))
InverseFastChebyshevITransform(src, dest) = error("The InverseFastChebyshevITransform is not available for the type ", eltype(src))
