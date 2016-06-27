# fouriertransforms.jl

# We wrap the discrete Fourier and cosine transforms in an operator.
# Separate definitions are used for Float64 and BigFloat type (for the time being).


abstract DiscreteFourierTransform{ELT} <: AbstractOperator{ELT}

abstract DiscreteFourierTransformFFTW{ELT} <: DiscreteFourierTransform{ELT}

# These types use FFTW and so they are (currently) limited to Float64.
# This will improve once the pure-julia implementation of FFT lands (#6193).
# But, we can also borrow from ApproxFun so let's do that right away

is_inplace(::DiscreteFourierTransformFFTW) = true


immutable FastFourierTransformFFTW{ELT} <: DiscreteFourierTransformFFTW{ELT}
    src         ::  FunctionSet
    dest        ::  FunctionSet
    plan!       ::  Base.DFT.FFTW.cFFTWPlan
    scalefactor ::  ELT

    function FastFourierTransformFFTW(src, dest, dims; fftwflags = FFTW.MEASURE, options...)
        scalefactor = one(ELT)
        for dim in dims
            scalefactor *= size(dest,dim)
        end
        scalefactor = 1/sqrt(scalefactor)
        new(src, dest, plan_fft!(zeros(ELT,size(dest)), dims; flags = fftwflags), scalefactor)
    end
end

FastFourierTransformFFTW(src::FunctionSet, dest::FunctionSet, dims = 1:ndims(dest); options...) =
    FastFourierTransformFFTW{op_eltype(src,dest)}(src, dest, dims; options...)

dimension_operator(src::FunctionSet, dest::FunctionSet, op::FastFourierTransformFFTW, dim; options...) =
    FastFourierTransformFFTW(src, dest, dim:dim; options...)

# Note that we choose to use bfft, an unscaled inverse fft.
immutable InverseFastFourierTransformFFTW{ELT} <: DiscreteFourierTransformFFTW{ELT}
    src         ::  FunctionSet
    dest        ::  FunctionSet
    plan!       ::  Base.DFT.FFTW.cFFTWPlan
    scalefactor ::  ELT

    function InverseFastFourierTransformFFTW(src, dest, dims; fftwflags = FFTW.MEASURE, options...)
        scalefactor = one(ELT)
        for dim in dims
            scalefactor *= size(dest,dim)
        end
        scalefactor = 1/sqrt(scalefactor)
        new(src, dest, plan_bfft!(zeros(ELT,size(src)), dims; flags = fftwflags), scalefactor)
    end
end

InverseFastFourierTransformFFTW(src, dest, dims = 1:ndims(src); options...) =
    InverseFastFourierTransformFFTW{op_eltype(src,dest)}(src, dest, dims; options...)

dimension_operator(src::FunctionSet, dest::FunctionSet, op::InverseFastFourierTransformFFTW, dim; options...) =
    InverseFastFourierTransformFFTW(src, dest, dim:dim; options...)

function apply_inplace!(op::DiscreteFourierTransformFFTW, coef_srcdest)
    op.plan!*coef_srcdest
    for i in eachindex(coef_srcdest)
        coef_srcdest[i] *= op.scalefactor
    end
    coef_srcdest
end


immutable FastFourierTransform{ELT} <: DiscreteFourierTransform{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
end

FastFourierTransform(src, dest) = FastFourierTransform{op_eltype(src,dest)}(src,dest)

# Our alternative for non-Float64 is to use ApproxFun's fft, at least for 1d.
# This allocates memory.
function apply!{ELT}(op::FastFourierTransform{ELT}, coef_dest, coef_src)
    l = sqrt(ELT(length(coef_src)))
    coef_dest[:] = fft(coef_src) / l
    coef_dest
end


immutable InverseFastFourierTransform{ELT} <: DiscreteFourierTransform{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
end

InverseFastFourierTransform(src, dest) =
    InverseFastFourierTransform{op_eltype(src,dest)}(src,dest)

function apply!{ELT}(op::InverseFastFourierTransform{ELT}, coef_dest, coef_src)
    l = sqrt(ELT(length(coef_src)))
    coef_dest[:] = ifft(coef_src) * l
    coef_dest
end

ctranspose(op::FastFourierTransform) = InverseFastFourierTransform(dest(op), src(op))
ctranspose(op::FastFourierTransformFFTW) = InverseFastFourierTransformFFTW(dest(op), src(op))

ctranspose(op::InverseFastFourierTransform) = FastFourierTransform(dest(op), src(op))
ctranspose(op::InverseFastFourierTransformFFTW) = FastFourierTransformFFTW(dest(op), src(op))

inv(op::DiscreteFourierTransform) = ctranspose(op)


##################################
# The Discrete Cosine transform
##################################

# We only implement the standard type II DCT.
# See Strang's paper for the different versions:
# http://www-math.mit.edu/~gs/papers/dct.pdf

abstract DiscreteChebyshevTransform{ELT} <: AbstractOperator{ELT}

abstract DiscreteChebyshevTransformFFTW{ELT} <: DiscreteChebyshevTransform{ELT}

# These types use FFTW and so they are (currently) limited to Float64.
# This may improve once the pure-julia implementation of FFT lands (#6193).

is_inplace(::DiscreteChebyshevTransformFFTW) = true


immutable FastChebyshevTransformFFTW{ELT} <: DiscreteChebyshevTransformFFTW{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
    plan!   ::  Base.DFT.FFTW.DCTPlan

    FastChebyshevTransformFFTW(src, dest, dims; fftwflags = FFTW.MEASURE, options...) =
        new(src, dest, plan_dct!(zeros(ELT,size(dest)), dims; flags = fftwflags))
end

FastChebyshevTransformFFTW(src, dest, dims = 1:ndims(dest); options...) =
    FastChebyshevTransformFFTW{op_eltype(src, dest)}(src, dest, dims; options...)

dimension_operator(src::FunctionSet, dest::FunctionSet, op::FastChebyshevTransformFFTW, dim; options...) =
    FastChebyshevTransformFFTW(src, dest, dim:dim; options...)


immutable InverseFastChebyshevTransformFFTW{ELT} <: DiscreteChebyshevTransformFFTW{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
    plan!   ::  Base.DFT.FFTW.DCTPlan

    InverseFastChebyshevTransformFFTW(src, dest, dims; fftwflags = FFTW.MEASURE, options...) =
        new(src, dest, plan_idct!(zeros(ELT,size(src)), dims; flags = fftwflags))
end

InverseFastChebyshevTransformFFTW(src, dest, dims = 1:ndims(src); options...) =
    InverseFastChebyshevTransformFFTW{op_eltype(src,dest)}(src, dest, dims; options...)

dimension_operator(src::FunctionSet, dest::FunctionSet, op::InverseFastChebyshevTransformFFTW, dim; options...) =
    InverseFastChebyshevTransformFFTW(src, dest, dim:dim; options...)

function apply_inplace!(op::DiscreteChebyshevTransformFFTW, coef_srcdest)
    op.plan!*coef_srcdest
end


immutable FastChebyshevTransform{ELT} <: DiscreteChebyshevTransform{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
end

FastChebyshevTransform(src, dest) = FastChebyshevTransform{op_eltype(src,dest)}(src,dest)

# Line below relies on a dct being available for the type of coefficients
# In particular, for BigFloat's we rely on FastTransforms.jl
# Same for idct further below.
apply!(op::FastChebyshevTransform, coef_dest, coef_src) = (coef_dest[:] = dct(coef_src))


immutable InverseFastChebyshevTransform{ELT} <: DiscreteChebyshevTransform{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
end

InverseFastChebyshevTransform(src, dest) = InverseFastChebyshevTransform{op_eltype(src,dest)}(src,dest)

apply!(op::InverseFastChebyshevTransform, coef_dest, coef_src) = (coef_dest[:] = idct(coef_src))


ctranspose(op::FastChebyshevTransform) = InverseFastChebyshevTransform(dest(op), src(op))
ctranspose(op::FastChebyshevTransformFFTW) = InverseFastChebyshevTransformFFTW(dest(op), src(op))

ctranspose(op::InverseFastChebyshevTransform) = FastChebyshevTransform(dest(op), src(op))
ctranspose(op::InverseFastChebyshevTransformFFTW) = FastChebyshevTransformFFTW(dest(op), src(op))

inv(op::DiscreteChebyshevTransform) = ctranspose(op)
