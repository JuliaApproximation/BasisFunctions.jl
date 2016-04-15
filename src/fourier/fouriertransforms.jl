# fouriertransforms.jl

# We wrap the discrete Fourier and cosine transforms in an operator.
# Separate definitions are used for Float64 and BigFloat type (for the time being).


abstract DiscreteFourierTransform{SRC,DEST} <: AbstractOperator{SRC,DEST}

abstract DiscreteFourierTransformFFTW{SRC,DEST} <: DiscreteFourierTransform{SRC,DEST}

# These types use FFTW and so they are (currently) limited to Float64.
# This will improve once the pure-julia implementation of FFT lands (#6193).
# But, we can also borrow from ApproxFun so let's do that right away

is_inplace{O <: DiscreteFourierTransformFFTW}(::Type{O}) = True


immutable FastFourierTransformFFTW{SRC,DEST} <: DiscreteFourierTransformFFTW{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
    plan!   ::  Base.DFT.FFTW.cFFTWPlan

    FastFourierTransformFFTW(src, dest, dims = 1:dim(dest); fftwflags = FFTW.MEASURE, options...) =
        new(src, dest, plan_fft!(zeros(eltype(dest),size(dest)), dims; flags = fftwflags))
end

FastFourierTransformFFTW{SRC,DEST}(src::SRC, dest::DEST; options...) = FastFourierTransformFFTW{SRC,DEST}(src, dest; options...)

dimension_operator{SRC,DEST}(src::SRC, dest::DEST, op::FastFourierTransformFFTW, dim; options...) =
    FastFourierTransformFFTW{SRC,DEST}(src, dest, dim:dim; options...)

# Note that we choose to use bfft, an unscaled inverse fft.
immutable InverseFastFourierTransformFFTW{SRC,DEST} <: DiscreteFourierTransformFFTW{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
    plan!   ::  Base.DFT.FFTW.cFFTWPlan

    InverseFastFourierTransformFFTW(src, dest, dims = 1:dim(src); fftwflags = FFTW.MEASURE, options...) =
        new(src, dest, plan_bfft!(zeros(eltype(src),size(src)), dims; flags = fftwflags))
end

InverseFastFourierTransformFFTW{SRC,DEST}(src::SRC, dest::DEST; options...) =
    InverseFastFourierTransformFFTW{SRC,DEST}(src, dest; options...)

dimension_operator{SRC,DEST}(src::SRC, dest::DEST, op::InverseFastFourierTransformFFTW, dim; options...) =
    InverseFastFourierTransformFFTW{SRC,DEST}(src, dest, dim:dim; options...)

function apply!(op::FastFourierTransformFFTW, dest, src, coef_srcdest)
    op.plan!*coef_srcdest
    for i = 1:length(coef_srcdest)
        coef_srcdest[i]/=sqrt(length(coef_srcdest))
    end
    coef_srcdest
end

function apply!(op::InverseFastFourierTransformFFTW, dest, src, coef_srcdest)
    op.plan!*coef_srcdest
    for i = 1:length(coef_srcdest)
        coef_srcdest[i]/=sqrt(length(coef_srcdest))
    end
    coef_srcdest
end


immutable FastFourierTransform{SRC,DEST} <: DiscreteFourierTransform{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

# Our alternative for non-Float64 is to use ApproxFun's fft, at least for 1d.
# This allocates memory.
function apply!(op::FastFourierTransform, dest, src, coef_dest, coef_src)
    coef_dest[:] = fft(coef_src)/sqrt(convert(eltype(coef_src),length(coef_src)))
    coef_dest
end


immutable InverseFastFourierTransform{SRC,DEST} <: DiscreteFourierTransform{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

# Why was the below line necessary?
## apply!(op::InverseFastFourierTransform, dest, src, coef_dest::Array{Complex{BigFloat}}, coef_src::Array{Complex{BigFloat}}) = (coef_dest[:] = ifft(coef_src) )
function apply!(op::InverseFastFourierTransform, dest, src, coef_dest, coef_src)
    coef_dest[:] = ifft(coef_src) * sqrt(convert(eltype(coef_src),length(coef_src)))
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

abstract DiscreteChebyshevTransform{SRC,DEST} <: AbstractOperator{SRC,DEST}

abstract DiscreteChebyshevTransformFFTW{SRC,DEST} <: DiscreteChebyshevTransform{SRC,DEST}

# These types use FFTW and so they are (currently) limited to Float64.
# This may improve once the pure-julia implementation of FFT lands (#6193).

is_inplace{O <: DiscreteChebyshevTransformFFTW}(::Type{O}) = True


immutable FastChebyshevTransformFFTW{SRC,DEST} <: DiscreteChebyshevTransformFFTW{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
    plan!   ::  Base.DFT.FFTW.DCTPlan

    FastChebyshevTransformFFTW(src, dest, dims = 1:dim(dest); fftwflags = FFTW.MEASURE, options...) =
        new(src, dest, plan_dct!(zeros(eltype(dest),size(dest)), dims; flags = fftwflags))
end

FastChebyshevTransformFFTW{SRC,DEST}(src::SRC, dest::DEST; options...) =
    FastChebyshevTransformFFTW{SRC,DEST}(src, dest; options...)

dimension_operator{SRC,DEST}(src::SRC, dest::DEST, op::FastChebyshevTransformFFTW, dim; options...) =
    FastChebyshevTransformFFTW{SRC,DEST}(src, dest, dim:dim; options...)


immutable InverseFastChebyshevTransformFFTW{SRC,DEST} <: DiscreteChebyshevTransformFFTW{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
    plan!   ::  Base.DFT.FFTW.DCTPlan

    InverseFastChebyshevTransformFFTW(src, dest, dims = 1:dim(src); fftwflags = FFTW.MEASURE, options...) =
        new(src, dest, plan_idct!(zeros(eltype(dest),size(src)), dims; flags = fftwflags))
end

InverseFastChebyshevTransformFFTW{SRC,DEST}(src::SRC, dest::DEST; options...) =
    InverseFastChebyshevTransformFFTW{SRC,DEST}(src, dest; options...)

dimension_operator{SRC,DEST}(src::SRC, dest::DEST, op::InverseFastChebyshevTransformFFTW, dim; options...) =
    InverseFastChebyshevTransformFFTW{SRC,DEST}(src, dest, dim:dim; options...)

function apply!(op::DiscreteChebyshevTransformFFTW, dest, src, coef_srcdest)
    op.plan!*coef_srcdest
end


immutable FastChebyshevTransform{SRC,DEST} <: DiscreteChebyshevTransform{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

# Our alternative for non-Float64 is to use ApproxFun's fft, at least for 1d.
# This allocates memory.
# We have to implement dct in terms of fft, which allocates more memory.
function dct(a::AbstractArray{Complex{BigFloat}})
    N = big(length(a))
    c = fft([a; flipdim(a,1)])
    d = c[1:N] .* exp(-im*big(pi)*(0:N-1)/(2*N))
    d[1] = d[1] / sqrt(big(2))
    d / sqrt(2*N)
end

dct(a::AbstractArray{BigFloat}) = real(dct(a+0im))

function idct(a::AbstractArray{Complex{BigFloat}})
    N = big(length(a))
    b = a * sqrt(2*N)
    b[1] = b[1] * sqrt(big(2))
    b = b .* exp(im*big(pi)*(0:N-1)/(2*N))
    b = [b; 0; conj(flipdim(b[2:end],1))]
    c = ifft(b)
    c[1:N]
end

idct(a::AbstractArray{BigFloat}) = real(idct(a+0im))

apply!(op::FastChebyshevTransform, dest, src, coef_dest, coef_src) = (coef_dest[:] = dct(coef_src))


immutable InverseFastChebyshevTransform{SRC,DEST} <: DiscreteChebyshevTransform{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

apply!(op::InverseFastChebyshevTransform, dest, src, coef_dest, coef_src) = (coef_dest[:] = idct(coef_src))


ctranspose(op::FastChebyshevTransform) = InverseFastChebyshevTransform(dest(op), src(op))
ctranspose(op::FastChebyshevTransformFFTW) = InverseFastChebyshevTransformFFTW(dest(op), src(op))

ctranspose(op::InverseFastChebyshevTransform) = FastChebyshevTransform(dest(op), src(op))
ctranspose(op::InverseFastChebyshevTransformFFTW) = FastChebyshevTransformFFTW(dest(op), src(op))

inv(op::DiscreteChebyshevTransform) = ctranspose(op)
