# bf_wavelets.jl

abstract WaveletBasis{T} <: FunctionSet1d{T}

dyadic_length(b::WaveletBasis) = b.L

length(b::WaveletBasis) = 1<<dyadic_length(b)

wavelet(b::WaveletBasis) = b.w

BasisFunctions.name(b::WaveletBasis) = "Basis of "*name(wavelet(b))*" wavelets"

# If only the first 2^L basis elements remains, this is equivalent to a smaller wavelet basis
function subset(b::WaveletBasis, idx::OrdinalRange)
  if (step(idx)==1) && (first(idx) == 1) && isdyadic(last(idx))
    resize(b, last(idx))
  else
    FunctionSubSet(b,idx)
  end
end

#  Extension is possible by putting zeros on the coefficients that correspond to detail information
function apply!{B<:WaveletBasis}(op::Extension, dest::B, src::B, coef_dest, coef_src)
  @assert dyadic_length(dest) > dyadic_length(src)

  coef_dest[1:length(src)] = coef_src[1:length(src)]
  coef_dest[length(src)+1:length(dest)] = 0
end
#  Restriction by discaring all detail information
function apply!{B<:WaveletBasis}(op::Restriction, dest::B, src::B, coef_dest, coef_src)
  @assert dyadic_length(dest) < dyadic_length(src)

  coef_dest[1:length(dest)] = coef_src[1:length(dest)]
  coef_dest
end

has_extension(b::WaveletBasis) = true

approx_length(b::WaveletBasis, n::Int) = 1<<ceil(Int, log2(n))

call_element(b::WaveletBasis, idx::Int, x) =
    error("There is no explicit formula for elements of wavelet basis, ", b)
# ASK resize is here defined on dyadic_length
resize{B<:WaveletBasis}(b::B, L::Int) = B(wavelet(b),L)

# TODO implement grid
has_grid(::WaveletBasis) = true
# TODO implement transform
has_transform(::WaveletBasis) = true

left(::WaveletBasis) = 0
right(::WaveletBasis) = 1
function left(w::WaveletBasis, i::Int)
  if i == 1
    support(primal_scalingfilter(wavelet(w)), 0, 0)[1]
  else
    j = level(length(w), i)
    k = mod(i,1<<(j+1))
    support(primal_waveletfilter(wavelet(w)), level(length(w), ), 0)[1]
  end
end

grid{T}(b::WaveletBasis{T}) = PeriodicEquispacedGrid(length(b), left(b), right(b), T)

transform_operator{G<:PeriodicEquispacedGrid}(src::DiscreteGridSpace{G}, dest::WaveletBasis; options...) =
    dwt_operator(src, dest, eltype(src, dest); options...)

transform_operator{G<:PeriodicEquispacedGrid}(src::WaveletBasis, dest::DiscreteGridSpace{G}; options...) =
    idwt_operator(src, dest, eltype(src, dest); options...)



function level(n::Int, i::Int)
  (i == 1 || i == 2) && (return 0)
  for l in 1:round(Int,log2(n))
    if i <= (1<<(l+1))
      return l
    end
  end
end





abstract OrthogonalWaveletBasis{T} <: WaveletBasis{T}

is_basis(b::OrthogonalWaveletBasis) = true
is_orthogonal(b::OrthogonalWaveletBasis) = true

abstract BiorthogonalWaveletBasis{T} <: WaveletBasis{T}

is_basis(b::BiorthogonalWaveletBasis) = true

immutable DaubechiesWaveletBasis{P,T} <: OrthogonalWaveletBasis{T}
  w   ::    DaubechiesWavelet{P,T}
  L   ::    Int
end

DaubechiesWaveletBasis{T}(P::Int, L::Int, ::Type{T} = Float64) =
  DaubechiesWaveletBasis{P,T}(DaubechiesWavelet{P,T}(), L)

promote_eltype{P,T,S}(b::DaubechiesWaveletBasis{P,T}, ::Type{S}) =
      DaubechiesWaveletBasis(DaubechiesWavelet{P,promote_type(T,S)}(), dyadic_length(b))
immutable CDFWaveletBasis{P,Q,T} <: BiorthogonalWaveletBasis{T}
  w   ::    CDFWavelet{P,Q,T}
  L   ::    Int
end

CDFWaveletBasis{T}(P::Int, Q::Int, L::Int, ::Type{T} = Float64) =
  CDFWaveletBasis{P,Q,T}(CDFWavelet{P,Q,T}(),L)

promote_eltype{P,Q,T,S}(b::CDFWaveletBasis{P,Q,T}, ::Type{S}) =
      CDFWaveletBasis(CDFWavelet{P,Q,promote_type(T,S)}(), dyadic_length(b))
