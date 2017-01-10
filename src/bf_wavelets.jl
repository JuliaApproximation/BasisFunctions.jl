# bf_wavelets.jl

abstract WaveletBasis{T} <: FunctionSet1d{T}

dyadic_length(b::WaveletBasis) = b.L

length(b::WaveletBasis) = 1<<dyadic_length(b)

BasisFunctions.wavelet(b::WaveletBasis) = b.w

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

resize{B<:WaveletBasis}(b::B, n::Int) = B(wavelet(b),round(Int, log2(n)))

has_grid(::WaveletBasis) = true
# TODO implement transform
has_transform(::WaveletBasis) = true

compatible_grid(set::WaveletBasis, grid::PeriodicEquispacedGrid) =
	(1+(left(set) - left(grid))≈1) && (1+(right(set) - right(grid))≈1) && (length(set)==length(grid))
compatible_grid(set::WaveletBasis, grid::AbstractGrid) = false
has_grid_transform(b::WaveletBasis, dgs, grid) = compatible_grid(b, grid)

left{T}(::WaveletBasis{T}) = T(0)
right{T}(::WaveletBasis{T}) = T(1)

function support{T}(b::WaveletBasis{T}, i::Int)
   l,r = support(primal, length(b), i, dyadic_length(b), wavelet(b))
   l < 0 || r > 1 ? (T(0),T(1)) : (l,r)
end

left(b::WaveletBasis, i::Int) = support(b,i)[1]
right(b::WaveletBasis, i::Int) = support(b,i)[2]

period{T}(::WaveletBasis{T}) = T(1)

grid{T}(b::WaveletBasis{T}) = PeriodicEquispacedGrid(length(b), left(b), right(b), T)

function idx2waveletidx(b::WaveletBasis, idx::Int)
  kind, j, k = wavelet_index(length(b), idx, dyadic_length(b))
  kind, j, k
end

waveletidx2idx(b::WaveletBasis, kind::Kind, j::Int, k::Int) = coefficient_index(kind, j, k)

native_index(b::WaveletBasis, idx::Int) = idx2waveletidx(b,idx)
linear_index(b::WaveletBasis, waveletidx::Tuple{Kind,Int,Int}) = waveletidx2idx(b, waveletidx...)

approximate_native_size(::WaveletBasis, size_l) = 1<<ceil(Int, log2(size_l))

approx_length(::WaveletBasis, n) = 1<<round(Int, log2(size_l))

extension_size(b::WaveletBasis) = 2*length(b)

function eval_element{T, S<:Real}(b::WaveletBasis{T}, idx::Int, x::S; xtol::S = 1e-4, options...)
  kind, j, k = native_index(b, idx)
  evaluate_periodic(primal, kind, wavelet(b), j, k, x; xtol = xtol, options...)
end

function transform_from_grid(src, dest::WaveletBasis, grid; options...)
  @assert compatible_grid(dest, grid)
  DiscreteWaveletTransform(src, dest, wavelet(dest); options...)
end

function transform_to_grid(src::WaveletBasis, dest, grid; options...)
  @assert compatible_grid(src, grid)
  InverseDistreteWaveletTransform(src, dest, wavelet(src); options...)
end

function DiscreteWaveletTransform(src::FunctionSet, dest::FunctionSet, w::DiscreteWavelet; options...)
  FunctionOperator(src, dest, x->full_dwt(x, w, perbound))
end

function InverseDistreteWaveletTransform(src::FunctionSet, dest::FunctionSet, w::DiscreteWavelet; options...)
  FunctionOperator(src, dest, x->full_idwt(x, w, perbound))
end

# TODO use evaluate_periodic_in_dyadicpoints if grid has only dyadic points
# function grid_evaluation_operator(set::WaveletBasis, dgs::DiscreteGridSpace, grid::EquispacedGrid; options)
#
# end

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

instantiate{T}(::Type{DaubechiesWaveletBasis}, n, ::Type{T}) = DaubechiesWaveletBasis(3, approx_length(n), T)

is_compatible{P,T1,T2}(src1::DaubechiesWaveletBasis{P,T1}, src2::DaubechiesWaveletBasis{P,T2}) = true

# TODO ensure that only bases with existing CDFwavelets are built
immutable CDFWaveletBasis{P,Q,T} <: BiorthogonalWaveletBasis{T}
  w   ::    CDFWavelet{P,Q,T}
  L   ::    Int
end

CDFWaveletBasis{T}(P::Int, Q::Int, L::Int, ::Type{T} = Float64) =
  CDFWaveletBasis{P,Q,T}(CDFWavelet{P,Q,T}(),L)

promote_eltype{P,Q,T,S}(b::CDFWaveletBasis{P,Q,T}, ::Type{S}) =
      CDFWaveletBasis(CDFWavelet{P,Q,promote_type(T,S)}(), dyadic_length(b))

instantiate{T}(::Type{CDFWaveletBasis}, n, ::Type{T}) = CDFWaveletBasis(2, 4, approx_length(n), T)

is_compatible{P,Q,T1,T2}(src1::CDFWaveletBasis{P,Q,T1}, src2::CDFWaveletBasis{P,Q,T2}) = true
