# bf_wavelets.jl

"""
`WaveletIndex` contains three parameters, uniquely identifying a wavelet coefficient:
Whether it is a scaling or a wavelet coefficient, the level of the wavelet and the displacement.
"""
struct WaveletIndex
    kind
    j   ::Int
    k   ::Int
end

Int(idxn::WaveletIndex) = coefficient_index(idxn.kind, idxn.j, idxn.k)

Base.show(io::IO, idx::BasisFunctions.NativeIndex{:fourier}) =
	print(io, "Wavelet integer index: $(Int(idx))")

abstract type WaveletBasis{T} <: Dictionary1d{T,T}
end

"""
The number of levels in the wavelet basis
"""
dyadic_length(b::WaveletBasis) = b.L

length(b::WaveletBasis) = 1<<dyadic_length(b)

"""
The wavelet type
"""
BasisFunctions.wavelet(b::WaveletBasis) = b.w

BasisFunctions.name(b::WaveletBasis) = "Basis of "*name(wavelet(b))*" wavelets"

# If only the first 2^L basis elements remain, this is equivalent to a smaller wavelet basis
function subdict(b::WaveletBasis, idx::OrdinalRange)
    if (step(idx)==1) && (first(idx) == 1) && isdyadic(last(idx))
        resize(b, last(idx))
    else
        LargeSubdict(b,idx)
    end
end

#  Extension is possible by putting zeros on the coefficients that correspond to detail information
function apply!{B<:WaveletBasis}(op::Extension, dest::B, src::B, coef_dest, coef_src)
    @assert dyadic_length(dest) > dyadic_length(src)

    coef_dest[1:length(src)] = coef_src[1:length(src)]
    coef_dest[length(src)+1:length(dest)] = 0
end

#  Restriction by discarding all detail information
function apply!{B<:WaveletBasis}(op::Restriction, dest::B, src::B, coef_dest, coef_src)
    @assert dyadic_length(dest) < dyadic_length(src)

    coef_dest[1:length(dest)] = coef_src[1:length(dest)]
    coef_dest
end

has_extension(b::WaveletBasis) = true

approx_length(b::WaveletBasis, n::Int) = 1<<ceil(Int, log2(n))

resize(b::B, n::Int) where {B<:WaveletBasis} = B(wavelet(b),round(Int, log2(n)))

has_grid(::WaveletBasis) = true

has_transform(::WaveletBasis) = true

compatible_grid(set::WaveletBasis, grid::PeriodicEquispacedGrid) =
	(1+(left(set) - left(grid))≈1) && (1+(right(set) - right(grid))≈1) && (length(set)==length(grid))
compatible_grid(set::WaveletBasis, grid::DyadicPeriodicEquispacedGrid) =
	(1+(left(set) - left(grid))≈1) && (1+(right(set) - right(grid))≈1) && (length(set)==length(grid))
compatible_grid(set::WaveletBasis, grid::AbstractGrid) = false
has_grid_transform(b::WaveletBasis, gb, grid) = compatible_grid(b, grid)

left{T}(::WaveletBasis{T}) = T(0)
right{T}(::WaveletBasis{T}) = T(1)

function support(b::WaveletBasis{T}, idxn::WaveletIndex) where {T}
    l,r = support(Primal, idxn.kind, wavelet(b), idxn.j, idxn.k)
    l < 0 || r > 1 ? (T(0),T(1)) : (T(l),T(r))
end

left(b::WaveletBasis, i) = support(b,i)[1]
right(b::WaveletBasis, i) = support(b,i)[2]

period{T}(::WaveletBasis{T}) = T(1)

grid{T}(b::WaveletBasis{T}) = DyadicPeriodicEquispacedGrid(dyadic_length(b), left(b), right(b), T)


"""
`DWTIndexList` defines the map from native indices to linear indices
for a finite wavelet basis, when the indices are ordered in the way they
are expected in the DWT routine.
"""
struct DWTIndexList <: IndexList{WaveletIndex}
	n	::	Int
end

length(list::DWTIndexList) = list.n
size(list::DWTIndexList) = (list.n,)

getindex(m::DWTIndexList, idx::Int) = WaveletIndex(wavelet_index(length(m), idx, Int(log2(length(m))))...)

getindex(list::DWTIndexList, idxn::WaveletIndex) = Int(idxn)

ordering(b::WaveletBasis) = DWTIndexList(length(b))

function idx2waveletidx(b::WaveletBasis, idx::Int)
    kind, j, k = wavelet_index(length(b), idx, dyadic_length(b))
    kind, j, k
end

waveletidx2idx(b::WaveletBasis, kind::Kind, j::Int, k::Int) = coefficient_index(kind, j, k)

native_index(b::WaveletBasis, idx::Int) = WaveletIndex(idx2waveletidx(b,idx)...)
linear_index(b::WaveletBasis, waveletidx::Tuple{Kind,Int,Int}) = waveletidx2idx(b, waveletidx...)

approximate_native_size(::WaveletBasis, size_l) = 1<<ceil(Int, log2(size_l))

approx_length(::WaveletBasis, n) = 1<<round(Int, log2(size_l))

extension_size(b::WaveletBasis) = 2*length(b)



unsafe_eval_element(dict::WaveletBasis, idxn::WaveletIndex, x; xtol=1e-4, options...) =
    evaluate_periodic(Primal, idxn.kind, wavelet(dict), idxn.j, idxn.k, x; xtol = xtol, options...)

function transform_from_grid(src, dest::WaveletBasis, grid; options...)
    @assert compatible_grid(dest, grid)
    L = length(src)
    ELT = eltype(src)
    S = ScalingOperator(dest, 1/sqrt(ELT(L)))
    T = DiscreteWaveletTransform(src, dest, wavelet(dest); options...)
    T*S
end

function transform_to_grid(src::WaveletBasis, dest, grid; options...)
    @assert compatible_grid(src, grid)
    L = length(src)
    ELT = eltype(src)
    S = ScalingOperator(dest, 1/sqrt(ELT(L)))
    T = InverseDistreteWaveletTransform(src, dest, wavelet(src); options...)
    T*S
end

function transform_from_grid_post(src, dest::WaveletBasis, grid; options...)
	@assert compatible_grid(dest, grid)
    L = length(src)
    ELT = eltype(src)
    ScalingOperator(dest, 1/sqrt(ELT(L)))
end

function transform_to_grid_pre(src::WaveletBasis, dest, grid; options...)
	@assert compatible_grid(src, grid)
	inv(transform_from_grid_post(dest, src, grid; options...))
end

function DiscreteWaveletTransform(src::Dictionary, dest::Dictionary, w::DiscreteWavelet; options...)
    FunctionOperator(src, dest, x->full_dwt(x, w, perbound))
end

function InverseDistreteWaveletTransform(src::Dictionary, dest::Dictionary, w::DiscreteWavelet; options...)
    FunctionOperator(src, dest, x->full_idwt(x, w, perbound))
end

# TODO use evaluate_periodic_in_dyadicpoints if grid has only dyadic points
# function grid_evaluation_operator(set::WaveletBasis, dgs::DiscreteGridSpace, grid::EquispacedGrid; options)
#
# end

# Used for fast plot of all elements in a WaveletBasis
#TODO make in place implementation of evaluate_periodic_in_dyadic_points
function eval_element!(result, set::WaveletBasis, idx, grid::DyadicPeriodicEquispacedGrid, outside_value = zero(eltype(set)))
    if (1+(left(set) - left(grid))≈1) && (1+(right(set) - right(grid))≈1)
        kind, j, k = native_index(set, idx)
        result = evaluate_periodic_in_dyadic_points(Primal, kind, wavelet(set), j, k, dyadic_length(grid))
    else
        eval_element!(result, set, idx, PeriodicEquispacedGrid(grid), outside_value)
    end
end

abstract type OrthogonalWaveletBasis{T} <: WaveletBasis{T} end

is_basis(b::OrthogonalWaveletBasis) = true
is_orthogonal(b::OrthogonalWaveletBasis) = true

abstract type BiorthogonalWaveletBasis{T} <: WaveletBasis{T} end

is_basis(b::BiorthogonalWaveletBasis) = true

struct DaubechiesWaveletBasis{P,T} <: OrthogonalWaveletBasis{T}
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
struct CDFWaveletBasis{P,Q,T} <: BiorthogonalWaveletBasis{T}
    w   ::    CDFWavelet{P,Q,T}
    L   ::    Int
end

CDFWaveletBasis{T}(P::Int, Q::Int, L::Int, ::Type{T} = Float64) =
    CDFWaveletBasis{P,Q,T}(CDFWavelet{P,Q,T}(),L)

promote_eltype{P,Q,T,S}(b::CDFWaveletBasis{P,Q,T}, ::Type{S}) =
    CDFWaveletBasis(CDFWavelet{P,Q,promote_type(T,S)}(), dyadic_length(b))

instantiate{T}(::Type{CDFWaveletBasis}, n, ::Type{T}) = CDFWaveletBasis(2, 4, approx_length(n), T)

is_compatible{P,Q,T1,T2}(src1::CDFWaveletBasis{P,Q,T1}, src2::CDFWaveletBasis{P,Q,T2}) = true

@recipe function f(F::WaveletBasis; plot_complex = false, n=200)
    grid = plotgrid(F,n)
    for i in eachindex(F)
        @series begin
            vals = F[i](grid)
            grid, postprocess(F[i],grid,vals)
        end
    end
    nothing
end


plotgrid(b::WaveletBasis, n) = DyadicPeriodicEquispacedGrid(round(Int,log2(n)), left(b), right(b))
