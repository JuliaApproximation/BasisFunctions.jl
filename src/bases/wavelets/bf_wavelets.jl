# bf_wavelets.jl

abstract type WaveletBasis{T,S,K} <: Dictionary1d{T,T} where {S <: Side,K<:Kind}
end

native_index(dict::WaveletBasis, idx::WaveletIndex) = idx


checkbounds(::Type{Bool}, dict::WaveletBasis, i::WaveletIndex) =
    checkbounds(Bool, dict, linear_index(dict, i))

"Create a similar wavelet basis, but replacing the wavelet with its biorthogonal dual"
function wavelet_dual end

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

side(::WaveletBasis{T,S}) where {T,S} = S()

kind(::WaveletBasis{T,S,K}) where {T,S,K} = K()

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

has_extension(b::WaveletBasis{T,S,Wvl}) where{T,S} = true

approx_length(b::WaveletBasis, n::Int) = 1<<ceil(Int, log2(n))

resize(b::B, n::Int) where {B<:WaveletBasis} = B(wavelet(b),round(Int, log2(n)))

has_grid(::WaveletBasis) = true

has_transform(::WaveletBasis) = true
has_unitary_transform(::WaveletBasis) = false

compatible_grid(set::WaveletBasis, grid::PeriodicEquispacedGrid) =
    has_grid_equal_span(set,grid) && (length(set)==length(grid))
compatible_grid(set::WaveletBasis, grid::DyadicPeriodicEquispacedGrid) =
	has_grid_equal_span(set,grid) && (length(set)==length(grid))
compatible_grid(set::WaveletBasis, grid::AbstractGrid) = false
has_grid_transform(b::WaveletBasis, gb, grid) = compatible_grid(b, grid)

support(b::WaveletBasis) = UnitInterval{domaintype(b)}()
left{T}(::WaveletBasis{T}) = T(0)
right{T}(::WaveletBasis{T}) = T(1)

BasisFunctions.support(b::WaveletBasis, idx) = support(b, native_index(b, idx))

function BasisFunctions.support(b::WaveletBasis{T,S}, idxn::WaveletIndex) where {T,S}
    l,r = support(side(b), kind(idxn), wavelet(b), level(idxn), offset(idxn))
    (r-l > 1) && (return support(b))
    (l < 0) && (return union(interval(T(0),T(r)), interval(T(l+1), T(1))))
    (r > 1) && (return union(interval(T(0),T(r-1)), interval(T(l), T(1))))
    interval(T(l), T(r))
end

BasisFunctions.left(b::WaveletBasis{T}, i::WaveletIndex) where {T} = T(0)
BasisFunctions.right(b::WaveletBasis{T}, i::WaveletIndex) where {T} = T(1)

function first_index(b::WaveletBasis, x::Real)
    ii, on_edge = BasisFunctions.interval_index(b, x)
    s = support(BasisFunctions.side(b), BasisFunctions.kind(b), wavelet(b))
    s1 = Int(s[1])
    L = Int(s[2]) - s1
    if L == 1
        return mod(ii-s1-1,length(b))+1, 1
    end
    if on_edge
        return mod(ii-s1-2,length(b))+1, L - 1
    else
        return mod(ii-s1-1,length(b))+1, L
    end
end
_element_spans_one(b::WaveletBasis) = support_length(side(b), kind(b), wavelet(b)) == 1

period{T}(::WaveletBasis{T}) = T(1)

grid{T}(b::WaveletBasis{T}) = DyadicPeriodicEquispacedGrid(dyadic_length(b), support(b), T)


ordering(b::WaveletBasis{T,S,Wvl}) where {T,S<:Side} = wavelet_indices(dyadic_length(b))
ordering(b::WaveletBasis{T,S,Scl}) where {T,S<:Side} = scaling_indices(dyadic_length(b))


BasisFunctions.native_index(b::WaveletBasis{T,S,Wvl}, idx::Int) where {T,S<:Side} =
    wavelet_index(dyadic_length(b), idx)
BasisFunctions.linear_index(b::WaveletBasis{T,S,Wvl}, idxn::WaveletIndex) where {T,S<:Side} =
    value(idxn)
BasisFunctions.native_index(b::WaveletBasis{T,S,Scl}, idx::Int) where {T,S<:Side} =
    scaling_index(dyadic_length(b), idx)
BasisFunctions.linear_index(b::WaveletBasis{T,S,Scl}, idxn::WaveletIndex) where {T,S<:Side} =
    scaling_value(idxn)

support_length_of_compact_function(s::WaveletBasis{T,S,Scl}) where {T,S,Scl} = T(support_length(side(s), kind(s), wavelet(s)))/T(length(s))

approximate_native_size(::WaveletBasis, size_l) = 1<<ceil(Int, log2(size_l))

approx_length(::WaveletBasis, n) = 1<<round(Int, log2(size_l))

extension_size(b::WaveletBasis) = 2*length(b)


function unsafe_eval_element(dict::WaveletBasis{T,S}, idxn::WaveletIndex, x; xtol=1e-4, options...) where {T,S}
    evaluate_periodic(S(), kind(idxn), wavelet(dict), level(idxn), offset(idxn), x; xtol = xtol, options...)
end

unsafe_eval_element1(dict::WaveletBasis, idxn::WaveletIndex, grid::DyadicPeriodicEquispacedGrid; options...) =
    _unsafe_eval_element_in_dyadic_grid(dict, idxn, grid; options...)

function unsafe_eval_element1(dict::WaveletBasis, idxn::WaveletIndex, grid::PeriodicEquispacedGrid; options...)
    if isdyadic(length(grid)) && has_grid_equal_span(dict,grid)
        _unsafe_eval_element_in_dyadic_grid(dict, idxn, grid; options...)
    else
        _default_unsafe_eval_element_in_grid(dict, idxn, grid)
    end
end

_unsafe_eval_element_in_dyadic_grid(dict::WaveletBasis{T,S}, idxn::WaveletIndex, grid::AbstractGrid; options...) where {T,S} =
    evaluate_periodic_in_dyadic_points(S(), kind(idxn), wavelet(dict), level(idxn), offset(idxn), round(Int,log2(length(grid))))


function BasisFunctions.transform_from_grid(src, dest::WaveletBasis{T,S,Scl}, grid; options...) where {T,S}
    @assert compatible_grid(dest, grid)
    scaling_transfrom_from_grid(src, dest, grid; options...)
end

function BasisFunctions.transform_from_grid(src, dest::WaveletBasis{T,S,Wvl}, grid; options...) where {T,S}
    @assert compatible_grid(dest, grid)
    DiscreteWaveletTransform(dest)*scaling_transfrom_from_grid(src, dest, grid; options...)
end

function scaling_transfrom_from_grid(src, dest::WaveletBasis, grid; options...)
    j = dyadic_length(dest)
    l = dyadic_length(grid)
    dyadic_os = l-j
    _weight_operator(dest, dyadic_os)
end


function _weight_operator(dest::WaveletBasis, dyadic_os)
    j = dyadic_length(dest)
    wav = wavelet(dest)
    if dyadic_os == 0 # Just a choice I made.
        return WeightOperator(wav, 1, j, 0)
    else
        return WeightOperator(wav, 2, j, dyadic_os-1)
    end
end

function BasisFunctions.transform_to_grid(src::WaveletBasis, dest, grid; options...)
    @assert compatible_grid(src, grid)
    EvalOperator(src, dest, dyadic_length(grid); options...)
end

"""
Transformation of wavelet coefficients to function values.
"""
struct DWTEvalOperator{T} <: DictionaryOperator{T}
    src::WaveletBasis
    dest::GridBasis

    s::Side
    w::DiscreteWavelet
    fb::Filterbank
    j::Int
    d::Int
    bnd::WaveletBoundary

    f::Vector{T}
    f_scaled::Vector{T}
    coefscopy::Vector{T}
    coefscopy2::Vector{T}
end

function EvalOperator(dict::WaveletBasis{T,S,Wvl}, dgs::GridBasis, d::Int; options...) where {T,S}
    w = wavelet(dict)
    j = dyadic_length(dict)
    s = side(dict)
    fb = SFilterBank(s, w)

    # DWT.evaluate_in_dyadic_points!(f, s, scaling, w, j, 0, d, scratch)
    f = evaluate_in_dyadic_points(s, scaling, w, j, 0, d)
    f_scaled = similar(f)
    coefscopy = zeros(1<<j)
    coefscopy2 = similar(coefscopy)

    DWTEvalOperator{T}(dict, dgs, s, w, fb, j, d, perbound, f, f_scaled, coefscopy, coefscopy2)
end

function apply!(op::DWTEvalOperator, y, coefs; options...)
    idwt!(op.coefscopy, coefs, op.fb, op.bnd, op.j, op.coefscopy2)
    _evaluate_periodic_scaling_basis_in_dyadic_points!(y, op.f, op.s, op.w, op.coefscopy, op.j, op.d, op.f_scaled)
    y
end

# """
# Transformation of scaling coefficients to function values.
# """
# struct DWTScalingEvalOperator{T} <: DictionaryOperator{T}
#     src::WaveletBasis
#     dest::GridBasis
#
#     s::Side
#     w::DiscreteWavelet
#     fb::Filterbank
#     j::Int
#     d::Int
#     bnd::WaveletBoundary
#
#     f::Vector{T}
#     f_scaled::Vector{T}
# end
#
# function EvalOperator(dict::WaveletBasis{T,S,Scl}, dgs::GridBasis, d::Int; options...) where {T,S}
#     w = wavelet(dict)
#     j = dyadic_length(dict)
#     s = side(dict)
#     fb = SFilterBank(s, w)
#
#     f = evaluate_in_dyadic_points(s, scaling, w, j, 0, d)
#     f_scaled = similar(f)
#
#     DWTScalingEvalOperator{T}(dict, dgs, s, w, fb, j, d, perbound, f, f_scaled)
# end
# function apply!(op::DWTScalingEvalOperator, y, coefs; options...)
#     _evaluate_periodic_scaling_basis_in_dyadic_points!(y, op.f, op.s, op.w, coefs, op.j, op.d, op.f_scaled)
#     y
# end
function EvalOperator(dict::WaveletBasis{T,S,Scl}, dgs::GridBasis, d::Int; options...) where {T,S}
    w = wavelet(dict)
    j = BasisFunctions.dyadic_length(dict)
    s = BasisFunctions.side(dict)
    coefs = zeros(dict)
    coefs[1] = 1
    y = evaluate_periodic_scaling_basis_in_dyadic_points(s, w, coefs, d)
    a, offset = _get_array_offset(y)
    BasisFunctions.VerticalBandedOperator(dict, dgs, a, 1<<(d-j), offset-1)
end

function _get_array_offset(a)
    b = a.!=0
    f = findfirst(b)
    if f==1
        if b[end]
            f = findlast(.!b)+1
            L = sum(b)
            vcat(a[f:end],a[1:L-length(a)+f]), f
        else
            a[f:f+sum(b)-1], f
        end
    else
        a[f:f+sum(b)-1], f
    end
end

struct DiscreteWaveletTransform{T} <: DictionaryOperator{T}
    src::WaveletBasis{T,S,Scl} where S
    dest::WaveletBasis{T,S,Wvl} where S

    fb::Filterbank
    j::Int

    scratch::Vector{T}
end

DiscreteWaveletTransform(dict::WaveletBasis{T,S,Scl}) where{T,S} =
    DiscreteWaveletTransform(dict, WaveletBasis(dict))

DiscreteWaveletTransform(dict::WaveletBasis{T,S,Wvl}) where{T,S} =
    DiscreteWaveletTransform(ScalingBasis(dict), dict)

function DiscreteWaveletTransform(src::WaveletBasis{T,S,Scl}, dest::WaveletBasis{T,S,Wvl}) where {T,S}
    w = wavelet(src)
    j = dyadic_length(src)
    s = side(src)
    fb = SFilterBank(s, w)
    scratch = zeros(T, 1<<j)
    DiscreteWaveletTransform{T}(src, dest, fb, j, scratch)
end

apply!(op::DiscreteWaveletTransform, dest, src; options...) =
    dwt!(dest, src, op.fb, perbound, op.j, op.scratch)

grid_evaluation_operator(s::WaveletBasis, dgs::GridBasis, grid::AbstractGrid; options...) =
    default_evaluation_operator(s, dgs; options...)

grid_evaluation_operator(s::WaveletBasis, dgs::GridBasis, grid::DyadicPeriodicEquispacedGrid; options...) =
    EvalOperator(s, dgs, dyadic_length(grid); options...)

function grid_evaluation_operator(s::WaveletBasis, dgs::GridBasis, subgrid::AbstractSubGrid; options...)
    # We make no attempt if the set has no associated grid
    if has_grid(s)
        # Is the associated grid of the same type as the supergrid at hand?
        if typeof(grid(s)) == typeof(supergrid(subgrid))
            # It is: we can use the evaluation operator of the supergrid
            super_dgs = gridbasis(s, supergrid(subgrid))
            E = evaluation_operator(s, super_dgs; options...)
            R = restriction_operator(super_dgs, dgs; options...)
            R*E
        else
            default_evaluation_operator(s, dgs; options...)
        end
    else
        default_evaluation_operator(s, dgs; options...)
    end
end

function evaluation_matrix!(a::AbstractMatrix, dict::WaveletBasis, pts::DyadicPeriodicEquispacedGrid)
    @assert size(a,1) == length(pts)
    @assert size(a,2) == length(dict)

    s = side(dict)
    w = wavelet(dict)
    d = dyadic_length(pts)

    f = zeros(length(pts))
    SS = EvalPeriodicScratchSpace(s, w, dyadic_length(dict), d)
    for index in ordering(dict)
        evaluate_periodic_in_dyadic_points!(f, s, kind(index), w, level(index), offset(index), d, SS)
        for i in 1:length(f)
            a[i,value(index)] = f[i]
        end
    end
    a
end

Gram(b::WaveletBasis{T,S,Wvl}) where {S,T} = IdentityOperator(b, b)

abstract type BiorthogonalWaveletBasis{T,S,K} <: WaveletBasis{T,S,K} end

is_basis(b::BiorthogonalWaveletBasis) = true

abstract type OrthogonalWaveletBasis{T,S,K} <: BiorthogonalWaveletBasis{T,S,K} end

is_basis(b::OrthogonalWaveletBasis) = true
is_orthogonal(b::OrthogonalWaveletBasis) = true

wavelet_dual(w::OrthogonalWaveletBasis) = w


struct DaubechiesWaveletBasis{P,T,S,K} <: OrthogonalWaveletBasis{T,S,K}
    w   ::    DaubechiesWavelet{P,T}
    L   ::    Int
end

ScalingBasis(w::DaubechiesWavelet{P,T}, L::Int, ::Type{S}=Prl) where {P,T,S} =
    DaubechiesScalingBasis(P, L, T)

ScalingBasis(dict::DaubechiesWaveletBasis{P,T,S,K}) where {P,T,S,K} =
    DaubechiesWaveletBasis{P,T,S,Scl}(dict.w, dict.L)

WaveletBasis(dict::DaubechiesWaveletBasis{P,T,S,K}) where {P,T,S,K} =
    DaubechiesWaveletBasis{P,T,S,Wvl}(dict.w, dict.L)

DaubechiesWaveletBasis(P::Int, L::Int, ::Type{T} = Float64) where {T} =
    DaubechiesWaveletBasis{P,T,Prl,Wvl}(DaubechiesWavelet{P,T}(), L)

DaubechiesScalingBasis(P::Int, L::Int, ::Type{T} = Float64) where {T} =
    DaubechiesWaveletBasis{P,T,Prl,Scl}(DaubechiesWavelet{P,T}(), L)

dict_promote_domaintype(b::DaubechiesWaveletBasis{P,T,SIDE,KIND}, ::Type{S}) where {P,T,S,SIDE,KIND} =
    DaubechiesWaveletBasis{P,promote_type(T,S),SIDE,KIND}(DaubechiesWavelet{P,promote_type(T,S)}(), dyadic_length(b))

instantiate{T}(::Type{DaubechiesWaveletBasis}, n, ::Type{T}) = DaubechiesWaveletBasis(3, approx_length(n), T)

is_compatible{P,T1,T2,S1,S2,K1,K2}(src1::DaubechiesWaveletBasis{P,T1,S1,K1}, src2::DaubechiesWaveletBasis{P,T2,S2,K2}) = true

# Note no check on existence of CDFXY is present.
struct CDFWaveletBasis{P,Q,T,S,K} <: BiorthogonalWaveletBasis{T,S,K}
    w   ::    CDFWavelet{P,Q,T}
    L   ::    Int
end

ScalingBasis(w::CDFWavelet{P,Q,T}, L::Int, ::Type{S}=Prl) where {P,Q,T,S} =
    CDFScalingBasis(P, Q, L, S, T)

CDFWaveletBasis(P::Int, Q::Int, L::Int, ::Type{S}=Prl, ::Type{T} = Float64) where {T,S<:Side} =
    CDFWaveletBasis{P,Q,T,S,Wvl}(CDFWavelet{P,Q,T}(),L)

CDFScalingBasis(P::Int, Q::Int, L::Int, ::Type{S}=Prl, ::Type{T} = Float64) where {T,S<:Side} =
    CDFWaveletBasis{P,Q,T,S,Scl}(CDFWavelet{P,Q,T}(),L)

ScalingBasis(dict::CDFWaveletBasis{P,Q,T,S,K}) where {P,Q,T,S,K} =
    CDFWaveletBasis{P,Q,T,S,Scl}(dict.w, dict.L)

WaveletBasis(dict::CDFWaveletBasis{P,Q,T,S,K}) where {P,Q,T,S,K} =
    CDFWaveletBasis{P,Q,T,S,Wvl}(dict.w, dict.L)

dict_promote_domaintype(b::CDFWaveletBasis{P,Q,T,SIDE,KIND}, ::Type{S}) where {P,Q,T,S,SIDE,KIND} =
    CDFWaveletBasis{P,Q,promote_type(T,S),SIDE,KIND}(CDFWavelet{P,Q,promote_type(T,S)}(), dyadic_length(b))

instantiate{T}(::Type{CDFWaveletBasis}, n, ::Type{T}) = CDFWaveletBasis(2, 4, approx_length(n), T)

is_compatible{P,Q,T1,T2,S1,S2,K1,K2}(src1::CDFWaveletBasis{P,Q,T1,S1,K1}, src2::CDFWaveletBasis{P,Q,T2,S2,K2}) = true

wavelet_dual(w::CDFWaveletBasis{P,Q,T,S,K}) where {P,Q,T,S,K} =
    CDFWaveletBasis{P,Q,T,inv(S),K}(CDFWavelet{P,Q,T}(),dyadic_length(w))

@recipe function f(F::WaveletBasis; plot_complex = false, n=200)
    legend --> false
    grid = plotgrid(F,n)
    for i in eachindex(F)
        @series begin
            vals = F[i](grid)
            grid, postprocess(F[i],grid,vals)
        end
    end
    nothing
end

plotgrid(b::WaveletBasis, n) = DyadicPeriodicEquispacedGrid(round(Int,log2(n)), support(b))

"""
A `DWTSamplingOperator` is an operator that maps a function to scaling coefficients.
"""
struct DWTSamplingOperator <: AbstractSamplingOperator
    sampler :: GridSamplingOperator
    weight  :: DictionaryOperator
    scratch :: Array

	# An inner constructor to enforce that the operators match
	function DWTSamplingOperator(sampler::GridSamplingOperator, weight::DictionaryOperator{ELT}) where {ELT}
        @assert real(ELT) == eltype(eltype(grid(sampler)))
        @assert size(weight, 2) == length(grid(sampler))
		new(sampler, weight, zeros(src(weight)))
    end
end
using WaveletsCopy.DWT: quad_sf_weights
Base.convert(::Type{OP}, dwt::DWTSamplingOperator) where {OP<:DictionaryOperator} = dwt.weight
Base.promote_rule(::Type{OP}, ::DWTSamplingOperator) where{OP<:DictionaryOperator} = OP

"""
A `WeightOperator` is an operator that maps function values to scaling coefficients.
"""
function WeightOperator(basis::WaveletBasis, oversampling::Int=1, recursion::Int=0)
    wav = wavelet(basis)
    @assert coeftype(basis) == eltype(wav)
    WeightOperator(wav, oversampling, dyadic_length(basis), recursion)
end

function WeightOperator(wav::DiscreteWavelet{T}, oversampling::Int, j::Int, d::Int) where {T}
    @assert isdyadic.(oversampling)
    w = quad_sf_weights(Dual, scaling, wav, oversampling*support_length(Dual, scaling, wav), d)
    # rescaling of weights because the previous step assumed sum(w)==1
    w .= w ./ sqrt(T(1<<j))
    src_size = 1<<(d+j+oversampling>>1)
    step = 1<<(d+oversampling>>1)
    os = mod(step*Sequences.offset(filter(Dual, scaling, wav))-1, src_size)+1
    try
        # HorizontalBandedOperator(DiscreteVectorDictionary(src_size), DiscreteVectorDictionary(1<<j), w, step, os)
        HorizontalBandedOperator(gridbasis(DyadicPeriodicEquispacedGrid(d+j+oversampling>>1), T), ScalingBasis(wav, j) , w, step, os)
    catch y
        if isa(y, AssertionError)
            error("Support of dual wavelet basis exceeds the width of the domain. Try more elements in your basis.")
        else
            rethrow(y)
        end
    end
end

function DWTSamplingOperator(dict::Dictionary, oversampling::Int=1, recursion::Int=0)
    weight = WeightOperator(dict, oversampling, recursion)
    sampler = GridSamplingOperator(gridbasis(dwt_oversampled_grid(dict, oversampling, recursion)))
    DWTSamplingOperator(sampler, weight)
end

dwt_oversampled_grid(dict::Dictionary, oversampling::Int, recursion::Int) =
    oversampled_grid(dict, 1<<(recursion+oversampling>>1))

src_space(op::DWTSamplingOperator) = src_space(op.sampler)
dest(op::DWTSamplingOperator) = dest(op.weight)

apply(op::DWTSamplingOperator, f) = apply!(zeros(dest(op)), op, f)
apply!(result, op::DWTSamplingOperator, f) = apply!(op.weight, result, apply!(op.scratch, op.sampler, f))

##################
# Platform
##################

# 1D generators
primal_scaling_generator(wavelet::DiscreteWavelet) = n->ScalingBasis(wavelet,n)
dual_scaling_generator(wavelet::DiscreteWavelet) = n->ScalingBasis(wavelet,n, Dul)

# ND generators
primal_scaling_generator(wav1::DiscreteWavelet, wav2::DiscreteWavelet, wav::DiscreteWavelet...) = primal_scaling_generator([wav1, wav2, wav...])

primal_scaling_generator(wav::AbstractVector{T}) where {T<:DiscreteWavelet} = tensor_generator(promote_eltype(map(eltype, wav)...), map(w->primal_scaling_generator(w), wav)...)

dual_scaling_generator(wav1::DiscreteWavelet, wav2::DiscreteWavelet, wav::DiscreteWavelet...) = dual_scaling_generator([wav1, wav2, wav...])

dual_scaling_generator(wav::AbstractVector{T}) where {T<:DiscreteWavelet} = tensor_generator(promote_eltype(map(eltype, wav)...), map(w->dual_scaling_generator(w), wav)...)
# Sampler
scaling_sampler(primal, oversampling::Int) =
    n->(
        basis = primal(n+Int(log2(oversampling)));
        GridSamplingOperator(gridbasis(grid(basis), coeftype(basis)));
    )

dual_scaling_sampler(primal, oversampling) =
    n -> (
        dyadic_os = Int(log2(oversampling));
        basis = primal(n+dyadic_os);
        W = _weight_operator(primal(n), dyadic_os);
        sampler = GridSamplingOperator(gridbasis(grid(basis)));
        DWTSamplingOperator(sampler, W);
    )

# params
scaling_param(init::Int) = SteppingSequence(init)

scaling_param(init::AbstractVector{Int}) = TensorSequence([SteppingSequence(i) for i in init])

# Platform
function scaling_platform(init::Union{Int,AbstractVector{Int}}, wav::Union{W,AbstractVector{W}}, oversampling::Int) where {W<:DiscreteWavelet}
    @assert isdyadic(oversampling)
	primal = primal_scaling_generator(wav)
	dual = dual_scaling_generator(wav)
	sampler = scaling_sampler(primal, oversampling)
    # dual_sampler = dual_scaling_sampler(wav, oversampling)
    dual_sampler = dual_scaling_sampler(primal, oversampling)
	params = scaling_param(init)
	GenericPlatform(primal = primal, dual = dual, sampler = sampler, dual_sampler=dual_sampler,
		params = params, name = "Scaling functions")
end

Zt(dual::WaveletBasis, dual_sampler::DWTSamplingOperator; options...) = DictionaryOperator(dual_sampler)

##################
# Tensor methods
##################
const WaveletTensorDict2d = TensorProductDict{2,Tuple{B1,B2}} where {B1<:WaveletBasis,B2<:WaveletBasis}
const WaveletTensorDict3d = TensorProductDict{3,Tuple{B1,B2,B3}} where {B1<:WaveletBasis,B2<:WaveletBasis,B3<:WaveletBasis}
const WaveletTensorDict4d = TensorProductDict{4,Tuple{B1,B2,B3,B4}} where {B1<:WaveletBasis,B2<:WaveletBasis,B3<:WaveletBasis,B4<:WaveletBasis}
const WaveletTensorDict = Union{TensorProductDict{N,NTuple{N,B}} where {N,B<:WaveletBasis},WaveletTensorDict2d,WaveletTensorDict3d,WaveletTensorDict4d}

wavelet_dual(dict::WaveletTensorDict) =
    tensorproduct([wavelet_dual(d) for d in elements(dict)])
WeightOperator(dict::WaveletTensorDict, oversampling::Vector{Int}, recursion::Vector{Int}) =
    TensorProductOperator([WeightOperator(di, osi, reci) for (di, osi, reci) in zip(elements(dict), oversampling, recursion)]...)
_weight_operator(dict::WaveletTensorDict, dyadic_os) =
    TensorProductOperator([_weight_operator(di, dyadic_os) for di in elements(dict)]...)
BasisFunctions.grid_evaluation_operator(s::BasisFunctions.WaveletTensorDict, dgs::GridBasis, grid::AbstractSubGrid; options...) =
    restriction_operator(gridbasis(supergrid(grid), coeftype(s)), gridbasis(grid, coeftype(s)))*grid_evaluation_operator(s, gridbasis(supergrid(grid), coeftype(s)), supergrid(grid))
BasisFunctions.grid_evaluation_operator(s::BasisFunctions.WaveletTensorDict, dgs::GridBasis, grid::ProductGrid; options...) =
    TensorProductOperator([grid_evaluation_operator(dict, gridbasis(g, coeftype(s)), g) for (dict, g) in zip(elements(s), elements(grid))]...)
BasisFunctions.Zt(dual::WaveletTensorDict, dual_sampler::DWTSamplingOperator; options...) = DictionaryOperator(dual_sampler)





wavelet_dual(dict::TensorProductDict) =
    tensorproduct([wavelet_dual(d) for d in elements(dict)])
