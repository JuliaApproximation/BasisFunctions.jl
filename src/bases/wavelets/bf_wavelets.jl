# bf_wavelets.jl

abstract type WaveletBasis{T,S,K} <: Dictionary1d{T,T} where {S <: Side,K<:Kind}
end

const WaveletSpan{A,S,T,D <: WaveletBasis} = Span{A,S,T,D}

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

has_extension(b::WaveletBasis) = true

approx_length(b::WaveletBasis, n::Int) = 1<<ceil(Int, log2(n))

resize(b::B, n::Int) where {B<:WaveletBasis} = B(wavelet(b),round(Int, log2(n)))

has_grid(::WaveletBasis) = true

has_transform(::WaveletBasis) = true

compatible_grid(set::WaveletBasis, grid::PeriodicEquispacedGrid) =
    has_grid_equal_span(set,grid) && (length(set)==length(grid))
compatible_grid(set::WaveletBasis, grid::DyadicPeriodicEquispacedGrid) =
	has_grid_equal_span(set,grid) && (length(set)==length(grid))
compatible_grid(set::WaveletBasis, grid::AbstractGrid) = false
has_grid_transform(b::WaveletBasis, gb, grid) = compatible_grid(b, grid)

support(b::WaveletBasis) = UnitInterval{domaintype(b)}()
left{T}(::WaveletBasis{T}) = T(0)
right{T}(::WaveletBasis{T}) = T(1)

BasisFunctions.support(b::WaveletBasis, idx) = BasisFunctions.support(b, native_index(b, idx))

function BasisFunctions.support(b::WaveletBasis{T,S}, idxn::WaveletIndex) where {T,S}
    l,r = support(S(), kind(idxn), wavelet(b), level(idxn), offset(idxn))
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


# TODO remove scaling if they disapear in Fourier
function transform_from_grid(src, dest::WaveletBasis, grid; options...)
    @assert compatible_grid(dest, grid)
    L = length(src)
    ELT = coeftype(src)
    S = ScalingOperator(dest, sqrt(ELT(L)))
    warn("Conversion from function values to scaling coefficients is approximate")
    T = FullDiscreteWaveletTransform(src, dest, wavelet(dest), side(dest); options...)
    T*S
end

function transform_to_grid(src::WaveletBasis, dest, grid; options...)
    @assert compatible_grid(src, grid)
    L = length(src)
    ELT = coeftype(src)
    S = ScalingOperator(dest, 1/sqrt(ELT(L)))
    T = FullInverseDistreteWaveletTransform(src, dest, wavelet(src), side(src); options...)
    T*S
end

function unitary_dwt(src, dest::WaveletBasis; options...)
    L = length(src)
    ELT = coeftype(src)
    S = ScalingOperator(dest, sqrt(ELT(L)))
    T = DiscreteWaveletTransform(src, dest, wavelet(dest), side(dest); options...)
    T*S
end

function unitary_idwt(src::WaveletBasis, dest; options...)
    L = length(src)
    ELT = coeftype(src)
    S = ScalingOperator(dest, 1/sqrt(ELT(L)))
    T = InverseDistreteWaveletTransform(src, dest, wavelet(src), side(src); options...)
    T*S
end

function transform_from_grid_post(src, dest::WaveletBasis, grid; options...)
	@assert compatible_grid(dest, grid)
    L = length(src)
    ELT = coeftype(src)
    ScalingOperator(dest, 1/sqrt(ELT(L)))
end

function transform_to_grid_pre(src::WaveletBasis, dest, grid; options...)
	@assert compatible_grid(src, grid)
	inv(transform_from_grid_post(dest, src, grid; options...))
end

dwtfunctions = [:full_dwt, :dwt, :dwt_dual]
idwtfuncions = [:full_idwt, :idwt, :idwt_dual]

dwtFunctions = [:FullDWTFunction, :DWTFunction, :DualDWTFunction]
idwtFunctions = [:FulliDWTFunction, :iDWTFunction, :DualiDWTFunction]

for (ffun, iffun, FFun, iFFun) in zip(dwtfunctions, idwtfuncions, dwtFunctions, idwtFunctions)
    for (f, F) in zip([ffun, iffun],[FFun, iFFun])
        @eval begin
            struct $F <: Function
                w::DiscreteWavelet
                s::Side
            end
            $F(w::DiscreteWavelet) = $F(w, Primal)
            (dwtf::$F)(x) = $f(x, dwtf.s, dwtf.w, perbound)
        end
    end
    @eval begin
        if $FFun == FullDWTFunction
            Base.inv(f::$iFFun) = (warn("Inverse of full_idwt is approximate since conversion of function samples to scaling coefficients is approximate"); $FFun(f.w))
            Base.inv(f::$FFun) = (warn("Inverse of full_dwt is approximate since conversion of function samples to scaling coefficients is approximate"); $iFFun(f.w))
        else
            Base.inv(f::$FFun) = $iFFun(f.w)
            Base.inv(f::$iFFun) = $FFun(f.w)
        end
    end
end

struct EvalDWTFunction <: Function
    w::DiscreteWavelet
    s::Side
    l::Int
end
(f::EvalDWTFunction)(c) = evaluate_periodic_in_dyadic_points(f.s, f.w, c, f.l)

struct invEvalDWTFunction <: Function
    w::DiscreteWavelet
    s::Side
    l::Int
end
(f::invEvalDWTFunction)(c) = inv_evaluate_periodic_in_dyadic_points(f.s, f.w, c, f.l)
# scaling is necessary
inv(f::EvalDWTFunction) = invEvalDWTFunction(f.w, f.s, f.l)
inv(f::invEvalDWTFunction) = EvalDWTFunction(f.w, f.s, f.l)

DiscreteWaveletTransform(src::Dictionary, dest::Dictionary, w::DiscreteWavelet, s::Side; options...) =
    FunctionOperator(src, dest, DWTFunction(w, s))

InverseDistreteWaveletTransform(src::Dictionary, dest::Dictionary, w::DiscreteWavelet, s::Side; options...) =
    FunctionOperator(src, dest, iDWTFunction(w, s))

FullDiscreteWaveletTransform(src::Dictionary, dest::Dictionary, w::DiscreteWavelet, s::Side; options...) =
    FunctionOperator(src, dest, FullDWTFunction(w, s))

FullInverseDistreteWaveletTransform(src::Dictionary, dest::Dictionary, w::DiscreteWavelet, s::Side; options...) =
    FunctionOperator(src, dest, FulliDWTFunction(w, s))

Base.ctranspose(op::FunctionOperator{F,T}) where {F<:BasisFunctions.DWTFunction,T} =
    InverseDistreteWaveletTransform(src(op), dest(op), op.fun.w, op.fun.s)*ScalingOperator(src(op),src(op),T(1)/length(src(op)))

Base.ctranspose(op::FunctionOperator{F,T}) where {F<:BasisFunctions.iDWTFunction,T} =
    DiscreteWaveletTransform(src(op), dest(op), op.fun.w, op.fun.s)*ScalingOperator(src(op),src(op),length(src(op)))

Base.ctranspose(op::FunctionOperator{F,T}) where {F<:BasisFunctions.FullDWTFunction,T} =
    FullInverseDistreteWaveletTransform(src(op), dest(op), op.fun.w, op.fun.s)*ScalingOperator(src(op),src(op),T(1)/length(src(op)))

Base.ctranspose(op::FunctionOperator{F,T}) where {F<:BasisFunctions.FulliDWTFunction,T} =
    FullDiscreteWaveletTransform(src(op), dest(op), op.fun.w, op.fun.s)*ScalingOperator(src(op),src(op),length(src(op)))

Base.inv(op::FunctionOperator{F,T}) where {F<:BasisFunctions.EvalDWTFunction,T} =
    ScalingOperator(dest(op), dest(op), T(1//(1<<op.fun.l)))*FunctionOperator(src(op), dest(op), inv(op.fun))

Base.inv(op::FunctionOperator{F,T}) where {F<:BasisFunctions.invEvalDWTFunction,T} =
    FunctionOperator(src(op), dest(op), inv(op.fun))*ScalingOperator(src(op), src(op), T(1//(1<<op.fun.l)))

grid_evaluation_operator(s::WaveletBasis, dgs::GridBasis, grid::AbstractGrid; options...) =
    default_evaluation_operator(s, dgs; options...)

grid_evaluation_operator(s::WaveletBasis, dgs::GridBasis, grid::DyadicPeriodicEquispacedGrid; options...) =
    grid_evaluation_operator(s, dgs, grid, kind(s); options...)

grid_evaluation_operator(s::WaveletBasis, dgs::GridBasis, grid::DyadicPeriodicEquispacedGrid, ::Wvl; options...) =
    DWTEvalOperator(s, dgs, dyadic_length(grid))

grid_evaluation_operator(s::WaveletBasis, dgs::GridBasis, grid::DyadicPeriodicEquispacedGrid, ::Scl; options...) =
    DWTScalingEvalOperator(s, dgs, dyadic_length(grid))

struct DWTEvalOperator{T} <: AbstractOperator{T}
    src::Dictionary
    dest::Dictionary

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

function BasisFunctions.DWTEvalOperator(span::BasisFunctions.WaveletBasis{T}, dgs::GridBasis, d::Int) where {T}
    w = BasisFunctions.wavelet(span)
    j = BasisFunctions.dyadic_length(span)
    s = BasisFunctions.side(span)
    fb = BasisFunctions.SFilterBank(s, w)

    # DWT.evaluate_in_dyadic_points!(f, s, scaling, w, j, 0, d, scratch)
    f = BasisFunctions.evaluate_in_dyadic_points(s, scaling, w, j, 0, d)
    f_scaled = similar(f)
    coefscopy = zeros(1<<j)
    coefscopy2 = similar(coefscopy)

    BasisFunctions.DWTEvalOperator{T}(span, dgs, s, w, fb, j, d, perbound, f, f_scaled, coefscopy, coefscopy2)
end

function BasisFunctions.apply!(op::BasisFunctions.DWTEvalOperator, y, coefs; options...)
    BasisFunctions.idwt!(op.coefscopy, coefs, op.fb, op.bnd, op.j, op.coefscopy2)
    BasisFunctions._evaluate_periodic_scaling_basis_in_dyadic_points!(y, op.f, op.s, op.w, op.coefscopy, op.j, op.d, op.f_scaled)
    y
end

struct DWTScalingEvalOperator{T} <: AbstractOperator{T}
    src::Dictionary
    dest::Dictionary

    s::Side
    w::DiscreteWavelet
    fb::Filterbank
    j::Int
    d::Int
    bnd::WaveletBoundary

    f::Vector{T}
    f_scaled::Vector{T}
end

function BasisFunctions.DWTScalingEvalOperator(span::BasisFunctions.WaveletBasis{T}, dgs::GridBasis, d::Int) where {T}
    w = BasisFunctions.wavelet(span)
    j = BasisFunctions.dyadic_length(span)
    s = BasisFunctions.side(span)
    fb = BasisFunctions.SFilterBank(s, w)

    f = BasisFunctions.evaluate_in_dyadic_points(s, scaling, w, j, 0, d)
    f_scaled = similar(f)

    BasisFunctions.DWTScalingEvalOperator{T}(span, dgs, s, w, fb, j, d, perbound, f, f_scaled)
end

function BasisFunctions.apply!(op::BasisFunctions.DWTScalingEvalOperator, y, coefs; options...)
    BasisFunctions._evaluate_periodic_scaling_basis_in_dyadic_points!(y, op.f, op.s, op.w, coefs, op.j, op.d, op.f_scaled)
    y
end


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
A `DWTSamplingOperator` is an operator that maps a function to wavelet coefficients.
"""
struct DWTSamplingOperator <: AbstractSamplingOperator
    sampler :: GridSamplingOperator
    weight  :: AbstractOperator
    scratch :: Vector

	# An inner constructor to enforce that the operators match
	function DWTSamplingOperator(sampler::GridSamplingOperator, weight::AbstractOperator{ELT}) where {ELT}
        @assert real(ELT) == eltype(eltype(grid(sampler)))
        @assert size(weight, 2) == length(grid(sampler))
		new(sampler, weight, zeros(src(weight)))
    end
end
using WaveletsCopy.DWT: quad_sf_weights
Base.convert(::Type{OP}, dwt::DWTSamplingOperator) where {OP<:AbstractOperator} = dwt.weight
Base.promote_rule(::Type{OP}, ::DWTSamplingOperator) where{OP<:AbstractOperator} = OP

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
    HorizontalBandedOperator(DiscreteVectorSet(src_size), DiscreteVectorSet(1<<j), w, step, os)
end

function DWTSamplingOperator(span::Dictionary, oversampling::Int=1, recursion::Int=0)
    weight = WeightOperator(span, oversampling, recursion)
    sampler = GridSamplingOperator(gridbasis(dwt_oversampled_grid(span, oversampling, recursion)))
    DWTSamplingOperator(sampler, weight)
end

dwt_oversampled_grid(dict::Dictionary, oversampling::Int, recursion::Int) =
    oversampled_grid(dict, 1<<(recursion+oversampling>>1))

src(op::DWTSamplingOperator) = src(op.sampler)
dest(op::DWTSamplingOperator) = dest(op.weight)

apply(op::DWTSamplingOperator, f) = op.weight*apply(op.sampler, f)
apply!(result, op::DWTSamplingOperator, f) = apply!(op.weight, result, sample!(op.scratch, op.sampler, f))

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
scaling_sampler(primal, oversampling::Int) = n-> GridSamplingOperator(gridbasis(grid(primal(n+Int(log2(oversampling))))))

dual_scaling_sampler(primal, oversampling) =
    n -> (
        basis = primal(n+Int(log2(oversampling)));
        wav = BasisFunctions.wavelet(basis);
        if oversampling==1; # Just a choice I made.
            W = BasisFunctions.WeightOperator(wav, 1, n, 0);
        elseif oversampling == 2 ;
            W = BasisFunctions.WeightOperator(wav, 2, n, 0);
        else ;
            W = BasisFunctions.WeightOperator(wav, 2, n, Int(log2(oversampling>>1))) ;
        end;
        sampler = GridSamplingOperator(gridbasis(grid(basis)));
        DWTSamplingOperator(sampler, W);
    )

# params
scaling_param(init::Int) = SteppingSequence(init)

scaling_param(init::AbstractVector{Int}) = TensorSequence([SteppingSequence(i) for i in init])

# Platform
function scaling_platform(init::Union{Int,AbstractVector{Int}}, wav::Union{W,AbstractVector{W}}, oversampling::Int) where {W<:DiscreteWavelet}
	primal = primal_scaling_generator(wav)
	dual = dual_scaling_generator(wav)
	sampler = scaling_sampler(primal, oversampling)
    # dual_sampler = dual_scaling_sampler(wav, oversampling)
    dual_sampler = dual_scaling_sampler(primal, oversampling)
	params = scaling_param(init)
	BasisFunctions.GenericPlatform(primal = primal, dual = dual, sampler = sampler, dual_sampler=dual_sampler,
		params = params, name = "Scaling functions")
end

Zt(dual::WaveletBasis, dual_sampler::DWTSamplingOperator; options...) = AbstractOperator(dual_sampler)
