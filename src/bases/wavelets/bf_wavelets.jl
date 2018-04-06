# bf_wavelets.jl

abstract type WaveletBasis{T} <: Dictionary1d{T,T}
end

const WaveletSpan{A,S,T,D <: WaveletBasis} = Span{A,S,T,D}

const WaveletIndex = Tuple{Kind,Int,Int}

checkbounds(::Type{Bool}, dict::WaveletBasis, i::WaveletIndex) =
    checkbounds(Bool, dict, linear_index(dict, i))

value(idxn::WaveletIndex) = coefficient_index(idxn...)

Base.show(io::IO, idxn::WaveletIndex) =
	print(io, "Wavelet index: $(value(idxn))")

kind(idxn::WaveletIndex) = idxn[1]
j(idxn::WaveletIndex) = idxn[2]
k(idxn::WaveletIndex) = idxn[3]


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
    has_grid_equal_span(set,grid) && (length(set)==length(grid))
compatible_grid(set::WaveletBasis, grid::DyadicPeriodicEquispacedGrid) =
	has_grid_equal_span(set,grid) && (length(set)==length(grid))
compatible_grid(set::WaveletBasis, grid::AbstractGrid) = false
has_grid_transform(b::WaveletBasis, gb, grid) = compatible_grid(b, grid)

left{T}(::WaveletBasis{T}) = T(0)
right{T}(::WaveletBasis{T}) = T(1)

function support(b::WaveletBasis{T}, idxn::WaveletIndex) where {T}
    l,r = support(Primal, kind(idxn), wavelet(b), j(idxn), k(idxn))
    l < 0 || r > 1 ? (T(0),T(1)) : (T(l),T(r))
end

left(b::WaveletBasis, i::WaveletIndex) = support(b,i)[1]
right(b::WaveletBasis, i::WaveletIndex) = support(b,i)[2]

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

Base.getindex(m::BasisFunctions.DWTIndexList, idx::Int) ::WaveletIndex = wavelet_index(length(m), idx, Int(log2(length(m))))

Base.getindex(list::BasisFunctions.DWTIndexList, idxn::WaveletIndex)::Int = value(idxn)

ordering(b::WaveletBasis) = DWTIndexList(length(b))


BasisFunctions.native_index(b::WaveletBasis, idx::Int)::WaveletIndex = wavelet_index(length(b), idx, dyadic_length(b))
BasisFunctions.linear_index(b::WaveletBasis, idxn::WaveletIndex)::Int = value(idxn)

approximate_native_size(::WaveletBasis, size_l) = 1<<ceil(Int, log2(size_l))

approx_length(::WaveletBasis, n) = 1<<round(Int, log2(size_l))

extension_size(b::WaveletBasis) = 2*length(b)


unsafe_eval_element(dict::WaveletBasis, idxn::WaveletIndex, x; xtol=1e-4, options...) =
    evaluate_periodic(Primal, kind(idxn), wavelet(dict), j(idxn), k(idxn), x; xtol = xtol, options...)

unsafe_eval_element1(dict::WaveletBasis, idxn::WaveletIndex, grid::DyadicPeriodicEquispacedGrid; options...) =
    _unsafe_eval_element_in_dyadic_grid(dict, idxn, grid; options...)

function unsafe_eval_element1(dict::WaveletBasis, idxn::WaveletIndex, grid::PeriodicEquispacedGrid; options...)
    if isdyadic(length(grid)) && has_grid_equal_span(dict,grid)
        _unsafe_eval_element_in_dyadic_grid(dict, idxn, grid; options...)
    else
        _default_unsafe_eval_element_in_grid(dict, idxn, grid)
    end
end

_unsafe_eval_element_in_dyadic_grid(dict::WaveletBasis, idxn::WaveletIndex, grid::AbstractGrid; options...) =
    evaluate_periodic_in_dyadic_points(Primal, kind(idxn), wavelet(dict), j(idxn), k(idxn), round(Int,log2(length(grid))))


# TODO remove scaling if they disapear in Fourier
function transform_from_grid(src, dest::WaveletSpan, grid; options...)
    @assert compatible_grid(dictionary(dest), grid)
    L = length(src)
    ELT = coeftype(src)
    S = ScalingOperator(dest, sqrt(ELT(L)))
    warn("Conversion from function values to scaling coefficients is approximate")
    T = FullDiscreteWaveletTransform(src, dest, wavelet(dictionary(dest)); options...)
    T*S
end

function transform_to_grid(src::WaveletSpan, dest, grid; options...)
    @assert compatible_grid(src, grid)
    L = length(src)
    ELT = coeftype(src)
    S = ScalingOperator(dest, 1/sqrt(ELT(L)))
    T = FullInverseDistreteWaveletTransform(src, dest, wavelet(dictionary(src)); options...)
    T*S
end

function unitary_dwt(src, dest::WaveletSpan; options...)
    L = length(src)
    ELT = coeftype(src)
    S = ScalingOperator(dest, sqrt(ELT(L)))
    T = DiscreteWaveletTransform(src, dest, wavelet(dictionary(dest)); options...)
    T*S
end

function unitary_idwt(src::WaveletSpan, dest; options...)
    L = length(src)
    ELT = coeftype(src)
    S = ScalingOperator(dest, 1/sqrt(ELT(L)))
    T = InverseDistreteWaveletTransform(src, dest, wavelet(dictionary(src)); options...)
    T*S
end

function transform_from_grid_post(src, dest::WaveletSpan, grid; options...)
	@assert compatible_grid(dictionary(dest), grid)
    L = length(src)
    ELT = coeftype(src)
    ScalingOperator(dest, 1/sqrt(ELT(L)))
end

function transform_to_grid_pre(src::WaveletSpan, dest, grid; options...)
	@assert compatible_grid(dictionary(src), grid)
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
            end
            (dwtf::$F)(x) = $f(x, dwtf.w, perbound)
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

DiscreteWaveletTransform(src::Span, dest::Span, w::DiscreteWavelet; options...) =
    FunctionOperator(src, dest, DWTFunction(w))

InverseDistreteWaveletTransform(src::Span, dest::Span, w::DiscreteWavelet; options...) =
    FunctionOperator(src, dest, iDWTFunction(w))

FullDiscreteWaveletTransform(src::Span, dest::Span, w::DiscreteWavelet; options...) =
    FunctionOperator(src, dest, FullDWTFunction(w))

FullInverseDistreteWaveletTransform(src::Span, dest::Span, w::DiscreteWavelet; options...) =
    FunctionOperator(src, dest, FulliDWTFunction(w))

Base.ctranspose(op::FunctionOperator{F,T}) where {F<:BasisFunctions.DWTFunction,T} =
    InverseDistreteWaveletTransform(src(op), dest(op), op.fun.w)*ScalingOperator(src(op),src(op),T(1)/length(src(op)))

Base.ctranspose(op::FunctionOperator{F,T}) where {F<:BasisFunctions.iDWTFunction,T} =
    DiscreteWaveletTransform(src(op), dest(op), op.fun.w)*ScalingOperator(src(op),src(op),length(src(op)))

Base.ctranspose(op::FunctionOperator{F,T}) where {F<:BasisFunctions.FullDWTFunction,T} =
    FullInverseDistreteWaveletTransform(src(op), dest(op), op.fun.w)*ScalingOperator(src(op),src(op),T(1)/length(src(op)))

Base.ctranspose(op::FunctionOperator{F,T}) where {F<:BasisFunctions.FulliDWTFunction,T} =
    FullDiscreteWaveletTransform(src(op), dest(op), op.fun.w)*ScalingOperator(src(op),src(op),length(src(op)))

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

dict_promote_domaintype{P,T,S}(b::DaubechiesWaveletBasis{P,T}, ::Type{S}) =
    DaubechiesWaveletBasis(DaubechiesWavelet{P,promote_type(T,S)}(), dyadic_length(b))

instantiate{T}(::Type{DaubechiesWaveletBasis}, n, ::Type{T}) = DaubechiesWaveletBasis(3, approx_length(n), T)

is_compatible{P,T1,T2}(src1::DaubechiesWaveletBasis{P,T1}, src2::DaubechiesWaveletBasis{P,T2}) = true

# Note no check on existence of CDFXY is present.
struct CDFWaveletBasis{P,Q,T} <: BiorthogonalWaveletBasis{T}
    w   ::    CDFWavelet{P,Q,T}
    L   ::    Int
end

CDFWaveletBasis{T}(P::Int, Q::Int, L::Int, ::Type{T} = Float64) =
    CDFWaveletBasis{P,Q,T}(CDFWavelet{P,Q,T}(),L)

dict_promote_domaintype{P,Q,T,S}(b::CDFWaveletBasis{P,Q,T}, ::Type{S}) =
    CDFWaveletBasis(CDFWavelet{P,Q,promote_type(T,S)}(), dyadic_length(b))

instantiate{T}(::Type{CDFWaveletBasis}, n, ::Type{T}) = CDFWaveletBasis(2, 4, approx_length(n), T)

is_compatible{P,Q,T1,T2}(src1::CDFWaveletBasis{P,Q,T1}, src2::CDFWaveletBasis{P,Q,T2}) = true

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


plotgrid(b::WaveletBasis, n) = DyadicPeriodicEquispacedGrid(round(Int,log2(n)), left(b), right(b))
