# translation_dict.jl
using CardinalBSplines
"""
Dictionary consisting of translates of one generating function.
"""
abstract type TranslationDict{T} <: Dictionary{T,T}
end

const TranslatesSpan{A,S,T,D <: TranslationDict} = Span{A,S,T,D}

length(set::TranslationDict) = set.n

is_biorthogonal(::TranslationDict) = true

is_basis(::TranslationDict) = true

name(b::TranslationDict) = "Set of translates of a function"
name(::Type{B}) where {B<:TranslationDict}= "Set of translates of a function"

fun(b::TranslationDict) = b.fun
fun(s::TranslatesSpan) = fun(dictionary(s))

# Indices of translates naturally range from 0 to n-1
const TransIndex = ShiftedIndex{1}
ordering(b::TranslationDict) = ShiftedIndexList{1}(length(b))

has_unitary_transform(::TranslationDict) = false


"""
  Set consisting of n equispaced translates of a periodic function.

  The set can be written as ``\left\{T_k f\right\}_{k=0}^n``, where ``T_k f(x) = f(x-p/n)``.
  ``p`` is the period of the set, ``n`` is the number of elements.
"""
abstract type PeriodicTranslationDict{T} <: TranslationDict{T}
end

const PeriodicTranslatesSpan{A,S,T,D <: PeriodicTranslationDict} = Span{A,S,T,D}

left{T}(set::PeriodicTranslationDict{T})::real(T) = real(T)(set.a)
right{T}(set::PeriodicTranslationDict{T})::real(T) = real(T)(set.b)
domain(set::PeriodicTranslationDict{T}) where {T} = interval(set.a,set.b)

left(set::PeriodicTranslationDict, j::TransIndex) = left(set)
right(set::PeriodicTranslationDict, j::TransIndex) = right(set)

has_grid(::PeriodicTranslationDict) = true

grid(set::PeriodicTranslationDict) = PeriodicEquispacedGrid(length(set), left(set), right(set))

period(set::PeriodicTranslationDict) = right(set)-left(set)

stepsize(set::PeriodicTranslationDict) = period(set)/length(set)

has_grid_transform(b::PeriodicTranslationDict, gb, grid::AbstractEquispacedGrid) =
    compatible_grid(b, grid)

compatible_grid(b::PeriodicTranslationDict, grid::AbstractEquispacedGrid) =
    periodic_compatible_grid(b, grid)

approx_length(b::PeriodicTranslationDict, n::Int) = ceil(Int,n/length(b))*length(b)

function periodic_compatible_grid(b::Dictionary, grid::AbstractEquispacedGrid)
    l1 = length(b)
    l2 = length(grid)
    l1 > l2 && ((l2,l1) = (l1, l2))
    n = l2/l1
    nInt = round(Int, n)
    (1+(left(b) - leftendpoint(grid))≈1) && (1+(right(b) - rightendpoint(grid))≈1) && isdyadic(nInt) && (n≈nInt)
end

native_nodes(b::PeriodicTranslationDict) = [k*stepsize(b) for k in 0:length(b)]

function transform_from_grid(src, dest::PeriodicTranslatesSpan, grid; options...)
    inv(transform_to_grid(dest, src, grid; options...))
end

function transform_to_grid(src::PeriodicTranslatesSpan, dest, grid; options...)
    @assert compatible_grid(dictionary(src), grid)
    CirculantOperator(src, dest, sample(grid, fun(src)); options...)
end


function grid_evaluation_operator(s::PeriodicTranslatesSpan, dgs::DiscreteGridSpace, grid::AbstractEquispacedGrid; sparse=true, options...)
    r = nothing
    if periodic_compatible_grid(dictionary(s), grid)
        lg = length(grid)
        ls = length(s)
        if lg == ls
            r = CirculantOperator(s, dgs, sample(grid, fun(s)); options...)
        elseif lg > ls
            r = CirculantOperator(dgs, dgs, sample(grid, fun(s)); options...)*IndexExtensionOperator(s, dgs, 1:Int(lg/ls):length(dgs))
        elseif lg < ls && has_extension(grid)
            r = IndexRestrictionOperator(s, dgs, 1:Int(ls/lg):length(s))*CirculantOperator(s, s, sample(extend(grid, Int(ls/lg)), fun(s)); options...)
        else
            r = default_evaluation_operator(s, dgs; options...)
        end
    else
        r = default_evaluation_operator(s, dgs; options...)
    end
    if sparse
        return SparseOperator(r; options...)
    else
        return r
    end
end

function BasisFunctions.grid_evaluation_operator(s::S, dgs::DiscreteGridSpace, grid::ProductGrid;
        options...) where {S<:BasisFunctions.Span{A,S,T,D} where {A,S,T,D<: TensorProductDict{N,DT,S,T} where {N,DT <: NTuple{N,BasisFunctions.PeriodicTranslationDict} where N,S,T}}}
    tensorproduct([BasisFunctions.grid_evaluation_operator(si, dgsi, gi; options...) for (si, dgsi, gi) in zip(elements(s), elements(dgs), elements(grid))]...)
end

unsafe_eval_element(b::PeriodicTranslationDict, idxn::TransIndex, x::Real) =
    fun(b)(x-value(idxn)*stepsize(b))

eval_dualelement(b::PeriodicTranslationDict, idx::LinearIndex, x::Real) =
    eval_dualelement(b, native_index(b, idx), x)

eval_dualelement(b::PeriodicTranslationDict, idxn::TransIndex, x::Real) =
    eval_expansion(b, circshift(dualgramcolumn(Span(b)),value(idxn)), x)

Gram(s::PeriodicTranslatesSpan; options...) = CirculantOperator(s, s, primalgramcolumn(s; options...))

function UnNormalizedGram(s::PeriodicTranslatesSpan, oversampling = 1)
    grid = oversampled_grid(dictionary(s), oversampling)
    CirculantOperator(evaluation_operator(s, grid)'*evaluation_operator(s, grid))
end

grammatrix(b::PeriodicTranslatesSpan; options...) = matrix(Gram(b; options...))

dualgrammatrix(b::PeriodicTranslatesSpan; options...) = matrix(inv(Gram(b; options...)))

# All inner products between elements of PeriodicTranslationDict are known by the first column of the (circulant) gram matrix.
function primalgramcolumn(s::PeriodicTranslatesSpan; options...)
    n = length(s)
    result = zeros(coeftype(s), n)
    for i in 1:length(result)
        result[i] = primalgramcolumnelement(s, i; options...)
    end
    result
end

primalgramcolumnelement(s::PeriodicTranslatesSpan, i::Int; options...) =
    defaultprimalgramcolumnelement(s, i; options...)

defaultprimalgramcolumnelement(s::Span1d, i::Int; options...)  = dot(s, 1, i; options...)

function dualgramcolumn(s::PeriodicTranslatesSpan; options...)
    G = inv(Gram(s; options...))
    e = zeros(eltype(G),size(G,1))
    e[1] = 1
    real(G*e)
end

"""
  The function set that correspands to the dual set ``Ψ={ψ_i}_{i∈ℕ}`` of the given set ``Φ={ϕ_i}_{i∈ℕ}``.

"""
dual(set::PeriodicTranslationDict; options...) =
    DualPeriodicTranslationDict(set; options...)
"""
  The function set that correspands to the dual set of the given set.
"""
discrete_dual(set::PeriodicTranslationDict; options...) =
    DiscreteDualPeriodicTranslationDict(set; options...)

"""
  Set consisting of n translates of a compact and periodic function.

  The support of the function is [c_1,c_2], where c_1, c_2 ∈R, c_2-c_1 <= p, 0 ∈ [c_1,c_2],
  and p is the period of the function.
"""
abstract type CompactPeriodicTranslationDict{T} <: PeriodicTranslationDict{T}
end

const CompactPeriodicTranslatesSpan{A,S,T,D <: CompactPeriodicTranslationDict} = Span{A,S,T,D}

"""
  Length of the function of a CompactPeriodicTranslationDict.
"""
length_compact_support{T}(b::CompactPeriodicTranslationDict{T})::real(T) = right_of_compact_function(b)-left_of_compact_function(b)

function left_of_compact_function end
function right_of_compact_function end

support_length_of_compact_function(f::CompactPeriodicTranslationDict) = right_of_compact_function(f::CompactPeriodicTranslationDict)-left_of_compact_function(f::CompactPeriodicTranslationDict)

function overlapping_elements(b::CompactPeriodicTranslationDict, x::Real)
   indices = ceil(Int, (x-BasisFunctions.right_of_compact_function(b))/stepsize(b)):floor(Int, (x-BasisFunctions.left_of_compact_function(b))/stepsize(b))
   Set(mod(i, length(b))+1 for i in indices)
end

left(b::CompactPeriodicTranslationDict, idx::TransIndex) =
    value(idx) * stepsize(b) + left_of_compact_function(b)

right(b::CompactPeriodicTranslationDict, idx::TransIndex) =
    left(b, idx) + length_compact_support(b)

in_support(set::CompactPeriodicTranslationDict, idx::LinearIndex, x::Real) =
    in_support(set, native_index(set, idx), x)

in_support(set::CompactPeriodicTranslationDict, idx::TransIndex, x::Real) =
    in_compact_support(set, idx, x)

unsafe_eval_element(b::CompactPeriodicTranslationDict, idx::TransIndex, x::Real) =
    eval_compact_element(b, idx, x)

eval_expansion(b::CompactPeriodicTranslationDict, coef, x::Real) =
    eval_compact_expansion(b, coef, x)

function eval_compact_element(b::CompactPeriodicTranslationDict{T}, idx::TransIndex, x::Real) where {T}
    !in_support(b, idx, x) ? zero(T) : fun(b)(x-value(idx)*stepsize(b))
end

function in_compact_support(set::CompactPeriodicTranslationDict, idx::TransIndex, x::Real)
	per = period(set)
	A = left(set) <= x <= right(set)
	B = (left(set, idx) <= x <= right(set, idx)) || (left(set, idx) <= x-per <= right(set, idx)) ||
		(left(set, idx) <= x+per <= right(set, idx))
	A && B
end

function eval_compact_expansion(b::CompactPeriodicTranslationDict, coef, x)
    z = zero(typeof(x))
    for idx = BasisFunctions.overlapping_elements(b, x)
        idxn = native_index(b, idx)
        z = z + coef[idx] * BasisFunctions.unsafe_eval_element(b, idxn, x)
    end
    z
end

"""
  Set of translates of a function f that is a linear combination of basis function of an other set of translates.

  `f(x) = ∑ coefficients(set)[k] * superdictionary(set)[k](x)`
"""
abstract type LinearCombinationOfPeriodicTranslationDict{PSoT<:PeriodicTranslationDict, T} <: PeriodicTranslationDict{T}
end

const LinearCombinationsSpan{A,S,T,D <: LinearCombinationOfPeriodicTranslationDict} = Span{A,S,T,D}

coefficients(b::LinearCombinationOfPeriodicTranslationDict) = b.coefficients

for op in (:length, :left, :right, :has_grid, :grid, :domain)
    @eval $op(b::LinearCombinationOfPeriodicTranslationDict) = $op(superdict(b))
end

function fun(b::LinearCombinationOfPeriodicTranslationDict)
    x->eval_expansion(superdict(b), real(coefficients(b)), CardinalBSplines.periodize(x, period(superdict(b))))
end

==(b1::LinearCombinationOfPeriodicTranslationDict, b2::LinearCombinationOfPeriodicTranslationDict) =
    superdict(b1)==superdict(b2) && coefficients(b1) ≈ coefficients(b2)

change_of_basis(b::LinearCombinationOfPeriodicTranslationDict; options...) =
    wrap_operator(Span(superdict(b)), Span(b), inv(change_of_basis(superdict(b), typeof(b))))

change_of_basis(b::PeriodicTranslationDict, ::Type{LinearCombinationOfPeriodicTranslationDict}; options...) = DualGram(Span(b); options...)

function coefficients_in_other_basis{B<:LinearCombinationOfPeriodicTranslationDict}(b::PeriodicTranslationDict, ::Type{B}; options...)
    e = zeros(b)
    e[1] = 1
    change_of_basis(b, B; options...)*e
end

superspan(s::LinearCombinationsSpan) = Span(superdict(dictionary(s)), coeftype(s))

extension_operator(s1::LinearCombinationsSpan, s2::LinearCombinationsSpan; options...) =
    wrap_operator(s1, s2, change_of_basis(dictionary(s2); options...)*extension_operator(superspan(s1), superspan(s2))*inv(change_of_basis(dictionary(s1); options...)))

restriction_operator(s1::LinearCombinationsSpan, s2::LinearCombinationsSpan; options...) =
    wrap_operator(s1, s2, change_of_basis(dictionary(s2); options...)*restriction_operator(superspan(s1), superspan(s2))*inv(change_of_basis(dictionary(s1); options...)))

"""
  Set representing the dual basis.
"""
struct DualPeriodicTranslationDict{T} <: LinearCombinationOfPeriodicTranslationDict{PeriodicTranslationDict, T}
    superdict       :: PeriodicTranslationDict{T}
    coefficients    :: Array{T,1}
end

const DualPeriodicTranslatesSpan{A,S,T,D <: DualPeriodicTranslationDict} = Span{A,S,T,D}

DualPeriodicTranslationDict(set::PeriodicTranslationDict{T}; options...) where {T} =
    DualPeriodicTranslationDict{T}(set, coefficients_in_other_basis(set, LinearCombinationOfPeriodicTranslationDict; options...))

superdict(b::DualPeriodicTranslationDict) = b.superdict

dual(b::DualPeriodicTranslationDict; options...) = superdict(b)

Gram(b::DualPeriodicTranslatesSpan; options...) = inv(Gram(superspan(b); options...))

"""
  Set representing the dual basis with respect to a discrete norm on the oversampled grid.
"""
struct DiscreteDualPeriodicTranslationDict{T} <: LinearCombinationOfPeriodicTranslationDict{PeriodicTranslationDict, T}
    superdict       :: PeriodicTranslationDict{T}
    coefficients    :: Array{T,1}

    oversampling    :: T
end

const DiscreteDualPeriodicTranslatesSpan{A,S,T,D <: DiscreteDualPeriodicTranslationDict} = Span{A,S,T,D}

function DiscreteDualPeriodicTranslationDict(set::PeriodicTranslationDict{T}; oversampling=default_oversampling(set), options...) where {T}
    DiscreteDualPeriodicTranslationDict{T}(set, coefficients_in_other_basis(set, DiscreteDualPeriodicTranslationDict; oversampling=oversampling, options...), oversampling)
end

superdict(b::DiscreteDualPeriodicTranslationDict) = b.superdict
coefficients(b::DiscreteDualPeriodicTranslationDict) = b.coefficients

default_oversampling(b::DiscreteDualPeriodicTranslationDict) = b.oversampling

dual(b::DiscreteDualPeriodicTranslationDict; options...) = superdict(b)

change_of_basis(b::PeriodicTranslationDict, ::Type{DiscreteDualPeriodicTranslationDict}; options...) = DiscreteDualGram(Span(b); options...)

resize(b::DiscreteDualPeriodicTranslationDict, n::Int) = discrete_dual(resize(dual(b), n); oversampling=default_oversampling(b))

dict_promote_domaintype(b::DiscreteDualPeriodicTranslationDict{T}, ::Type{S}) where {T,S} =
    discrete_dual(promote_domaintype(dual(b), S); oversampling=default_oversampling(b))
