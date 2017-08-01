# set_of_translates.jl

"""
  Set consisting of translates of a function.
"""
abstract type SetOfTranslates{T} <: FunctionSet{T}
end

const TranslatesSpan{A, F <: SetOfTranslates} = Span{A,F}

length(set::SetOfTranslates) = set.n

is_biorthogonal(::SetOfTranslates) = true

is_basis(::SetOfTranslates) = true

name(b::SetOfTranslates) = "Set of translates of a function"
name(::Type{B}) where {B<:SetOfTranslates}= "Set of translates of a function"

fun(b::SetOfTranslates) = b.fun
fun(s::TranslatesSpan) = fun(set(s))

# Indices set of translates naturally range from 0 to n-1
native_index(b::SetOfTranslates, idx::Int) = idx-1
linear_index(b::SetOfTranslates, idxn::Int) = idxn+1

has_unitary_transform(::SetOfTranslates) = false

"""
  Set consisting of n equispaced translates of a periodic function.

  The set can be written as ``\left\{T_k f\right\}_{k=0}^n``, where ``T_k f(x) = f(x-p/n)``.
  ``p`` is the period of the set, ``n`` is the number of elements.
"""
abstract type PeriodicSetOfTranslates{T} <: SetOfTranslates{T}
end

const PeriodicTranslatesSpan{A, F <: PeriodicSetOfTranslates} = Span{A,F}

left{T}(set::PeriodicSetOfTranslates{T})::real(T) = real(T)(set.a)

right{T}(set::PeriodicSetOfTranslates{T})::real(T) = real(T)(set.b)

left(set::PeriodicSetOfTranslates, j::Int) = left(set)

right(set::PeriodicSetOfTranslates, j::Int) = right(set)

has_grid(::PeriodicSetOfTranslates) = true

grid(set::PeriodicSetOfTranslates) = PeriodicEquispacedGrid(length(set), left(set), right(set))

period(set::PeriodicSetOfTranslates) = right(set)-left(set)

stepsize(set::PeriodicSetOfTranslates) = period(set)/length(set)

has_grid_transform(b::PeriodicSetOfTranslates, dgs, grid::AbstractEquispacedGrid) =
    compatible_grid(b, grid)

compatible_grid(b::PeriodicSetOfTranslates, grid::AbstractEquispacedGrid) =
    periodic_compatible_grid(b, grid)

approx_length(b::PeriodicSetOfTranslates, n::Int) = ceil(Int,n/length(b))*length(b)

function periodic_compatible_grid(b::FunctionSet, grid::AbstractEquispacedGrid)
    l1 = length(b)
    l2 = length(grid)
    l1 > l2 && ((l2,l1) = (l1, l2))
    n = l2/l1
    nInt = round(Int, n)
    (1+(left(b) - leftendpoint(grid))≈1) && (1+(right(b) - rightendpoint(grid))≈1) && isdyadic(nInt) && (n≈nInt)
end

native_nodes(b::PeriodicSetOfTranslates) = [k*stepsize(b) for k in 0:length(b)]

function transform_from_grid(src, dest::PeriodicTranslatesSpan, grid; options...)
    inv(transform_to_grid(dest, src, grid; options...))
end

function transform_to_grid(src::PeriodicTranslatesSpan, dest, grid; options...)
    @assert compatible_grid(set(src), grid)
    CirculantOperator(src, dest, sample(grid, fun(src)); options...)
end

function grid_evaluation_operator(s::PeriodicTranslatesSpan, dgs::DiscreteGridSpace, grid::AbstractEquispacedGrid; options...)
    if periodic_compatible_grid(set(s), grid)
        lg = length(grid)
        ls = length(s)
        if lg == ls
            return CirculantOperator(s, dgs, sample(grid, fun(s)); options...)
        elseif lg > ls
            return CirculantOperator(dgs, dgs, sample(grid, fun(s)); options...)*IndexExtensionOperator(s, dgs, 1:Int(lg/ls):length(dgs))
        elseif lg < ls && has_extension(grid)
            return IndexRestrictionOperator(s, dgs, 1:Int(ls/lg):length(s))*CirculantOperator(s, s, sample(extend(grid, Int(ls/lg)), fun(set)); options...)
        else
            return default_evaluation_operator(s, dgs; options...)
        end
    end
    default_evaluation_operator(s, dgs; options...)
end

eval_element(b::PeriodicSetOfTranslates, idx::Int, x::Real) = fun(b)(x-native_index(b, idx)*stepsize(b))

eval_dualelement(b::PeriodicSetOfTranslates, idx::Int, x::Real) = eval_expansion(b, circshift(dualgramcolumn(span(b)),native_index(b, idx)), x)

Gram(s::PeriodicTranslatesSpan; options...) = CirculantOperator(s, s, primalgramcolumn(s; options...))

function UnNormalizedGram(s::PeriodicTranslatesSpan, oversampling = 1)
    grid = oversampled_grid(set(s), oversampling)
    CirculantOperator(evaluation_operator(s, grid)'*evaluation_operator(s, grid))
end

grammatrix(b::PeriodicTranslatesSpan; options...) = matrix(Gram(b; options...))

dualgrammatrix(b::PeriodicTranslatesSpan; options...) = matrix(inv(Gram(b; options...)))

# All inner products between elements of PeriodicSetOfTranslates are known by the first column of the (circulant) gram matrix.
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
dual(set::PeriodicSetOfTranslates; options...) =
    DualPeriodicSetOfTranslates(set; options...)
"""
  The function set that correspands to the dual set of the given set.
"""
discrete_dual(set::PeriodicSetOfTranslates; options...) =
    DiscreteDualPeriodicSetOfTranslates(set; options...)

"""
  Set consisting of n translates of a compact and periodic function.

  The support of the function is [c_1,c_2], where c_1, c_2 ∈R, c_2-c_1 <= p, 0 ∈ [c_1,c_2],
  and p is the period of the function.
"""
abstract type CompactPeriodicSetOfTranslates{T} <: PeriodicSetOfTranslates{T}
end

const CompactPeriodicTranslatesSpan{A, F <: CompactPeriodicSetOfTranslates} = Span{A,F}

"""
  Length of the function of a CompactPeriodicSetOfTranslates.
"""
length_compact_support{T}(b::CompactPeriodicSetOfTranslates{T})::real(T) = right_of_compact_function(b)-left_of_compact_function(b)

function left_of_compact_function end
function right_of_compact_function end

overlapping_elements(b::CompactPeriodicSetOfTranslates, x) =
   floor(Int, (x-right_of_compact_function(b))/stepsize(b)):ceil(Int, (x-left_of_compact_function(b))/stepsize(b))

left(b::CompactPeriodicSetOfTranslates, j::Int) = native_index(b, j) * stepsize(b) + left_of_compact_function(b)

right(b::CompactPeriodicSetOfTranslates, j::Int) = left(b, j) + length_compact_support(b)

in_support(set::CompactPeriodicSetOfTranslates, idx::Int, x::Real) =
    in_compact_support(set, idx, x)

eval_element(b::CompactPeriodicSetOfTranslates, idx::Int, x::Real) =
    eval_compact_element(b, idx, x)

eval_expansion(b::CompactPeriodicSetOfTranslates, coef, x::Real) =
    eval_compact_expansion(b, coef, x)

function eval_compact_element(b::CompactPeriodicSetOfTranslates{T}, idx::Int, x::Real) where {T}
  !in_support(b, idx, x) ? zero(T) : fun(b)(x-native_index(b, idx)*stepsize(b))
end

function in_compact_support(set::CompactPeriodicSetOfTranslates, idx::Int, x::Real)
	per = period(set)
	A = left(set) <= x <= right(set)
	B = (left(set, idx) <= x <= right(set, idx)) || (left(set, idx) <= x-per <= right(set, idx)) ||
		(left(set, idx) <= x+per <= right(set, idx))
	A && B
end

function eval_compact_expansion{T <: Number}(b::CompactPeriodicSetOfTranslates, coef, x::T)
	z = zero(T)
	for idxn = overlapping_elements(b, x)
		idx = linear_index(b, mod(idxn,length(b)))
		z = z + coef[idx] * eval_element(b, idx, x)
	end
	z
end

"""
  Set of translates of a function f that is a linear combination of basis function of an other set of translates.

  `f(x) = ∑ coefficients(set)[k] * supserset(set)[k](x)`
"""
abstract type LinearCombinationOfPeriodicSetOfTranslates{PSoT<:PeriodicSetOfTranslates, T} <: PeriodicSetOfTranslates{T}
end

for op in (:length, :left, :right, :has_grid, :grid)
    @eval $op(b::LinearCombinationOfPeriodicSetOfTranslates) = $op(superset(b))
end

function fun(b::LinearCombinationOfPeriodicSetOfTranslates)
    x->eval_expansion(superset(b), real(coefficients(b)), BasisFunctions.Cardinal_b_splines.periodize(x, period(superset(b))))
end

==(b1::LinearCombinationOfPeriodicSetOfTranslates, b2::LinearCombinationOfPeriodicSetOfTranslates) =
    superset(b1)==superset(b2) && coefficients(b1) ≈ coefficients(b2)

change_of_basis(b::LinearCombinationOfPeriodicSetOfTranslates; options...) = wrap_operator(superset(b), b, inv(change_of_basis(superset(b), typeof(b))))

change_of_basis(b::PeriodicSetOfTranslates, ::Type{LinearCombinationOfPeriodicSetOfTranslates}; options...) = DualGram(b; options...)

function coefficients_in_other_basis{B<:LinearCombinationOfPeriodicSetOfTranslates}(b::PeriodicSetOfTranslates, ::Type{B}; options...)
    e = zeros(b)
    e[1] = 1
    change_of_basis(b, B; options...)*e
end

extension_operator{B,T}(s1::LinearCombinationOfPeriodicSetOfTranslates{B,T}, s2::LinearCombinationOfPeriodicSetOfTranslates{B,T}; options...) =
    wrap_operator(s1, s2, change_of_basis(s2; options...)*extension_operator(superset(s1), superset(s2))*inv(change_of_basis(s1; options...)))

restriction_operator{B,T}(s1::LinearCombinationOfPeriodicSetOfTranslates{B,T}, s2::LinearCombinationOfPeriodicSetOfTranslates{B,T}; options...) =
    wrap_operator(s1, s2, change_of_basis(s2; options...)*restriction_operator(superset(s1), superset(s2))*inv(change_of_basis(s1; options...)))

"""
  Set representing the dual basis.
"""
struct DualPeriodicSetOfTranslates{T} <: LinearCombinationOfPeriodicSetOfTranslates{PeriodicSetOfTranslates, T}
    superset        :: PeriodicSetOfTranslates{T}
    coefficients    :: Array{T,1}
end

const DualPeriodicTranslatesSpan{A, F <: DualPeriodicSetOfTranslates} = Span{A,F}

function DualPeriodicSetOfTranslates(set::PeriodicSetOfTranslates; options...)
  DualPeriodicSetOfTranslates{eltype(set)}(set, coefficients_in_other_basis(set, LinearCombinationOfPeriodicSetOfTranslates; options...))
end

superset(b::DualPeriodicSetOfTranslates) = b.superset
coefficients(b::DualPeriodicSetOfTranslates) = b.coefficients

dual(b::DualPeriodicSetOfTranslates; options...) = superset(b)

Gram(b::Span{A,F}; options...) where {A,F<:DualPeriodicSetOfTranslates} = inv(Gram(Span(superset(set(b)), A); options...))

"""
  Set representing the dual basis with respect to a discrete norm on the oversampled grid.
"""
struct DiscreteDualPeriodicSetOfTranslates{T} <: LinearCombinationOfPeriodicSetOfTranslates{PeriodicSetOfTranslates, T}
    superset        :: PeriodicSetOfTranslates{T}
    coefficients    :: Array{T,1}

    oversampling    :: T
end

const DiscreteDualPeriodicTranslatesSpan{A, F <: DiscreteDualPeriodicSetOfTranslates} = Span{A,F}

function DiscreteDualPeriodicSetOfTranslates(set::PeriodicSetOfTranslates; oversampling=default_oversampling(set), options...)
    DiscreteDualPeriodicSetOfTranslates{eltype(set)}(set, coefficients_in_other_basis(set, DiscreteDualPeriodicSetOfTranslates; oversampling=oversampling, options...), oversampling)
end

superset(b::DiscreteDualPeriodicSetOfTranslates) = b.superset
coeffs(b::DiscreteDualPeriodicSetOfTranslates) = b.coeffs

default_oversampling(b::DiscreteDualPeriodicSetOfTranslates) = b.oversampling

dual(b::DiscreteDualPeriodicSetOfTranslates; options...) = superset(b)

change_of_basis(b::PeriodicSetOfTranslates, ::Type{DiscreteDualPeriodicSetOfTranslates}; options...) = DiscreteDualGram(b; options...)

resize(b::DiscreteDualPeriodicSetOfTranslates, n::Int) = discrete_dual(resize(dual(b), n); oversampling=default_oversampling(b))

set_promote_domaintype{T,S}(b::DiscreteDualPeriodicSetOfTranslates{T}, ::Type{S}) = discrete_dual(promote_domaintype(dual(b), S); oversampling=default_oversampling(b))
