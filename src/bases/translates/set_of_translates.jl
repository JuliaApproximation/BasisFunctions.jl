"""
  Set consisting of translates of a function.
"""
abstract SetOfTranslates{T} <: FunctionSet1d{T}

length(set::SetOfTranslates) = set.n

is_biorthogonal(::SetOfTranslates) = true

is_basis(::SetOfTranslates) = true

name{B<:SetOfTranslates}(::Type{B}) = "Set of translates of a function"

name(b::SetOfTranslates) = name(typeof(b))

fun(b::SetOfTranslates) = b.fun

# Indices set of translates naturally range from 0 to n-1
native_index(b::SetOfTranslates, idx::Int) = idx-1
linear_index(b::SetOfTranslates, idxn::Int) = idxn+1

has_unitary_transform(::SetOfTranslates) = false

"""
  Set consisting of n equispaced translates of a periodic function.

  The set can be written as ``\left\{T_k f\right\}_{k=0}^n``, where ``T_k f(x) = f(x-p/n)``.
  ``p`` is the period of the set, ``n`` is the number of elements.
"""
abstract PeriodicSetOfTranslates{T} <: SetOfTranslates{T}

left{T}(set::PeriodicSetOfTranslates{T})::real(T) = real(T)(set.a)

right{T}(set::PeriodicSetOfTranslates{T})::real(T) = real(T)(set.b)

left(set::PeriodicSetOfTranslates, j::Int) = left(set)

right(set::PeriodicSetOfTranslates, j::Int) = right(set)

has_grid(::PeriodicSetOfTranslates) = true

grid(set::PeriodicSetOfTranslates) = PeriodicEquispacedGrid(length(set), left(set), right(set))

period{T}(set::PeriodicSetOfTranslates{T})::real(T) = real(T)(right(set)-left(set))

stepsize{T}(set::PeriodicSetOfTranslates{T})::real(T) = real(T)(period(set)/length(set))

has_grid_transform(b::PeriodicSetOfTranslates, dgs, grid::AbstractEquispacedGrid) =
    compatible_grid(b, grid)

compatible_grid(b::PeriodicSetOfTranslates, grid::AbstractEquispacedGrid) =
    periodic_compatible_grid(b, grid)

function periodic_compatible_grid(b::FunctionSet, grid::AbstractEquispacedGrid)
  l1 = length(b)
  l2 = length(grid)
  l1 > l2 && ((l2,l1) = (l1, l2))
  n = l2/l1
  nInt = round(Int, n)
  (1+(left(b) - left(grid))≈1) && (1+(right(b) - right(grid))≈1) && isdyadic(nInt) && (n≈nInt)
end

native_nodes{T}(b::PeriodicSetOfTranslates{T}) = [real(T)(k*stepsize(b)) for k in 0:length(b)]

function transform_from_grid(src, dest::PeriodicSetOfTranslates, grid; options...)
	inv(transform_to_grid(dest, src, grid; options...))
end

function transform_to_grid(src::PeriodicSetOfTranslates, dest, grid; options...)
  @assert compatible_grid(src, grid)
  CirculantOperator(src, dest, sample(grid, fun(src)); options...)
end

function grid_evaluation_operator(set::PeriodicSetOfTranslates, dgs::DiscreteGridSpace, grid::AbstractEquispacedGrid; options...)
  if periodic_compatible_grid(set, grid)
    lg = length(grid)
    ls = length(set)
    if lg == ls
      return CirculantOperator(set, dgs, sample(grid, fun(set)); options...)
    elseif lg > ls
      return CirculantOperator(dgs, dgs, sample(grid, fun(set)); options...)*IndexExtensionOperator(set, dgs, 1:Int(lg/ls):length(dgs))
    elseif lg < ls && has_extension(grid)
      return IndexRestrictionOperator(set, dgs, 1:Int(ls/lg):length(set))*CirculantOperator(set, set, sample(extend(grid, Int(ls/lg)), fun(set)); options...)
    else
      return default_evaluation_operator(set, dgs; options...)
    end
  end
  default_evaluation_operator(set, dgs; options...)
end

eval_element{T}(b::PeriodicSetOfTranslates{T}, idx::Int, x::Real) = fun(b)(real(T)(x)-native_index(b, idx)*stepsize(b))

eval_dualelement{T}(b::PeriodicSetOfTranslates{T}, idx::Int, x::Real) = eval_expansion(b, circshift(dualgramcolumn(b),native_index(b, idx)), x)

Gram(b::PeriodicSetOfTranslates; options...) = CirculantOperator(b, b, primalgramcolumn(b; options...))

grammatrix(b::PeriodicSetOfTranslates; options...) = matrix(Gram(b; options...))

dualgrammatrix(b::PeriodicSetOfTranslates; options...) = matrix(inv(Gram(b; options...)))

# All inner products between elements of PeriodicSetOfTranslates are known by the first column of the (circulant) gram matrix.
function primalgramcolumn{T}(set::PeriodicSetOfTranslates{T}; options...)
  n = length(set)
  result = zeros(real(T), n)
  for i in 1:length(result)
    result[i] = dot(set, 1, i; options...)
  end
  result
end

function dualgramcolumn(set::PeriodicSetOfTranslates; options...)
  G = inv(Gram(set; options...))
  e = zeros(eltype(G),size(G,1))
  e[1] = 1
  real(G*e)
end

function dual(set::PeriodicSetOfTranslates; options...)
  DualPeriodicSetOfTranslates(set; options...)
end
"""
  Set consisting of n translates of a compact and periodic function.

  The support of the function is [c_1,c_2], where c_1, c_2 ∈R, c_2-c_1 <= p, 0 ∈ [c_1,c_2],
  and p is the period of the function.
"""
abstract CompactPeriodicSetOfTranslates{T} <: PeriodicSetOfTranslates{T}

"""
  Length of the function of a CompactPeriodicSetOfTranslates.
"""
length_compact_support{T}(b::CompactPeriodicSetOfTranslates{T})::real(T) = right_of_compact_function(b)-left_of_compact_function(b)

function left_of_compact_function end
function right_of_compact_function end

overlapping_elements(b::CompactPeriodicSetOfTranslates, x) =
  floor(Int, (x-right_of_compact_function(b))/stepsize(b)):ceil(Int, (x-left_of_compact_function(b))/stepsize(b))

left{T}(b::CompactPeriodicSetOfTranslates{T}, j::Int) ::real(T) = native_index(b, j) * stepsize(b) + left_of_compact_function(b)

right{T}(b::CompactPeriodicSetOfTranslates{T}, j::Int)::real(T) = left(b, j) + length_compact_support(b)

in_support(set::CompactPeriodicSetOfTranslates, idx::Int, x::Real) =
    in_compact_support(set, idx, x)

eval_element(b::CompactPeriodicSetOfTranslates, idx::Int, x::Real) =
    eval_compact_element(b, idx, x)

eval_expansion(b::CompactPeriodicSetOfTranslates, coef, x::Real) =
    eval_compact_expansion(b, coef, x)

function eval_compact_element{T}(b::CompactPeriodicSetOfTranslates{T}, idx::Int, x::Real)
  !in_support(b, idx, x) ?
  real(T)(0) :
  fun(b)(real(T)(x)-native_index(b, idx)*stepsize(b))
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

  `f(x) = ∑ coeffs(set)[k] * supserset(set)[k](x)`
"""
abstract LinearCombinationOfPeriodicSetOfTranslates{PSoT<:PeriodicSetOfTranslates, T} <: PeriodicSetOfTranslates{T}

for op in (:length, :left, :right, :has_grid, :grid)
  @eval $op(b::LinearCombinationOfPeriodicSetOfTranslates) = $op(superset(b))
end

function fun(b::LinearCombinationOfPeriodicSetOfTranslates)
  x->eval_expansion(superset(b), real(coeffs(b)), BasisFunctions.Cardinal_b_splines.periodize(x, period(superset(b))))
end

==(b1::LinearCombinationOfPeriodicSetOfTranslates, b2::LinearCombinationOfPeriodicSetOfTranslates) =
    superset(b1)==superset(b2) && coeffs(b1) ≈ coeffs(b2)

change_of_basis(b::LinearCombinationOfPeriodicSetOfTranslates; options...) = wrap_operator(superset(b), b, inv(change_of_basis(superset(b), typeof(b))))

change_of_basis(b::PeriodicSetOfTranslates, ::Type{LinearCombinationOfPeriodicSetOfTranslates}; options...) = DualGram(b; options...)

function coeffs_in_other_basis{B<:LinearCombinationOfPeriodicSetOfTranslates}(b::PeriodicSetOfTranslates, ::Type{B}; options...)
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
immutable DualPeriodicSetOfTranslates{T} <: LinearCombinationOfPeriodicSetOfTranslates{PeriodicSetOfTranslates, T}
  superset  :: PeriodicSetOfTranslates{T}
  coeffs    :: Array{T,1}
end

function DualPeriodicSetOfTranslates(set::PeriodicSetOfTranslates; options...)
  DualPeriodicSetOfTranslates{eltype(set)}(set, coeffs_in_other_basis(set, LinearCombinationOfPeriodicSetOfTranslates; options...))
end

superset(b::DualPeriodicSetOfTranslates) = b.superset
coeffs(b::DualPeriodicSetOfTranslates) = b.coeffs

dual(b::DualPeriodicSetOfTranslates; options...) = superset(b)

Gram(b::DualPeriodicSetOfTranslates; options...) = inv(Gram(superset(b); options...))
