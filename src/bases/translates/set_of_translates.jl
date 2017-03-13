"""
  Set consisting of translates of a function.
"""
abstract SetOfTranslates{T} <: FunctionSet1d{T}

length(set::SetOfTranslates) = set.n

is_biorthogonal(::SetOfTranslates) = true

is_basis(::SetOfTranslates) = true

name(b::SetOfTranslates) = "Set of translates of a function"

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

left{T}(set::PeriodicSetOfTranslates{T}) = real(T)(set.a)

right{T}(set::PeriodicSetOfTranslates{T}) = real(T)(set.b)

left(set::PeriodicSetOfTranslates, j::Int) = left(set)

right(set::PeriodicSetOfTranslates, j::Int) = right(set)

has_grid(::PeriodicSetOfTranslates) = true

# grid(set::PeriodicSetOfTranslates) = MidpointEquispacedGrid(length(set), left(set), right(set))

period{T}(set::PeriodicSetOfTranslates{T}) = T(right(set)-left(set))

stepsize{T}(set::PeriodicSetOfTranslates{T}) = T(period(set)/length(set))

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

native_nodes{T}(b::PeriodicSetOfTranslates{T}) = [T(k*stepsize(b)) for k in 0:length(b)]

function transform_from_grid(src, dest::PeriodicSetOfTranslates, grid; options...)
	inv(transform_to_grid(dest, src, grid; options...))
end

function transform_to_grid(src::PeriodicSetOfTranslates, dest, grid; options...)
  @assert compatible_grid(src, grid)
  CirculantOperator(src, dest, sample(grid, fun(src)); options...)
end

"Return the index of the interval between two knots in which x lies, starting from index 0."
interval(b::PeriodicSetOfTranslates, x) = round(Int, floor( (x-left(b))/stepsize(b) ))

eval_element{T}(b::PeriodicSetOfTranslates{T}, idx::Int, x) = fun(T(x)-native_index(b, idx)*stepsize(b))

eval_dualelement{T}(b::PeriodicSetOfTranslates{T}, idx::Int, x) = eval_expansion(b, circshift(dualgramcolumn(b),native_index(b, idx)), x)

grammatrix(b::PeriodicSetOfTranslates) = full(Circulant(primalgramcolumn(b)))

dualgrammatrix(b::PeriodicSetOfTranslates) = full(Circulant(dualgramcolumn(b)))

# All inner products between elements of PeriodicSetOfTranslates are known by the first column of the (circulant) gram matrix.
function primalgramcolumn{T}(set::PeriodicSetOfTranslates{T}; options...)
  n = length(set)
  result = zeros(real(T), n)
  for i in 1:length(result)
    result[i] = dot(set, 1, i; options...)
  end
  result
end

dualgramcolumn(set::PeriodicSetOfTranslates; options...) =
    dualgramcolumn(primalgramcolumn(set; options...))

# The inverse of a circulant matrix is calculated in O(n logn) by use of an fft.
function dualgramcolumn{T}(primalgramcolumn::Array{T,1})
    n = length(primalgramcolumn)
    e1 = zeros(T,n); e1[1] = 1;
    d = 1./fft(primalgramcolumn)
    real(ifft(Diagonal(d)*fft(e1)))
end

"""
  Set consisting of n translates of a compact and periodic function.

  The support of the function is [0,c], where c∈R and 0 < c < p,
  where p is the period of the function.
"""
abstract CompactPeriodicSetOfTranslates{T} <: PeriodicSetOfTranslates{T}

"""
  Length of the function of a CompactPeriodicSetOfTranslates.
"""
function length_compact_support end

left(b::CompactPeriodicSetOfTranslates, j::Int) = native_index(b, j) * stepsize(b)

right(b::CompactPeriodicSetOfTranslates, j::Int) = left(b, j) + length_compact_support(b)

# return whether x lays in the support of the idxth element of set. x should lay in the support of the set.
function in_support{K}(set::CompactPeriodicSetOfTranslates{K}, idx::Int, x)
	per = period(set)
	A = left(set) <= x <= right(set)
	B = (left(set, idx) <= x <= right(set, idx)) || (left(set, idx) <= x-per <= right(set, idx)) ||
		(left(set, idx) <= x+per <= right(set, idx))
	A && B
end

eval_element{T}(b::CompactPeriodicSetOfTranslates{T}, idx::Int, x) = !in_support(b, idx, x) ?
  T(0) :
  fun(b)(T(x)-native_index(b, idx)*stepsize(b))

function eval_expansion{T <: Number}(b::CompactPeriodicSetOfTranslates, coef, x::T)
	i = interval(b, x)
	n = length(b)
	z = zero(T)
  nointervals = ceil(Int, length_compact_support(b)/stepsize(b))
	for idxn = i-nointervals:i
		idx = linear_index(b, mod(idxn,n))
		z = z + coef[idx] * eval_element(b, idx, x)
	end
	z
end

function grid_evaluation_operator(set::PeriodicSetOfTranslates, dgs::DiscreteGridSpace, grid::AbstractEquispacedGrid; options...)
  if periodic_compatible_grid(set, grid)
    lg = length(grid)
    ls = length(set)
    if lg == ls
      return CirculantOperator(set, dgs, sample(grid, fun(set)); options...)
    elseif lg > ls
      return CirculantOperator(dgs, dgs, sample(grid, fun(set)); options...)*ExpandOperator(set, dgs, 1, Int(lg/ls))
    elseif lg < ls && has_extension(grid)
      return SelectOperator(set, dgs, 1, Int(ls/lg))*CirculantOperator(set, set, sample(extend(grid, Int(ls/lg)), fun(set)); options...)
    else
      return default_evaluation_operator(set, dgs; options...)
    end
  end
  default_evaluation_operator(set, dgs; options...)
end
