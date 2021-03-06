
"""
An `OperatedDict` represents a set that is acted on by an operator, for example
the differentiation operator. The `OperatedDict` has the dimension of the source
set of the operator, but each basis function is acted on by the operator.

In practice the operator `D` acts on the coefficients. Thus, an expansion
in an `OperatedDict` with coefficients `c` corresponds to an expansion in the
underlying dictionary with coefficients `Dc`. If `D` represents differentiation,
then the `OperatedDict` effectively represents the dictionary of derivatives of
the underlying dictionary elements.
"""
struct OperatedDict{S,T} <: DerivedDict{S,T}
    "The operator that acts on the set"
    op          ::  DictionaryOperator{T}

    scratch_src
    scratch_dest

    function OperatedDict{S,T}(op::DictionaryOperator) where {S,T}
        scratch_src = zeros(src(op))
        scratch_dest = zeros(dest(op))
        new(op, scratch_src, scratch_dest)
    end
end

name(dict::OperatedDict) = "Dictionary modified by a linear operator"

superdict(dict::OperatedDict) = src(dict)

function OperatedDict(op::DictionaryOperator{T}) where {T}
    S = domaintype(src(op))
    OperatedDict{S,T}(op)
end

OperatedDict(op::IdentityOperator) = src(op)

## Printing

modifiersymbol(dict::OperatedDict) = operator(dict)

stencilarray(dict::OperatedDict) = [operator(dict), " * ", superdict(dict)]
stencil_parentheses(dict::OperatedDict) = true
object_parentheses(dict::OperatedDict) = true

src(s::OperatedDict) = src(s.op)

dest(s::OperatedDict) = dest(s.op)

domaintype(s::OperatedDict) = domaintype(src(s))

operator(set::OperatedDict) = set.op

function similar(d::OperatedDict, ::Type{T}, dims::Int...) where {T}
    # We enforce an equal length since we may be able to resize dictionaries,
    # but we can not in general resize the operator (we typically do not store what
    # the operator means in its type).
    @assert length(d) == prod(dims)
    # not sure what to do with dest(d) here - in the meantime invoke similar
    OperatedDict(similar_operator(operator(d), similar(src(d), T), similar(dest(d),T) ))
end

function similardictionary(d::OperatedDict, dict::Dictionary)
    if dict == src(d)
        d
    elseif src(d) == dest(d)
        OperatedDict(wrap_operator(dict, dict, operator(d)))
    end
end

for op in (:support, :length)
    @eval $op(s::OperatedDict) = $op(src(s))
end

# We don't know in general what the support of a specific basis functions is.
# The safe option is to return the support of the set itself for each element.
for op in (:support,)
    @eval $op(s::OperatedDict, idx) = $op(src(s))
end

dict_in_support(set::OperatedDict, i, x) = in_support(superdict(set), x)


zeros(::Type{T}, s::OperatedDict) where {T} = zeros(T, src(s))
tocoefficientformat(a, d::OperatedDict) = tocoefficientformat(a, src(d))


##########################
# Indexing and evaluation
##########################

ordering(dict::OperatedDict) = ordering(src(dict))

unsafe_eval_element(dict::OperatedDict, i, x) =
    _unsafe_eval_element(dict, i, x, operator(dict), dict.scratch_src, dict.scratch_dest)

function _unsafe_eval_element(dict::OperatedDict, idxn, x, op, scratch_src, scratch_dest)
    idx = linear_index(dict, idxn)

    scratch_src[idx] = 1
    apply!(op, scratch_dest, scratch_src)
    scratch_src[idx] = 0
    eval_expansion(dest(dict), scratch_dest, x)
end

function _unsafe_eval_element(dict::OperatedDict, idxn, x, op::ScalingOperator, scratch_src, scratch_dest)
    idx = linear_index(dict, idxn)
    diag(op, idx) * unsafe_eval_element(src(dict), idxn, x)
end

evaluation(::Type{T}, dict::OperatedDict, gb::GridBasis, grid::AbstractGrid; options...) where {T} =
    wrap_operator(dict, gb, evaluation(T, dest(dict), gb, grid; options...) * operator(dict))

## Properties

# Perhaps we can extend the underlying dictionary, but in general we can not extend the operator
hasextension(dict::OperatedDict) = false

isreal(dict::OperatedDict) = isreal(operator(dict))

for op in (:isbasis, :isframe)
    @eval $op(set::OperatedDict) = isdiag(operator(set)) && $op(src(set))
end


#################
# Transform
#################

hastransform(dict::OperatedDict) = isdiag(operator(dict)) && hastransform(src(dict))
hastransform(dict::OperatedDict, dgs::GridBasis) = isdiag(operator(dict)) && hastransform(src(dict), dgs)

transform_dict(dict::OperatedDict; options...) = transform_dict(superdict(dict); options...)

function transform_to_grid(T, src::OperatedDict, dest::GridBasis, grid; options...)
	op = transform_to_grid(T, superdict(src), dest, grid; options...) * operator(src)
	wrap_operator(src, dest, op)
end

function transform_from_grid(T, src::GridBasis, dest::OperatedDict, grid; options...)
	 op = inv(operator(dest)) * transform_from_grid(T, src, superdict(dest), grid; options...)
	 wrap_operator(src, dest, op)
 end


#################
# Differentiation
#################

diff(dict::OperatedDict, args...; options...) = differentiation(dest(dict), args...; options...) * dict

hasderivative(dict::OperatedDict) = hasderivative(superdict(dict))
hasderivative(dict::OperatedDict, order) = hasderivative(superdict(dict), order)

# TODO: allow a diagonal operator
hasantiderivative(dict::OperatedDict) = false

function derivative_dict(dict::OperatedDict, order; options...)
	if orderiszero(order)
		dict
	else
		derivative_dict(dest(dict), order; options...)
	end
end

differentiation(::Type{T}, dsrc::OperatedDict, ddest::Dictionary, order; options...) where {T} =
	differentiation(T, dest(dsrc), ddest, order; options...) * operator(dsrc)

function unsafe_eval_element_derivative(dict::OperatedDict, i, x, order)
	@assert order == 1
    _unsafe_eval_element_derivative(dict, i, x, order, operator(dict), dict.scratch_src, dict.scratch_dest)
end

function _unsafe_eval_element_derivative(dict::OperatedDict, idxn, x, order, op, scratch_src, scratch_dest)
    idx = linear_index(dict, idxn)
    if isdiag(op)
        diag(op, idx) * unsafe_eval_element_derivative(src(dict), idxn, x, order)
    else
        scratch_src[idx] = 1
        apply!(op, scratch_dest, scratch_src)
        scratch_src[idx] = 0
		D = differentiation(superdict(dict))
        eval_expansion(dest(D), D*scratch_dest, x)
    end
end

#################
# Projections
#################

hasmeasure(dict::OperatedDict) = hasmeasure(superdict(dict))

measure(dict::OperatedDict) = measure(superdict(dict))

for f in (:isorthogonal, :isorthonormal, :isbiorthogonal)
    @eval $f(dict::OperatedDict, measure::AbstractMeasure) =
        $f(superdict(dict), measure) && $f(operator(dict))
end

gram1(T, dict::OperatedDict, measure; options...) =
	adjoint(operator(dict)) * gram(T, dest(operator(dict)), measure; options...) * operator(dict)

function innerproduct1(d1::OperatedDict, i, d2, j, measure; options...)
	op = operator(d1)
	if isdiag(op)
		idx = linear_index(d1, i)
		conj(diag(op, idx)) * innerproduct(superdict(d1), i, d2, j, measure; options...)
	else
		innerproduct2(d1, i, d2, j, measure; options...)
	end
end

function innerproduct2(d1, i, d2::OperatedDict, j, measure; options...)
	op = operator(d2)
	if isdiag(op)
		idx = linear_index(d2, j)
		diag(op, idx) * innerproduct(d1, i, superdict(d2), j, measure; options...)
	else
		default_dict_innerproduct(d1, i, d2, j, measure; options...)
	end
end

function mixedgram1(T, dict1::OperatedDict, dict2, measure; options...)
	G = mixedgram(T, superdict(dict1), dict2, measure; options...)
	wrap_operator(dict2, dict1, adjoint(operator(dict1)) * G)
end

function mixedgram2(T, dict1, dict2::OperatedDict, measure; options...)
	G = mixedgram(T, dict1, superdict(dict2), measure; options...)
	wrap_operator(dict2, dict1, G *  operator(dict2))
end

#################
# Special cases
#################
(*)(a::Number, s::Dictionary) = ScalingOperator(s, a) * s
(*)(a::Number, s::OperatedDict) = (ScalingOperator(dest(s), a) * operator(s)) * superdict(s)

function (*)(op::DictionaryOperator, dict::Dictionary)
    @assert src(op) == dict
    OperatedDict(op)
end

function (*)(op::DictionaryOperator, dict::OperatedDict)
    @assert src(op) == dest(dict)
    OperatedDict(op*operator(dict))
end
