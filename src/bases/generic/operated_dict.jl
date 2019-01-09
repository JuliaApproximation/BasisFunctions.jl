
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



superdict(dict::OperatedDict) = src(dict)

function OperatedDict(op::DictionaryOperator{T}) where {T}
    S = domaintype(src(op))
    OperatedDict{S,T}(op)
end

has_stencil(s::OperatedDict) = true
function stencil(s::OperatedDict,S)
    A = Any[]
    push!(A,operator(s))
    push!(A," * ")
    push!(A,src(s))
    return recurse_stencil(s,A,S)
end
myLeaves(s::OperatedDict) = (operator(s),myLeaves(src(s))...)

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

function similar_dictionary(d::OperatedDict, dict::Dictionary)
    if dict == src(d)
        d
    elseif src(d) == dest(d)
        OperatedDictionary(wrap_operator(dict, dict, operator(d)))
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


##########################
# Indexing and evaluation
##########################

ordering(dict::OperatedDict) = ordering(src(dict))

unsafe_eval_element(dict::OperatedDict, i, x) =
    _unsafe_eval_element(dict, i, x, operator(dict), dict.scratch_src, dict.scratch_dest)

function _unsafe_eval_element(dict::OperatedDict, idxn, x, op, scratch_src, scratch_dest)
    idx = linear_index(dict, idxn)
    if isdiagonal(op)
        diagonal(op, idx) * unsafe_eval_element(src(dict), idxn, x)
    else
        scratch_src[idx] = 1
        apply!(op, scratch_dest, scratch_src)
        scratch_src[idx] = 0
        eval_expansion(dest(dict), scratch_dest, x)
    end
end

function _unsafe_eval_element(dict::OperatedDict, idxn, x, op::ScalingOperator, scratch_src, scratch_dest)
    idx = linear_index(dict, idxn)
    diagonal(op, idx) * unsafe_eval_element(src(dict), idxn, x)
end

grid_evaluation_operator(dict::OperatedDict, gb::GridBasis, grid::AbstractGrid; options...) =
    wrap_operator(dict, gb, grid_evaluation_operator(dest(dict), gb, grid; options...) * operator(dict))

## Properties

# Perhaps we can extend the underlying dictionary, but in general we can not extend the operator
has_extension(dict::OperatedDict) = false

isreal(dict::OperatedDict) = isreal(operator(dict))

for op in (:isbasis, :isframe)
    @eval $op(set::OperatedDict) = isdiagonal(operator(set)) && $op(src(set))
end


#################
# Transform
#################

has_transform(dict::OperatedDict) = isdiagonal(operator(dict)) && has_transform(src(dict))
has_transform(dict::OperatedDict, dgs::GridBasis) = isdiagonal(operator(dict)) && has_transform(src(dict), dgs)

transform_dict(dict::OperatedDict; options...) = transform_dict(superdict(dict); options...)

function transform_to_grid(src::OperatedDict, dest::GridBasis, grid; options...)
	op = transform_to_grid(superdict(src), dest, grid; options...) * operator(src)
	wrap_operator(src, dest, op)
end

function transform_from_grid(src::GridBasis, dest::OperatedDict, grid; options...)
	 op = inv(operator(dest)) * transform_from_grid(src, superdict(dest), grid; options...)
	 wrap_operator(src, dest, op)
 end


#################
# Differentiation
#################

# has_derivative(dict::OperatedDict) = has_derivative(superdict(dict))
has_derivative(dict::OperatedDict) = has_derivative(superdict(dict))
has_antiderivative(dict::OperatedDict) = false

derivative_dict(dict::OperatedDict, order::Int; options...) =
	derivative_dict(dest(dict), order; options...)

differentiation_operator(dict::OperatedDict, ddict::Dictionary, order::Int; options...) =
	differentiation_operator(dest(dict), ddict, order; options...) * operator(dict)

unsafe_eval_element_derivative(dict::OperatedDict, i, x) =
    _unsafe_eval_element_derivative(dict, i, x, operator(dict), dict.scratch_src, dict.scratch_dest)

function _unsafe_eval_element_derivative(dict::OperatedDict, idxn, x, op, scratch_src, scratch_dest)
    idx = linear_index(dict, idxn)
    if isdiagonal(op)
        diagonal(op, idx) * unsafe_eval_element_derivative(src(dict), idxn, x)
    else
        scratch_src[idx] = 1
        apply!(op, scratch_dest, scratch_src)
        scratch_src[idx] = 0
		D = differentiation_operator(superdict(dict))
        eval_expansion(dest(D), D*scratch_dest, x)
    end
end

#################
# Projections
#################

has_measure(dict::OperatedDict) = has_measure(superdict(dict))

measure(dict::OperatedDict) = measure(superdict(dict))

gramoperator(dict::OperatedDict; options...) =
	adjoint(operator(dict)) * gramoperator(superdict(dict); options...) * operator(dict)

function innerproduct1(d1::OperatedDict, i, d2, j, measure; options...)
	op = operator(d1)
	if is_diagonal(op)
		idx = linear_index(d1, i)
		diagonal(op, idx) * innerproduct(superdict(d1), i, d2, j, measure; options...)
	else
		innerproduct2(d1, i, d2, j, measure; options...)
	end
end

function innerproduct2(d1, i, d2::OperatedDict, j, measure; options...)
	op = operator(d2)
	if is_diagonal(op)
		idx = linear_index(d2, j)
		diagonal(op, idx) * innerproduct(d1, i, superdict(d2), j, measure; options...)
	else
		default_dict_innerproduct(d1, i, d2, j, measure; options...)
	end
end

#################
# Special cases
#################

# If a set has a differentiation operator, then we can represent the set of derivatives
# by an OperatedDict.
derivative(dict::Dictionary; options...) = differentiation_operator(dict; options...) * dict

derivative(dict::OperatedDict; options...) = differentiation_operator(dest(dict); options...) * dict

function (*)(a::Number, s::Dictionary)
    T = promote_type(typeof(a), coefficienttype(s))
    OperatedDict(ScalingOperator(s, convert(T, a)))
end

function (*)(op::DictionaryOperator, dict::Dictionary)
    @assert src(op) == dict
    OperatedDict(op)
end

function (*)(op::DictionaryOperator, dict::OperatedDict)
    @assert src(op) == dest(dict)
    OperatedDict(op*operator(dict))
end
