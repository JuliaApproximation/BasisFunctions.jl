# operated_dict.jl

"""
An `OperatedDict` represents a set that is acted on by an operator, for example
the differentiation operator. The `OperatedDict` has the dimension of the source
set of the operator, but each basis function is acted on by the operator.
"""
struct OperatedDict{S,T} <: DerivedDict{S,T}
    "The operator that acts on the set"
    op          ::  AbstractOperator{T}

    scratch_src
    scratch_dest

    function OperatedDict{S,T}(op::AbstractOperator) where {S,T}
        scratch_src = zeros(src(op))
        scratch_dest = zeros(dest(op))
        new(op, scratch_src, scratch_dest)
    end
end

const OperatedDictSpan{A,S,T,D <: OperatedDict} = Span{A,S,T,D}

# TODO: OperatedDict should really be a DerivedDict, deriving from src(op)

superdict(dict::OperatedDict) = dictionary(src(dict))

function OperatedDict(op::AbstractOperator{T}) where {T}
    S = domaintype(src(op))
    OperatedDict{S,T}(op)
end

name(s::OperatedDict) = name(src_dictionary(s) * " transformed by an operator")

src(s::OperatedDict) = src(s.op)
src_dictionary(s::OperatedDict) = dictionary(src(s))

dest(s::OperatedDict) = dest(s.op)
dest_dictionary(s::OperatedDict) = dictionary(dest(s))

domaintype(s::OperatedDict) = domaintype(src_dictionary(s))

operator(set::OperatedDict) = set.op

in_support(dict::OperatedDict, idx, x) = default_in_support(dict, idx, x)

dict_promote_domaintype(s::OperatedDict{T}, ::Type{S}) where {S,T} =
    OperatedDict(similar_operator(operator(s), T, promote_domaintype(src(s), S), dest(s) ) )

for op in (:left, :right, :length)
    @eval $op(s::OperatedDict) = $op(src_dictionary(s))
end

# We don't know in general what the support of a specific basis functions is.
# The safe option is to return the support of the set itself for each element.
for op in (:left, :right)
    @eval $op(s::OperatedDict, idx) = $op(src_dictionary(s))
end

zeros(::Type{T}, s::OperatedDict) where {T} = zeros(T, src_dictionary(s))

##########################
# Indexing and evaluation
##########################

ordering(dict::OperatedDict) = ordering(src_dictionary(dict))

unsafe_eval_element(dict::OperatedDict, i, x) =
    _unsafe_eval_element(dict, i, x, operator(dict), dict.scratch_src, dict.scratch_dest)

function _unsafe_eval_element(dict::OperatedDict, idxn, x, op, scratch_src, scratch_dest)
    idx = linear_index(dict, idxn)
    if is_diagonal(op)
        diagonal(op, idx) * unsafe_eval_element(src_dictionary(dict), idxn, x)
    else
        scratch_src[idx] = 1
        apply!(op, scratch_dest, scratch_src)
        scratch_src[idx] = 0
        eval_expansion(dest_dictionary(dict), scratch_dest, x)
    end
end

function _unsafe_eval_element(dict::OperatedDict, idxn, x, op::ScalingOperator, scratch_src, scratch_dest)
    idx = linear_index(dict, idxn)
    diagonal(op, idx) * unsafe_eval_element(src_dictionary(dict), idxn, x)
end

grid_evaluation_operator(span::OperatedDictSpan, dgs::DiscreteGridSpace, grid::AbstractGrid; options...) = grid_evaluation_operator(src(dictionary(span)), dgs, grid; options...)*operator(dictionary(span))

## Properties

isreal(dict::OperatedDict) = isreal(operator(dict))

# Disable for now
# for op in (:is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
#     @eval $op(set::OperatedDict) = is_diagonal(operator(set)) && $op(src(set))
# end


#################
# Special cases
#################

# If a set has a differentiation operator, then we can represent the set of derivatives
# by an OperatedDict.
derivative(s::Span; options...) = OperatedDict(differentiation_operator(s; options...))
derivative(dict::Dictionary; options...) = derivative(Span(dict); options...)

function (*)(a::Number, s::Span)
    T = promote_type(typeof(a), coeftype(s))
    OperatedDict(ScalingOperator(s, convert(T, a)))
end

function (*)(op::AbstractOperator, s::Span)
    @assert src(op) == s
    OperatedDict(op)
end
