# operated_set.jl

"""
An OperatedSet represents a set that is acted on by an operator, for example the differentiation operator.
The OperatedSet has the dimension of the source set of the operator, but each basis function
is acted on by the operator.
"""
struct OperatedSet{T} <: FunctionSet{T}
    "The operator that acts on the set"
    op          ::  AbstractOperator{T}

    scratch_src
    scratch_dest

    function OperatedSet{T}(op::AbstractOperator) where T
        scratch_src = zeros(src(op))
        scratch_dest = zeros(dest(op))
        new(op, scratch_src, scratch_dest)
    end
end

# TODO: OperatedSet should really be a DerivedSet, deriving from src(op)

OperatedSet(op::AbstractOperator{T}) where {T} = OperatedSet{T}(op)

name(s::OperatedSet) = name(src_set(s) * " transformed by an operator")

src(s::OperatedSet) = src(s.op)
src_set(s::OperatedSet) = set(src(s))

dest(s::OperatedSet) = dest(s.op)
dest_set(s::OperatedSet) = set(dest(s))

domaintype(s::OperatedSet) = domaintype(src_set(s))

operator(set::OperatedSet) = set.op

set_promote_domaintype(s::OperatedSet{T}, ::Type{S}) where {S,T} =
    OperatedSet(similar_operator(operator(s), T, promote_domaintype(src(s), S), dest(s) ) )

for op in (:left, :right, :length)
    @eval $op(s::OperatedSet) = $op(src_set(s))
end

# We don't know in general what the support of a specific basis functions is.
# The safe option is to return the support of the set itself for each element.
for op in (:left, :right)
    @eval $op(s::OperatedSet, idx) = $op(src_set(s))
end

zeros(::Type{T}, s::OperatedSet) where {T} = zeros(T, src_set(s))


eval_element(set::OperatedSet, i, x) = _eval_element(set, operator(set), i, x)

function _eval_element(s::OperatedSet, op::AbstractOperator, i, x)
    if is_diagonal(op)
        diagonal(op, i) * eval_element(src_set(s), i, x)
    else
        idx = native_index(s, i)
        s.scratch_src[idx] = 1
        apply!(s.op, s.scratch_dest, s.scratch_src)
        s.scratch_src[idx] = 0
        eval_expansion(dest_set(s), s.scratch_dest, x)
    end
end

_eval_element(s::OperatedSet, op::ScalingOperator, i, x) = diagonal(op, i) * eval_element(src_set(s), i, x)

## Properties

isreal(set::OperatedSet) = isreal(operator(set))

# Disable for now
# for op in (:is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
#     @eval $op(set::OperatedSet) = is_diagonal(operator(set)) && $op(src(set))
# end


#################
# Special cases
#################

# If a set has a differentiation operator, then we can represent the set of derivatives
# by an OperatedSet.
derivative(s::Span; options...) = OperatedSet(differentiation_operator(s; options...))

function (*)(a::Number, s::Span)
    T = promote_type(typeof(a), coeftype(s))
    OperatedSet(ScalingOperator(s, convert(T, a)))
end

function (*)(op::AbstractOperator, s::Span)
    @assert src(op) == s
    OperatedSet(op)
end
