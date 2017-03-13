# operated_set.jl

"""
An OperatedSet represents a set that is acted on by an operator, for example the differentiation operator.
The OperatedSet has the dimension of the source set of the operator, but each basis function
is acted on by the operator.
"""
immutable OperatedSet{T} <: FunctionSet{1,T}
    "The operator that acts on the set"
    op          ::  AbstractOperator{T}

    scratch_src
    scratch_dest

    function OperatedSet(op::AbstractOperator)
        scratch_src = zeros(T, src(op))
        scratch_dest = zeros(T, dest(op))
        new(op, scratch_src, scratch_dest)
    end
end

# TODO: OperatedSet should really be a DerivedSet, deriving from src(op)

OperatedSet{ELT}(op::AbstractOperator{ELT}) =
    OperatedSet{ELT}(op)

name(set::OperatedSet) = name(dest(set)) * " transformed by an operator"

src(set::OperatedSet) = src(set.op)

dest(set::OperatedSet) = dest(set.op)

operator(set::OperatedSet) = set.op

set_promote_eltype{T,S}(set::OperatedSet{T}, ::Type{S}) = OperatedSet(promote_eltype(operator(set), S))

for op in (:left, :right, :length)
    @eval $op(set::OperatedSet) = $op(src(set))
end

# We don't know in general what the support of a specific basis functions is.
# The safe option is to return the support of the set itself for each element.
for op in (:left, :right)
    @eval $op(set::OperatedSet, idx) = $op(src(set))
end

zeros(ELT::Type, set::OperatedSet) = zeros(ELT, src(set))


eval_element(set::OperatedSet, i, x) = _eval_element(set, operator(set), i, x)

function _eval_element(set::OperatedSet, op::AbstractOperator, i, x)
    if is_diagonal(op)
        diagonal(op, i) * eval_element(src(set), i, x)
    else
        idx = native_index(set, i)
        set.scratch_src[idx] = 1
        apply!(set.op, set.scratch_dest, set.scratch_src)
        set.scratch_src[idx] = 0
        eval_expansion(dest(set), set.scratch_dest, x)
    end
end

_eval_element(set::OperatedSet, op::ScalingOperator, i, x) = diagonal(op, i) * eval_element(src(set), i, x)

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
derivative(s::FunctionSet; options...) = OperatedSet(differentiation_operator(s; options...))

function (*)(a::Number, set::FunctionSet)
    T = promote_type(typeof(a), eltype(set))
    OperatedSet(ScalingOperator(set, convert(T, a)))
end

function (*)(op::AbstractOperator, set::FunctionSet)
    @assert src(op) == set
    OperatedSet(op)
end
