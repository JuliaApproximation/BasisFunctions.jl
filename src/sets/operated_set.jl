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


OperatedSet{ELT}(op::AbstractOperator{ELT}) =
    OperatedSet{ELT}(op)

name(s::OperatedSet) = name(dest(s)) * " transformed by an operator"

src(s::OperatedSet) = src(s.op)

dest(s::OperatedSet) = dest(s.op)

operator(s::OperatedSet) = s.op

# We want to do this in the future:
#promote_eltype{T,S}(s::OperatedSet{T}, ::Type{S}) = OperatedSet(promote_eltype(operator(s), S))
# But for now:
promote_eltype{T,S}(s::OperatedSet{T}, ::Type{S}) = s

for op in (:left, :right, :length)
    @eval $op(b::OperatedSet) = $op(src(b))
end

zeros(ELT::Type, s::OperatedSet) = zeros(ELT, src(s))


function call_element(s::OperatedSet, i, x)
    idx = native_index(s, i)
    s.scratch_src[idx] = 1
    apply!(s.op, s.scratch_dest, s.scratch_src)
    s.scratch_src[idx] = 0
    call_expansion(dest(s), s.scratch_dest, x)
end

## Properties

isreal(op::OperatedSet) = isreal(op)

for op in (:is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
    @eval $op(op::OperatedSet) = false
end

derivative(s::FunctionSet; options...) = OperatedSet(differentiation_operator(s; options...))
