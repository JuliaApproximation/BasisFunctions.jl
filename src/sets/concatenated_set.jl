# concatenated_set.jl

"A ConcatenatedSet represents the direct sum of two one-dimensional sets."
immutable ConcatenatedSet{S1,S2,T} <: FunctionSet{1,T}
    set1    ::  S1
    set2    ::  S2

    ConcatenatedSet(s1::FunctionSet{1,T}, s2::FunctionSet{1,T}) = new(s1,s2)
end

ConcatenatedSet{T}(s1::FunctionSet{1,T}, s2::FunctionSet{1,T}, ::Type{T}) = ConcatenatedSet{typeof(s1),typeof(s2),T}(s1, s2)

ConcatenatedSet{T1,T2,S}(s1::FunctionSet{1,T1}, s2::FunctionSet{1,T2}, ::Type{S} = promote_type(T1,T2)) =
    ConcatenatedSet(promote_eltype(s1,S), promote_eltype(s2,S))

⊕(s1::FunctionSet, s2::FunctionSet) = ConcatenatedSet(s1, s2)

name(s::ConcatenatedSet) = "The direct sum of " * name(s.set1) * " and " * name(s.set2)

set(b::ConcatenatedSet, i::Int) = i==1 ? b.set1 : b.set2

set1(b::ConcatenatedSet) = b.set1
set2(b::ConcatenatedSet) = b.set2

length(b::ConcatenatedSet) = length(b.set1) + length(b.set2)

resize(s::ConcatenatedSet, n::Int) = ConcatenatedSet(resize(set1(s), n-(n>>1)), resize(set2(s), n>>1))

resize(s::ConcatenatedSet, n::NTuple{2,Int}) = ConcatenatedSet(resize(set1(s), n[1]), resize(set2(s), n[2]))

promote_eltype{S1,S2,T,S}(s::ConcatenatedSet{S1,S2,T}, ::Type{S}) = ConcatenatedSet(s.set1, s.set2, S)

# Method delegation
for op in (:has_derivative, :has_extension)
    @eval $op(b::ConcatenatedSet) = $op(b.set1) & $op(b.set2)
end

for op in (:left, :right)
    @eval $op(b::ConcatenatedSet) = $op(b.set1)
end

## Traits

for op in (:isreal,)
    @eval $op{S1,S2,T}(::Type{ConcatenatedSet{S1,S2,T}}) = $op(S1) & $op(S2)
end

# The following traits we can not decide in general. We call a _concat_trait function that can be
# defined by the user for specific combinations of sets.
is_basis{S1,S2,T}(::Type{ConcatenatedSet{S1,S2,T}}) = _concat_is_basis(S1, S2)
_concat_is_basis(S1,S2) = False

is_frame{S1,S2,T}(::Type{ConcatenatedSet{S1,S2,T}}) = _concat_is_frame(S1, S2)
_concat_is_frame(S1,S2) = False

is_orthogonal{S1,S2,T}(::Type{ConcatenatedSet{S1,S2,T}}) = _concat_is_orthogonal(S1, S2)
_concat_is_orthogonal(S1,S2) = False

is_biorthogonal{S1,S2,T}(::Type{ConcatenatedSet{S1,S2,T}}) = _concat_is_biorthogonal(S1, S2)
_concat_is_biorthogonal(S1,S2) = False


call_element(b::ConcatenatedSet, i, x) = i <= length(b.set1) ? call(b.set1, i, x) : call(b.set2, i-length(b.set1), x)



"A ConcatenatedOperator is the direct sum of two operators, and can be applied to concatenated sets."
immutable ConcatenatedOperator{OP1,OP2,T,SRC,DEST} <: AbstractOperator{SRC,DEST}
    op1     ::  OP1
    op2     ::  OP2
    src     ::  SRC
    dest    ::  DEST

    # Reserve scratch space for copying source and destination of both operators to an array
    # of the right size, for use when applying the concatenated operator in terms of op1 and op2.
    scratch_src1    ::  Array{T,1}
    scratch_dest1   ::  Array{T,1}
    scratch_src2    ::  Array{T,1}
    scratch_dest2   ::  Array{T,1}

    function ConcatenatedOperator(op1, op2, src_set, dest_set)
        scratch_src1  = Array(T, length(src(op1)))
        scratch_dest1 = Array(T, length(dest(op1)))
        scratch_src2  = Array(T, length(src(op2)))
        scratch_dest2 = Array(T, length(dest(op2)))
        new(op1, op2, src_set, dest_set, scratch_src1, scratch_dest1, scratch_src2, scratch_dest2)
    end
end

function ConcatenatedOperator(op1::AbstractOperator, op2::AbstractOperator)
    op_src = ConcatenatedSet(src(op1), src(op2))
    op_dest = ConcatenatedSet(dest(op1), dest(op2))
    T = promote_type(eltype(op1), eltype(op2))
    ConcatenatedOperator{typeof(op1), typeof(op2), T, typeof(op_src), typeof(op_dest)}(op1, op2, op_src, op_dest)
end


⊕(op1::AbstractOperator, op2::AbstractOperator) = ConcatenatedOperator(op1, op2)


function apply!(op::ConcatenatedOperator, dest::ConcatenatedSet, src::ConcatenatedSet, coef_dest, coef_src)
    coef_src1 = op.scratch_src1
    coef_src2 = op.scratch_src2
    coef_dest1 = op.scratch_dest1
    coef_dest2 = op.scratch_dest2

    # First copy the long vector coef_src and coef_dest to the vectors of the right size of the individual operators
    L1 = length(set1(src))
    L2 = length(set2(src))
    for i in 1:L1
        coef_src1[i] = coef_src[i]
    end
    for i in 1:L2
        coef_src2[i] = coef_src[L1+i]
    end

    # Next, apply the two operators
    apply!(op.op1, coef_dest1, coef_src1)
    apply!(op.op2, coef_dest2, coef_src2)

    # Finally, copy the results back into coef_dest
    L1 = length(set1(dest))
    L2 = length(set2(dest))
    for i in 1:L1
        coef_dest[i] = coef_dest1[i]
    end
    for i in 1:L2
        coef_dest[L1+i] = coef_dest2[i]
    end
    coef_dest
end

derivative_set(s::ConcatenatedSet, order) =
    ConcatenatedSet(derivative_set(set1(s), order), derivative_set(set2(s), order))


function differentiation_operator(s1::ConcatenatedSet, s2::ConcatenatedSet, order; options...)
    op1 = differentiation_operator(set1(s1), set1(s2), order; options...)
    op2 = differentiation_operator(set2(s1), set2(s2), order; options...)
    op1 ⊕ op2
end


extension_size(b::ConcatenatedSet) = (extension_size(set1(b)), extension_size(set2(b)))

extension_operator(s1::ConcatenatedSet, s2::ConcatenatedSet; options...) =
    extension_operator(set1(s1), set1(s2); options...) ⊕ extension_operator(set2(s1), set2(s2); options...)

restriction_operator(s1::ConcatenatedSet, s2::ConcatenatedSet; options...) =
    restriction_operator(set1(s1), set1(s2); options...) ⊕ restriction_operator(set2(s1), set2(s2); options...)


"A HorizontalConcatenatedOperator maps a ConcatenatedSet to a common destination set."
immutable HorizontalConcatenatedOperator{OP1,OP2,T,SRC,DEST} <: AbstractOperator{SRC,DEST}
    op1     ::  OP1
    op2     ::  OP2
    src     ::  SRC
    dest    ::  DEST

    # Reserve scratch space for copying source and destination of both operators to an array
    # of the right size, for use when applying the concatenated operator in terms of op1 and op2.
    scratch_src1    ::  Array{T,1}
    scratch_dest1   ::  Array{T,1}
    scratch_src2    ::  Array{T,1}
    scratch_dest2   ::  Array{T,1}

    function HorizontalConcatenatedOperator(op1, op2, src_set, dest_set)
        scratch_src1  = Array(T, length(src(op1)))
        scratch_dest1 = Array(T, length(dest(op1)))
        scratch_src2  = Array(T, length(src(op2)))
        scratch_dest2 = Array(T, length(dest(op2)))
        new(op1, op2, src_set, dest_set, scratch_src1, scratch_dest1, scratch_src2, scratch_dest2)
    end
end

function HorizontalConcatenatedOperator(op1::AbstractOperator, op2::AbstractOperator)
    @assert dest(op1) == dest(op2)
    op_src = ConcatenatedSet(src(op1), src(op2))
    T = promote_type(eltype(op1), eltype(op2))
    HorizontalConcatenatedOperator{typeof(op1), typeof(op2), T, typeof(op_src), typeof(dest(op1))}(op1, op2, op_src, dest(op1))
end


hcat(op1::AbstractOperator, op2::AbstractOperator) = HorizontalConcatenatedOperator(op1, op2)


function apply!(op::HorizontalConcatenatedOperator, dest::FunctionSet, src::ConcatenatedSet, coef_dest, coef_src)
    coef_src1 = op.scratch_src1
    coef_src2 = op.scratch_src2
    coef_dest1 = op.scratch_dest1
    coef_dest2 = op.scratch_dest2

    # First copy the long vector coef_src and coef_dest to the vectors of the right size of the individual operators
    L1 = length(set1(src))
    L2 = length(set2(src))
    for i in 1:L1
        coef_src1[i] = coef_src[i]
    end
    for i in 1:L2
        coef_src2[i] = coef_src[L1+i]
    end

    # Next, apply the two operators
    apply!(op.op1, coef_dest1, coef_src1)
    apply!(op.op2, coef_dest2, coef_src2)

    # Finally, sum the results into coef_dest
    for i in 1:length(coef_dest)
        coef_dest[i] = coef_dest1[i]+coef_dest2[i]
    end
    coef_dest
end


"A VerticalConcatenatedOperator an operator that maps a set to a ConcatenedSet"
immutable VerticalConcatenatedOperator{OP1,OP2,T,SRC,DEST} <: AbstractOperator{SRC,DEST}
    op1     ::  OP1
    op2     ::  OP2
    src     ::  SRC
    dest    ::  DEST

    # Reserve scratch space for copying source and destination of both operators to an array
    # of the right size, for use when applying the concatenated operator in terms of op1 and op2.
    scratch_src1    ::  Array{T,1}
    scratch_dest1   ::  Array{T,1}
    scratch_src2    ::  Array{T,1}
    scratch_dest2   ::  Array{T,1}

    function VerticalConcatenatedOperator(op1, op2, src_set, dest_set)
        scratch_src1  = Array(T, length(src(op1)))
        scratch_dest1 = Array(T, length(dest(op1)))
        scratch_src2  = Array(T, length(src(op2)))
        scratch_dest2 = Array(T, length(dest(op2)))
        new(op1, op2, src_set, dest_set, scratch_src1, scratch_dest1, scratch_src2, scratch_dest2)
    end
end

function VerticalConcatenatedOperator(op1::AbstractOperator, op2::AbstractOperator)
    @assert src(op1) == src(op2)
    op_dest = ConcatenatedSet(dest(op1), dest(op2))
    T = promote_type(eltype(op1), eltype(op2))
    VerticalConcatenatedOperator{typeof(op1), typeof(op2), T, typeof(src(op1)), typeof(op_dest)}(op1, op2, src(op1), op_dest)
end


vcat(op1::AbstractOperator, op2::AbstractOperator) = VerticalConcatenatedOperator(op1, op2)


function apply!(op::VerticalConcatenatedOperator, dest::ConcatenatedSet, src::FunctionSet, coef_dest, coef_src)
    coef_src1 = op.scratch_src1
    coef_src2 = op.scratch_src2
    coef_dest1 = op.scratch_dest1
    coef_dest2 = op.scratch_dest2

    # Apply the two operators
    apply!(op.op1, coef_dest1, coef_src)
    apply!(op.op2, coef_dest2, coef_src)

    # Finally, copy the results back into coef_dest
    L1 = length(set1(dest))
    L2 = length(set2(dest))
    for i in 1:L1
        coef_dest[i] = coef_dest1[i]
    end
    for i in 1:L2
        coef_dest[L1+i] = coef_dest2[i]
    end
    coef_dest
end
