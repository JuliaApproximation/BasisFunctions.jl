#augmented_set.jl


"""
An AugmentedSet represents some function f(x) times an existing set.
"""
immutable AugmentedSet{S,F,T} <: FunctionSet{1,T}
    set     ::  S
    f       ::  F
end

AugmentedSet{T}(s::FunctionSet{1,T}, f::AbstractFunction) = AugmentedSet{typeof(s),typeof(f),T}(s, f)

set(s::AugmentedSet) = s.set
fun(s::AugmentedSet) = s.f

# Method delegation
for op in (:length,)
    @eval $op(s::AugmentedSet) = $op(s.set)
end

# Delegation of type methods
for op in (:is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
    @eval $op{S,F,T}(::Type{AugmentedSet{S,F,T}}) = $op(S)
end

isreal{S,F,T}(::Type{AugmentedSet{S,F,T}}) = isreal(S) & isreal(F)


call_element(b::AugmentedSet, i, x) = b.f(x) * call(b.set, i, x)

# Only left multiplication will do
(*){T}(f::AbstractFunction, b::FunctionSet{1,T}) = AugmentedSet(b, f)



"A ConcatenatedSet represents the direct sum of two one-dimensional sets."
immutable ConcatenatedSet{S1,S2,T} <: FunctionSet{1,T}
    set1    ::  S1
    set2    ::  S2

    ConcatenatedSet(s1::FunctionSet{1,T}, s2::FunctionSet{1,T}) = new(s1, s2)
end

ConcatenatedSet{T}(s1::FunctionSet{1,T}, s2::FunctionSet{1,T}) = ConcatenatedSet{typeof(s1),typeof(s2),T}(s1, s2)

⊕(s1::FunctionSet, s2::FunctionSet) = ConcatenatedSet(s1, s2)

set(b::ConcatenatedSet, i::Int) = i==1 ? b.set1 : b.set2

set1(b::ConcatenatedSet) = b.set1
set2(b::ConcatenatedSet) = b.set2

length(b::ConcatenatedSet) = length(b.set1) + length(b.set2)


# Method delegation
for op in (:has_derivative,)
    @eval $op(b::ConcatenatedSet) = $op(b.set1) & $op(b.set2)
end

# Delegation of type methods
for op in (:isreal,)
    @eval $op{S1,S2,T}(::Type{ConcatenatedSet{S1,S2,T}}) = $op(S1) & $op(S2)
end


eltype{S1,S2,T}(::Type{ConcatenatedSet{S1,S2,T}}) = promote_type(eltype(S1), eltype(S2))

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
end


function differentiation_operator(s1::ConcatenatedSet, s2::ConcatenatedSet, var, order)
    op1 = differentiation_operator(set1(s1), set1(s2), var, order)
    op2 = differentiation_operator(set2(s1), set2(s2), var, order)
    op1 ⊕ op2
end

extension_operator(s1::ConcatenatedSet, s2::ConcatenatedSet) =
    extension_operator(set1(s1), set1(s2)) ⊕ extension_operator(set2(s1), set2(s2))


# This may give some errors, because the src and dest of the extension operator are not the augmented
# sets, but the underlying sets.
extension_operator{F,S1,S2}(s1::AugmentedSet{S1,F}, s2::AugmentedSet{S2,F}) =
    extension_operator(set(s1), set(s2))

differentiation_operator(s::AugmentedSet, s2, var, order) = AugmentedSetDifferentiation(s)


"The AugmentedSetDifferentiation enables differentiation of an AugmentedSet of the
form f × S to a ConcatenatedSet of the form f' × S ⊕ f × S'."
immutable AugmentedSetDifferentiation{D,T,SRC,DEST} <: AbstractOperator{SRC,DEST}
    # The differentiation operator of the underlying set
    D_op    ::  D

    src     ::  SRC
    dest    ::  DEST

    # Reserve scratch space for storing coefficients of the concatenated sets in dest
    scratch_dest1   ::  Array{T,1}
    scratch_dest2   ::  Array{T,1}

    function AugmentedSetDifferentiation(D_op, src, dest::ConcatenatedSet)
        scratch_dest1 = Array(T, length(set1(dest)))
        scratch_dest2 = Array(T, length(set2(dest)))

        new(D_op, src, dest, scratch_dest1, scratch_dest2)
    end
end

AugmentedSetDifferentiation{D,T,SRC,DEST}(D_op::D, src::SRC, dest::DEST, ::Type{T}) = AugmentedSetDifferentiation{D,T,SRC,DEST}(D_op, src, dest)

function AugmentedSetDifferentiation(src_set::AugmentedSet)
    f = fun(src_set)
    f_prime = derivative(f)
    s = set(src_set)
    D_op = differentiation_operator(s)
    s_prime = dest(D_op)
    dest_set = (f_prime * s) ⊕ (f * s_prime)

    T = eltype(dest_set)

    AugmentedSetDifferentiation(D_op, src_set, dest_set, T)
end


function apply!(op::AugmentedSetDifferentiation, dest::ConcatenatedSet, src, coef_dest, coef_src)
    coef_dest1 = op.scratch_dest1
    coef_dest2 = op.scratch_dest2

    # The derivative of f(x) * EXPANSION = f'(x) * EXPANSION + f(x) * D(EXPANSION)
    # The first part of the result is simply the expansion given by coef_src
    coef_dest1[:] = coef_src[:]

    # The second part is the derivative of this expansion.
    apply!(op.D_op, coef_dest2, coef_src)

    # Finally, copy the results back into coef_dest
    L1 = length(set1(dest))
    L2 = length(set2(dest))
    for i in 1:L1
        coef_dest[i] = coef_dest1[i]
    end
    for i in 1:L2
        coef_dest[L1+i] = coef_dest2[i]
    end
end




"An OperatedSet represents a set that is acted on by an operator (for example differentiation)."
immutable OperatedSet{S,OP,ELT,T} <: FunctionSet{1,T}
    "The underlying function set"
    set     ::  S
    "The operator that acts on the set"
    op      ::  OP

    scratch_src  ::  Array{ELT,1}
    scratch_dest ::  Array{ELT,1}

    function OperatedSet(s::FunctionSet{1,T}, op::AbstractOperator{S})
        scratch_src = zeros(ELT, length(s))
        scratch_dest = zeros(ELT, length(dest(op)))
        new(s, op, scratch_src, scratch_dest)
    end
end


function OperatedSet{S <: FunctionSet1d,DEST}(s::S, op::AbstractOperator{S,DEST})
    ELT = promote_type(eltype(op), eltype(S))
    OperatedSet{S,typeof(op),ELT,numtype(S)}(s, op)
end

set(s::OperatedSet) = s.set

operator(s::OperatedSet) = s.op

eltype{S,OP,ELT,T}(::Type{OperatedSet{S,OP,ELT,T}}) = ELT

dest(s::OperatedSet) = dest(operator(s))

length(s::OperatedSet) = length(dest(s))


function call_element(s::OperatedSet, i, x)
    s.scratch_src[i] = 1
    apply!(s.op, s.scratch_dest, s.scratch_src)
    s.scratch_src[i] = 0
    call_expansion(dest(s), s.scratch_dest, x)
end


derivative(s::FunctionSet) = OperatedSet(s, differentiation_operator(s))


