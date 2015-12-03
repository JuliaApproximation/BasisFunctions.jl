#augmented_set.jl

"AbstractFunction is the supertype of all functors."
abstract AbstractFunction

isreal{F <: AbstractFunction}(::Type{F}) = True
isreal(f::AbstractFunction) = isreal(typeof(f))

"The function x^α"
immutable PowerFunction <: AbstractFunction
    α   ::  Int
end

call(f::PowerFunction, x) = x^(f.α)


"Functor for the logarithmic function."
immutable Log <: AbstractFunction
end

call(f::Log, x) = log(x)

derivative(f::Log) = PowerFunction(-1)


"Functor for the exponential function."
immutable Exp <: AbstractFunction
end

call(f::Exp, x) = exp(x)

derivative(f::Exp) = f


"Functor for the cosine function."
immutable Cos <: AbstractFunction
end

call(f::Cos, x) = cos(x)


"Functor for the sine function."
immutable Sin <: AbstractFunction
end

call(f::Sin, x) = sin(x)

derivative(f::Cos) = -1 * Sin()

derivative(f::Sin) = Cos()


"A ScaledFunction represents a scalar times a function."
immutable ScaledFunction{F <: AbstractFunction,T} <: AbstractFunction
    f   ::  F
    a   ::  T
end

scalar(f::ScaledFunction) = f.a

*(a::Number, f::AbstractFunction) = ScaledFunction(f, a)
*(a::Number, f::ScaledFunction) = ScaledFunction(f.f, a*f.a)

call(f::ScaledFunction, x) = f.a * f.f(x)

derivative(f::ScaledFunction) = f.a * derivative(f.f)

isreal{F,T}(::Type{ScaledFunction{F,T}}) = isreal(F) & isreal(T)



"A DilatedFunction represents f(a*x) where a is a scalar."
immutable DilatedFunction{F,T} <: AbstractFunction
    f   ::  F
    a   ::  T
end

scalar(f::DilatedFunction) = f.a

call(f::DilatedFunction, x) = f.f(f.a*x)

derivative(f::DilatedFunction) = f.a * DilatedFunction(derivative(f.f), f.a)

isreal{F,T}(::Type{DilatedFunction{F,T}}) = isreal(F) & isreal(T)


"A CombinedFunction represents f op g, where op can be any binary operator."
immutable CombinedFunction{F,G,OP} <: AbstractFunction
    f   ::  F
    g   ::  G
    # op is supposed to be a type that can be called with two arguments
    # Such as: AddFun, MulFun, etc. -> see base/functors.jl
    op  ::  OP
end

+(f::AbstractFunction, g::AbstractFunction) = CombinedFunction(f, g, Base.AddFun())
-(f::AbstractFunction, g::AbstractFunction) = CombinedFunction(f, g, Base.SubFun())
*(f::AbstractFunction, g::AbstractFunction) = CombinedFunction(f, g, Base.MulFun())

fun1(f::CombinedFunction) = f.f
fun2(f::CombinedFunction) = f.g
operator(f::CombinedFunction) = f.op

call(f::CombinedFunction, x) = f.op(f.f(x), f.g(x))

derivative(f::CombinedFunction) = derivative_op(f, fun1(f), fun2(f), operator(f))

derivative_op(::CombinedFunction, f, g, ::Base.AddFun) = derivative(f) + derivative(g)

derivative_op(::CombinedFunction, f, g, ::Base.SubFun) = derivative(f) - derivative(g)

# The chain rule
derivative_op(::CombinedFunction, f, g, ::Base.MulFun) = derivative(f) * g + f * derivative(g)


isreal{F,G,OP}(::Type{CombinedFunction{F,G,OP}}) = isreal(F) & isreal(G)


"A CompositeFunction represents f(g(x))."
immutable CompositeFunction{F,G} <: AbstractFunction
    f   ::  F
    g   ::  G
end

∘(f::AbstractFunction, g::AbstractFunction) = CompositeFunction(f, g)

call(f::CompositeFunction, x) = f.f(f.g(x))

derivative(f::CompositeFunction) = (derivative(f.f) ∘ f.g) * derivative(f.g)

isreal{F,G}(::Type{CompositeFunction{F,G}}) = isreal(F) & isreal(G)


"The identity function"
immutable IdentityFunction <: AbstractFunction
end

call(f::IdentityFunction, x) = x

derivative(f::IdentityFunction) = ConstantFunction()

^(f::IdentityFunction, α::Int) = PowerFunction(α)

"The constant function 1"
immutable ConstantFunction <: AbstractFunction
end

call(f::ConstantFunction, x) = one(x)

x = IdentityFunction()

*(f::ConstantFunction, g::ConstantFunction) = f
*(f::AbstractFunction, g::ConstantFunction) = f
*(f::ConstantFunction, g::AbstractFunction) = g

cos(f::AbstractFunction) = CompositeFunction(Cos(), f)
sin(f::AbstractFunction) = CompositeFunction(Sin(), f)
exp(f::AbstractFunction) = CompositeFunction(Exp(), f)
log(f::AbstractFunction) = CompositeFunction(Log(), f)



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
immutable ConcatenatedSet{S1 <: FunctionSet,S2 <: FunctionSet,T} <: FunctionSet{1,T}
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

    function ConcatenatedOperator(op1, op2, src, dest)
        scratch_src1  = Array(T, length(src(op1)))
        scratch_dest1 = Array(T, length(dest(op1)))
        scratch_src2  = Array(T, length(src(op2)))
        scratch_dest2 = Array(T, length(dest(op2)))
        new(op1, op2, src, dest, scratch_src1, scratch_dest1, scratch_src2, scratch_dest2)
    end
end

function ConcatenatedOperator(op1::AbstractOperator, op2::AbstractOperator)
    src = ConcatenatedSet(src(op1), src(op2))
    dest = ConcatenatedSet(dest(op1), dest(op2))
    T = promote_type(eltype(op1), eltype(op2))
    ConcatenatedOperator{typeof(op1), typeof(op2), T, typeof(src), typeof(dest)}(op1, op2, src, dest)
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
        coef_dest[i] = coef_src1[i]
    end
    for i in 1:L2
        coef_dest[L1+i] = coef_src2[i]
    end
end


function differentiation_operator(s1::ConcatenatedSet, s2::ConcatenatedSet, var, order)
    op1 = differentiation_operator(set1(s1), set1(s2), var, order)
    op2 = differentiation_operator(set2(s1), set2(s2), var, order)
    op1 ⊕ op2
end


differentiation_operator(s::AugmentedSet) = AugmentedSetDifferentiation(s)


immutable AugmentedSetDifferentiation{D,T,SRC,DEST} <: AbstractOperator{SRC,DEST}
    # The differentiation operator of the underlying set
    D_op    ::  D

    src     ::  SRC
    dest    ::  DEST

    # Reserve scratch space for storing coefficients of the concatenated sets in dest
    scratch_dest1   ::  Array{T,1}
    scratch_dest2   ::  Array{T,1}

    function AugmentedSetDifferentiation(D_op, src, dest)
        scratch_dest1 = Array(T, length(set1(dest)))
        scratch_dest2 = Array(T, length(set2(dest)))

        new(D_op, src, dest, scratch_dest1, scratch_dest2)
    end
end

AugmentedSetDifferentiation{D,T,SRC,DEST}(D_op::D, src::SRC, dest::DEST, ::Type{T}) = AugmentedSetDifferentiation{D,T,SRC,DEST}(D_op, src, dest)

function AugmentedSetDifferentiation(src::AugmentedSet)
    f = fun(src)
    fd = derivative(f)
    s = set(src)
    dest = (fd * s) ⊕ (f * s)

    D_op = differentiation_operator(s)
    T = eltype(s)

    AugmentedSetDifferentiation(D_op, src, dest, T)
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



