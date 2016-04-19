# augmented_set.jl


"""
An AugmentedSet represents some function f(x) times an existing set.
"""
immutable AugmentedSet{S,F,T} <: FunctionSet{1,T}
    set     ::  S
    f       ::  F
end

AugmentedSet{T}(s::FunctionSet{1,T}, f::AbstractFunction) = AugmentedSet{typeof(s),typeof(f),T}(s, f)

name(s::AugmentedSet) = name(fun(s)) * " * " * name(set(s))

set(s::AugmentedSet) = s.set
fun(s::AugmentedSet) = s.f

resize(s::AugmentedSet, n) = AugmentedSet(resize(set(s), n), fun(s))

promote_eltype{S,F,T,T2}(s::AugmentedSet{S,F,T}, ::Type{T2}) = AugmentedSet(promote_eltype(s.set, T2), s.f)

# Method delegation
for op in (:length, :left, :right)
    @eval $op(s::AugmentedSet) = $op(s.set)
end

# Delegation of properties
for op in (:is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
    @eval $op(s::AugmentedSet) = $op(set(s))
end

isreal(s::AugmentedSet) = isreal(set(s)) && isreal(fun(s))


call_element(b::AugmentedSet, i, x) = b.f(x) * call(b.set, i, x)

# Only left multiplication will do
(*){T}(f::AbstractFunction, b::FunctionSet{1,T}) = AugmentedSet(b, f)


extension_operator{F,S1,S2}(s1::AugmentedSet{S1,F}, s2::AugmentedSet{S2,F}; options...) =
    WrappedOperator(s1, s2, extension_operator(set(s1), set(s2); options...))



"The AugmentedSetDifferentiation enables differentiation of an AugmentedSet of the
form f × S to a ConcatenatedSet of the form f' × S ⊕ f × S'."
immutable AugmentedSetDifferentiation{D,ELT} <: AbstractOperator{ELT}
    # The differentiation operator of the underlying set
    D_op    ::  D

    src     ::  FunctionSet
    dest    ::  FunctionSet

    # Reserve scratch space for storing coefficients of the concatenated sets in dest
    scratch_dest1   ::  Array{ELT,1}
    scratch_dest2   ::  Array{ELT,1}

    function AugmentedSetDifferentiation(D_op, src, dest::ConcatenatedSet)
        scratch_dest1 = Array(ELT, length(set1(dest)))
        scratch_dest2 = Array(ELT, length(set2(dest)))

        new(D_op, src, dest, scratch_dest1, scratch_dest2)
    end
end

AugmentedSetDifferentiation(diff_op::AbstractOperator, src::FunctionSet, dest::FunctionSet) =
    AugmentedSetDifferentiation{typeof(diff_op),op_eltype(src,dest)}(D_op, src, dest)

function derivative_set(src::AugmentedSet, order)
    @assert order == 1

    s = set(src)
    f = fun(src)
    f_prime = derivative(f)
    s_prime = derivative_set(s)
    (f_prime * s) ⊕ (f * s_prime)
end

function AugmentedSetDifferentiation(src::AugmentedSet)
    diff_op = differentiation_operator(set(src))
    AugmentedSetDifferentiation(diff_op, src, derivative_set(src))
end


function apply!(op::AugmentedSetDifferentiation, coef_dest, coef_src)
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
    coef_dest
end


# Assume order = 1...
function differentiation_operator(s1::AugmentedSet, s2::FunctionSet, order; options...)
    @assert order == 1
    result = AugmentedSetDifferentiation(s1)
    @assert dest(result) == s2
    result
end
