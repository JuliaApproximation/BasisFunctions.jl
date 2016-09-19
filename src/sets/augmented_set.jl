# augmented_set.jl


"""
An AugmentedSet represents some function f(x) times an existing set.
"""
immutable AugmentedSet{S,F,N,T} <: FunctionSet{N,T}
    set     ::  S
    fun     ::  F
end

AugmentedSet{N,T}(s::FunctionSet{N,T}, fun) = AugmentedSet{typeof(s),typeof(fun),N,T}(s, fun)

name(s::AugmentedSet) = _name(s, set(s), fun(s))
_name(s::AugmentedSet, set, fun::AbstractFunction) = "An augmented set based on " * name(set)
_name(s::AugmentedSet, set, fun::Function) = name(fun) * " * " * name(set)

set(s::AugmentedSet) = s.set
fun(s::AugmentedSet) = s.fun

resize(s::AugmentedSet, n) = AugmentedSet(resize(set(s), n), fun(s))

promote_eltype{S,F,N,T,T2}(s::AugmentedSet{S,F,N,T}, ::Type{T2}) = AugmentedSet(promote_eltype(s.set, T2), s.fun)

# Method delegation
for op in (:length, :size, :left, :right)
    @eval $op(s::AugmentedSet) = $op(s.set)
end

for op in (:left, :right)
    @eval $op(s::AugmentedSet, i) = $op(s.set, i)
end

# Delegation of properties
for op in (:is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
    @eval $op(s::AugmentedSet) = $op(set(s))
end

isreal(s::AugmentedSet) = _isreal(s, set(s), fun(s))
_isreal(s::AugmentedSet, set, fun::AbstractFunction) = isreal(set) && isreal(fun)
_isreal(s::AugmentedSet, set, fun::Function) = isreal(set)


call_element(s::AugmentedSet, i, x::Number) = s.fun(x) * call_set(s.set, i, x)
call_element(s::AugmentedSet, i, x::Vec{2}) = s.fun(x[1], x[2]) * call_set(s.set, i, x)
call_element(s::AugmentedSet, i, x::Vec{3}) = s.fun(x[1], x[2], x[3]) * call_set(s.set, i, x)
call_element(s::AugmentedSet, i, x::Vec{4}) = s.fun(x[1], x[2], x[3], x[4]) * call_set(s.set, i, x)

# You can create an AugmentedSet by multiplying a function with a set, using
# left multiplication.
# We support any Julia function:
(*)(f::Function, s::FunctionSet) = AugmentedSet(s, f)
# and our own functors:
(*)(f::AbstractFunction, s::FunctionSet) = AugmentedSet(s, f)

zeros(ELT::Type, s::AugmentedSet) = zeros(ELT, set(s))

eachindex(s::AugmentedSet) = eachindex(set(s))

native_index(s::AugmentedSet, idx) = native_index(set(s), idx)

linear_index(s::AugmentedSet, idxn) = linear_index(set(s), idxn)

linearize_coefficients!(s::AugmentedSet, coef_linear, coef_native) =
    linearize_coefficients!(set(s), coef_linear, coef_native)

delinearize_coefficients!(s::AugmentedSet, coef_native, coef_linear) =
    delinearize_coefficients!(set(s), coef_native, coef_linear)

approximate_native_size(s::AugmentedSet, size_l) = approximate_native_size(set(s), size_l)

linear_size(s::AugmentedSet, size_n) = linear_size(set(s), size_n)

approx_length(s::AugmentedSet, n) = approx_length(set(s), n)

##################
# Set properties
##################

#for op in [:has_derivative, :has_extension, :has_transform, :has_grid]
for op in [:has_derivative, :has_extension, :has_grid]
    @eval $op(s::AugmentedSet) = $op(set(s))
end

# For now, until transforms are working
has_transform(s::AugmentedSet) = false

extension_size(s::AugmentedSet) = extension_size(set(s))

grid(s::AugmentedSet) = grid(set(s))

for op in [:extension_operator, :restriction_operator]
    @eval $op{F,S1,S2}(s1::AugmentedSet{S1,F}, s2::AugmentedSet{S2,F}; options...) =
        wrap_operator(s1, s2, $op(set(s1), set(s2); options...))
end

function derivative_set(src::AugmentedSet, order)
    @assert order == 1

    s = set(src)
    f = fun(src)
    f_prime = derivative(f)
    s_prime = derivative_set(s, order)
    (f_prime * s) ⊕ (f * s_prime)
end

# Assume order = 1...
function differentiation_operator(s1::AugmentedSet, s2::MultiSet, order; options...)
    @assert order == 1
    @assert s2 == derivative_set(s1, order)

    I = IdentityOperator(s1, element(s2, 1))
    D = differentiation_operator(set(s1))
    DW = wrap_operator(s1, element(s2, 2), D)
    block_column_operator([I,DW])
end

# We make a transform between two augmented sets: if the underlying transform
# maps between functions ϕ_i(x) and values v_j, then the augmented variant
# maps between f(x)ϕ_i and values f(x_j)v_j. The transform is unchanged.
transform_set(s::AugmentedSet) = AugmentedSet(transform_set(set(s)), fun(s))

transform_operator(s1::AugmentedSet, s2::AugmentedSet; options...) =
    wrap_operator(s1, s2, transform_operator(set(s1), set(s2); options...))
