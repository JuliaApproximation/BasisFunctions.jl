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

isreal(s::AugmentedSet) = isreal(set(s)) && isreal(fun(s))


call_element(b::AugmentedSet, i, x) = b.f(x) * call_set(b.set, i, x)

# You can create an AugmentedSet by multiplying a function with a set.
# Only left multiplication will do.
(*){T}(f::AbstractFunction, b::FunctionSet{1,T}) = AugmentedSet(b, f)

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

function transform_normalization_operator(s::AugmentedSet; options...)
    N = transform_normalization_operator(set(s); options...)
    f = fun(s)
    D = DiagonalOperator([1/f(x) for x in grid(s)])
    D*N
end
