# mappedsets.jl

# An AbstractMappedSet collects all sets that are defined in terms of another set through
# a mapping of the form y = map(x).
abstract AbstractMappedSet{S,N,T} <: FunctionSet{N,T}

set(s::AbstractMappedSet) = s.set

# Delegation of methods
for op in (:length, :approx_length)
    @eval $op(s::AbstractMappedSet, args...) = $op(set(s),args...)
end

# Delegation of type methods
for op in (:isreal, :is_basis, :is_frame, :is_orthogonal, :is_biorthogonal, :index_dim, :eltype)
    @eval $op{S,N,T}(::Type{AbstractMappedSet{S,N,T}}) = $op(S)
#    @eval $op{S <: AbstractMappedSet}(::Type{S}) = $op(super(S))
end




# Delegate feature methods
for op in (:has_derivative, :has_grid, :has_transform, :has_extension)
    @eval $op(s::AbstractMappedSet) = $op(set(s))
end




"""
A set defined via a linear map.
"""
immutable LinearMappedSet{S,T,ELT} <: AbstractMappedSet{S,1,ELT}
    set     ::  S
    a       ::  T
    b       ::  T

    function LinearMappedSet(set::FunctionSet1d{ELT}, a::T, b::T)
        new(set, a, b)
    end
end
# The underlying set s should support left(s) and right(s).

function LinearMappedSet{T,S}(s::FunctionSet1d{T}, a::S, b::S)
    # ELT = promote_type(T,S)
    LinearMappedSet{typeof(s),S,eltype(s)}(s, a, b)
end

set(s::LinearMappedSet) = s.set

promote_eltype{S,T,ELT,ELT2}(b::LinearMappedSet{S,T,ELT}, ::Type{ELT2}) =
    LinearMappedSet(promote_eltype(b.set, ELT2), b.a, b.b)

resize(s::LinearMappedSet, n) = LinearMappedSet(resize(set(s), n), s.a, s.b)

extension_size(s::LinearMappedSet) = extension_size(set(s))

left(s::LinearMappedSet) = s.a
right(s::LinearMappedSet) = s.b

name(s::LinearMappedSet) = name(set(s)) * ", mapped to [ $(left(s))  ,  $(right(s)) ]"

# Ma the point x in [c,d] to a point in [a,b]
map_linear(x, a, b, c, d) = a + (x-c)/(d-c) * (b-a)

# Map the point y in [a,b] to a point in [c,d]
imap_linear(y, a, b, c, d) = map_linear(y, c, d, a, b)

mapx(s::LinearMappedSet, x) = map_linear(x, s.a, s.b, left(set(s)), right(set(s)))

imapx(s::LinearMappedSet, y) = imap_linear(y, s.a, s.b, left(set(s)), right(set(s)))

call_element(s::LinearMappedSet, idx, y) = call(set(s), idx, imapx(s,y))

grid(s::LinearMappedSet) = rescale(grid(set(s)), left(s), right(s))

"Rescale a function set to an interval [a,b]."
function rescale(s::FunctionSet1d, a, b)
    T = numtype(s)
    if abs(a-left(s)) < 10eps(T) && abs(b-right(s)) < 10eps(T)
        s
    else
        LinearMappedSet(s, promote(a,b)...)
    end
end

# avoid multiple linear mappings
rescale(s::LinearMappedSet, a, b) = rescale(set(s), a, b)



# Preserve tensor product structure
function rescale{TS,SN,N}(s::TensorProductSet{TS,SN,N,N}, a::Vec{N}, b::Vec{N})
    scaled_sets = [ rescale(set(s,i), a[i], b[i]) for i in 1:N]
    TensorProductSet(scaled_sets...)
end


(*){T <: Number}(a::T, s::FunctionSet1d) = rescale(s, a*left(s), a*right(s))

(+){T <: Number}(a::T, s::FunctionSet1d) = rescale(s, a+left(s), a+right(s))


# Standard Implementation

for op in (:extension_operator, :restriction_operator, :transform_operator,
           :normalization_operator)
    @eval $op(s1::AbstractMappedSet, s2::AbstractMappedSet) = WrappedOperator(s1,s2,$op(set(s1),set(s2)))
end

for op in (:extension_operator, :restriction_operator, :transform_operator,
           :normalization_operator)
    @eval $op(s1::AbstractMappedSet, s2::FunctionSet) = WrappedOperator(s1,s2,$op(set(s1),s2))
end

for op in (:extension_operator, :restriction_operator, :transform_operator,
           :normalization_operator)
    @eval $op(s1::FunctionSet, s2::AbstractMappedSet) = WrappedOperator(s1,s2,$op(s1,set(s2)))
end

for op in (:interpolation_operator, :evaluation_operator, :approximation_operator,
    :normalization_operator, :transform_normalization_operator)
    @eval $op(s1::AbstractMappedSet) = WrappedOperator(s1,s1,$op(set(s1)))
end

# Undo set mappings in a TensorProductSet
unmapset(s::FunctionSet) = s
unmapset(s::AbstractMappedSet) = set(s)
unmapset(s::TensorProductSet) = TensorProductSet(map(unmapset, sets(s))...)

transform_operator_tensor(s1, s2, src_set1::AbstractMappedSet, src_set2::AbstractMappedSet, dest_set1::AbstractMappedSet, dest_set2::AbstractMappedSet) =
    WrappedOperator(s1, s2, transform_operator_tensor(unmapset(s1), unmapset(s2), set(src_set1), set(src_set2), set(dest_set1), set(dest_set2)))

transform_operator_tensor(s1, s2, src_set1, src_set2, dest_set1::AbstractMappedSet, dest_set2::AbstractMappedSet) =
    WrappedOperator(s1, s2, transform_operator_tensor(s1, unmapset(s2), src_set1, src_set2, set(dest_set1), set(dest_set2)))

transform_operator_tensor(s1, s2, src_set1::AbstractMappedSet, src_set2::AbstractMappedSet, dest_set1, dest_set2) =
    WrappedOperator(s1, s2, transform_operator_tensor(unmapset(s1), s2, set(src_set1), set(src_set2), dest_set1, dest_set2))

transform_operator_tensor(s1, s2,
    src_set1::AbstractMappedSet, src_set2::AbstractMappedSet, src_set3::AbstractMappedSet,
    dest_set1::AbstractMappedSet, dest_set2::AbstractMappedSet, dest_set3::AbstractMappedSet) =
        WrappedOperator(s1, s2, transform_operator_tensor(unmapset(s1), unmapset(s2),
            set(src_set1), set(src_set2), set(src_set3),
            set(dest_set1), set(dest_set2), set(dest_set3)))

transform_operator_tensor(s1, s2,
    src_set1, src_set2, src_set3,
    dest_set1::AbstractMappedSet, dest_set2::AbstractMappedSet, dest_set3::AbstractMappedSet) =
        WrappedOperator(s1, s2, transform_operator_tensor(unmapset(s1), unmapset(s2),
            src_set1, src_set2, src_set3,
            set(dest_set1), set(dest_set2), set(dest_set3)))

transform_operator_tensor(s1, s2,
    src_set1::AbstractMappedSet, src_set2::AbstractMappedSet, src_set3::AbstractMappedSet,
    dest_set1, dest_set2, dest_set3) =
        WrappedOperator(s1, s2, transform_operator_tensor(unmapset(s1), unmapset(s2),
            set(src_set1), set(src_set2), set(src_set3),
            dest_set1, dest_set2, dest_set3))

transform_normalization_operator(s1::AbstractMappedSet) = WrappedOperator(s1,s1,transform_normalization_operator(set(s1)))

function differentiation_operator(s1::AbstractMappedSet, s2::AbstractMappedSet, order)
    D = differentiation_operator(s1.set, s2.set, order)
    S = ScalingOperator(dest(D),convert(eltype(s1),((right(s1.set)-left(s1.set))/(s1.b-s1.a)))^order)
    WrappedOperator(rescale(src(D),s1.a,s1.b),rescale(dest(D),s1.a,s1.b),S*D)
end

function antidifferentiation_operator(s1::AbstractMappedSet, s2::AbstractMappedSet, order)
    D = antidifferentiation_operator(s1.set, s2.set, order)
    S = ScalingOperator(dest(D),convert(eltype(s1),(s1.b-s1.a)/(right(s1.set)-left(s1.set)))^order)
    WrappedOperator(rescale(src(D),s1.a,s1.b),rescale(dest(D),s1.a,s1.b),S*D)
end
# The above definition does not work, super(S) goes straight up to FunctionSet
# Delegation of type methods
for op in (:isreal, :is_basis, :is_frame, :is_orthogonal, :is_biorthogonal, :index_dim, :eltype)
    @eval $op{S,T,ELT}(::Type{LinearMappedSet{S,T,ELT}}) = $op(S)
end

for op in (:derivative_space, :antiderivative_space)
    @eval $op(s1::AbstractMappedSet,order::Int) = rescale($op(set(s1),order),left(s1), right(s1))
end
