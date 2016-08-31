# mappedsets.jl

# An AbstractMappedSet collects all sets that are defined in terms of another set through
# a mapping of the form y = map(x).
abstract AbstractMappedSet{S,N,T} <: FunctionSet{N,T}

set(s::AbstractMappedSet) = s.set

# Delegation of methods
for op in (:length, :approx_length)
    @eval $op(s::AbstractMappedSet, options...) = $op(set(s),options...)
end

# Delegation of property methods
for op in (:isreal, :is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
    @eval $op(s::AbstractMappedSet) = $op(set(s))
end

# Delegate feature methods
for op in (:has_derivative, :has_grid, :has_transform, :has_extension)
    @eval $op(s::AbstractMappedSet) = $op(set(s))
end

zeros(ELT::Type, s::AbstractMappedSet) = zeros(ELT, set(s))

native_index(s::AbstractMappedSet, idx) = native_index(set(s), idx)

linear_index(s::AbstractMappedSet, idxn) = linear_index(set(s), idxn)

linearize_coefficients!(s::AbstractMappedSet, coef_linear, coef_native) =
    linearize_coefficients!(set(s), coef_linear, coef_native)

delinearize_coefficients!(s::AbstractMappedSet, coef_native, coef_linear) =
    delinearize_coefficients!(set(s), coef_native, coef_linear)

approximate_native_size(s::AbstractMappedSet, size_l) = approximate_native_size(set(s), size_l)

linear_size(s::AbstractMappedSet, size_n) = linear_size(set(s), size_n)

approx_length(s::AbstractMappedSet, n) = approx_length(set(s), n)

for op in [:left, :right]
    @eval $op(s::AbstractMappedSet, i) = mapx(s, $op(set(s), i))
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

endpoint_type(s::LinearMappedSet) = typeof(left(s))

name(s::LinearMappedSet) = name(set(s)) * ", mapped to [ $(left(s))  ,  $(right(s)) ]"

# Ma the point x in [c,d] to a point in [a,b]
map_linear(x, a, b, c, d) = a + (x-c)/(d-c) * (b-a)

# Map the point y in [a,b] to a point in [c,d]
imap_linear(y, a, b, c, d) = map_linear(y, c, d, a, b)

mapx(s::LinearMappedSet, x) = map_linear(x, s.a, s.b, left(set(s)), right(set(s)))

imapx(s::LinearMappedSet, y) = imap_linear(y, s.a, s.b, left(set(s)), right(set(s)))

call_element(s::LinearMappedSet, idx, y) = call_set(set(s), idx, imapx(s,y))

grid(s::LinearMappedSet) = rescale(grid(set(s)), left(s), right(s))

is_compatible(a::LinearMappedSet, b::LinearMappedSet) = left(a)==left(b) && right(a)==right(b) && is_compatible(set(a),set(b))

function (*)(s1::LinearMappedSet, s2::LinearMappedSet, coef_src1, coef_src2)
    @assert is_compatible(set(s1),set(s2))
    (mset,mcoef) = (*)(set(s1),set(s2),coef_src1, coef_src2)
    (rescale(mset,left(s1),right(s1)),mcoef)
end

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
function rescale{N}(s::TensorProductSet, a::Vec{N}, b::Vec{N})
    scaled_sets = [ rescale(set(s,i), a[i], b[i]) for i in 1:N]
    tensorproduct(scaled_sets...)
end


(*){T <: Number}(a::T, s::FunctionSet1d) = rescale(s, a*left(s), a*right(s))

(+){T <: Number}(a::T, s::FunctionSet1d) = rescale(s, a+left(s), a+right(s))


# Standard Implementation

for op in (:extension_operator, :restriction_operator, :transform_operator,
           :normalization_operator)
    # We assume for simplicity without checking that the mapped sets below are compatible...
    @eval $op(s1::AbstractMappedSet, s2::AbstractMappedSet; options...) =
        wrap_operator(s1, s2, $op(set(s1), set(s2); options...) )
    @eval $op(s1::AbstractMappedSet, s2::FunctionSet; options...) =
        wrap_operator(s1, s2, $op(set(s1), s2; options...) )
    @eval $op(s1::FunctionSet, s2::AbstractMappedSet; options...) =
        wrap_operator(s1, s2, $op(s1, set(s2); options...) )
end

for op in (:interpolation_operator, :evaluation_operator, :approximation_operator,
    :normalization_operator, :transform_normalization_operator)
    @eval $op(s1::AbstractMappedSet; options...) = wrap_operator(s1, s1, $op(set(s1); options...) )
end

# Undo set mappings in a TensorProductSet
unmapset(s::FunctionSet) = s
unmapset(s::AbstractMappedSet) = set(s)
unmapset(s::TensorProductSet) = tensorproduct(map(unmapset, elements(s))...)

transform_operator_tensor(s1, s2,
    src_set1::AbstractMappedSet, src_set2::AbstractMappedSet,
    dest_set1::AbstractMappedSet, dest_set2::AbstractMappedSet; options...) =
        wrap_operator(s1, s2,
            transform_operator_tensor(unmapset(s1), unmapset(s2), set(src_set1), set(src_set2), set(dest_set1), set(dest_set2); options...) )

transform_operator_tensor(s1, s2, src_set1, src_set2,
    dest_set1::AbstractMappedSet, dest_set2::AbstractMappedSet; options...) =
        wrap_operator(s1, s2,
            transform_operator_tensor(s1, unmapset(s2), src_set1, src_set2, set(dest_set1), set(dest_set2); options...) )

transform_operator_tensor(s1, s2,
    src_set1::AbstractMappedSet, src_set2::AbstractMappedSet, dest_set1, dest_set2; options...) =
    wrap_operator(s1, s2, transform_operator_tensor(unmapset(s1), s2, set(src_set1), set(src_set2), dest_set1, dest_set2; options...))

transform_operator_tensor(s1, s2,
    src_set1::AbstractMappedSet, src_set2::AbstractMappedSet, src_set3::AbstractMappedSet,
    dest_set1::AbstractMappedSet, dest_set2::AbstractMappedSet, dest_set3::AbstractMappedSet; options...) =
        wrap_operator(s1, s2, transform_operator_tensor(unmapset(s1), unmapset(s2),
            set(src_set1), set(src_set2), set(src_set3),
            set(dest_set1), set(dest_set2), set(dest_set3); options...) )

transform_operator_tensor(s1, s2,
    src_set1, src_set2, src_set3,
    dest_set1::AbstractMappedSet, dest_set2::AbstractMappedSet, dest_set3::AbstractMappedSet; options...) =
        wrap_operator(s1, s2, transform_operator_tensor(unmapset(s1), unmapset(s2),
            src_set1, src_set2, src_set3,
            set(dest_set1), set(dest_set2), set(dest_set3); options...))

transform_operator_tensor(s1, s2,
    src_set1::AbstractMappedSet, src_set2::AbstractMappedSet, src_set3::AbstractMappedSet,
    dest_set1, dest_set2, dest_set3; options...) =
        wrap_operator(s1, s2, transform_operator_tensor(unmapset(s1), unmapset(s2),
            set(src_set1), set(src_set2), set(src_set3),
            dest_set1, dest_set2, dest_set3; options...))

function differentiation_operator(s1::LinearMappedSet, s2::LinearMappedSet, order::Int; options...)
    D = differentiation_operator(s1.set, s2.set, order; options...)
    T = promote_type(eltype(s1), endpoint_type(s1))
    S = ScalingOperator(dest(D), (T(right(s1.set)-left(s1.set))/T(right(s1)-left(s1)))^order)
    wrap_operator( rescale(src(D), left(s1), right(s1)), rescale(dest(D), left(s1), right(s1)), S*D )
end

function antidifferentiation_operator(s1::LinearMappedSet, s2::LinearMappedSet, order::Int; options...)
    D = antidifferentiation_operator(s1.set, s2.set, order; options...)
    T = promote_type(eltype(s1), endpoint_type(s1))
    S = ScalingOperator(dest(D), (T(right(s1)-left(s1))/T(right(s1.set)-left(s1.set)))^order)
    wrap_operator( rescale(src(D), left(s1), right(s1)), rescale(dest(D), left(s1), right(s1)), S*D )
end

for op in (:derivative_set, :antiderivative_set)
    @eval $op(s1::LinearMappedSet, order::Int; options...) = rescale( $op(set(s1), order; options...), left(s1), right(s1) )
end
