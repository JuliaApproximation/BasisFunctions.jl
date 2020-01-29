

# op1 is the first operator to be applied to a vector.
# compose_and_simplify(op1, op2, op3, ops...) = Base.afoldl(compose_and_simplify, compose_and_simplify(compose_and_simplify(op1,op2),op3), ops...)
compose_and_simplify(op1::AbstractOperator, op2::AbstractOperator, op3::AbstractOperator, ops::AbstractOperator...) =
    _compose_and_simplify(op1::AbstractOperator, op2::AbstractOperator, op3::AbstractOperator, ops...)

function _compose_and_simplify(ops::AbstractOperator...)
    ops = collect(ops)
    unchanged = false
    while !unchanged
        unchanged = true
        i = 1
        if length(ops) == 1
            break;
        end
        while i <= length(ops)-1
            changed, result = _compose_and_simplify(ops[i], ops[i+1])
            unchanged = !changed
            if changed
                if result isa AbstractOperator
                    ops[i] = result
                    deleteat!(ops, i+1)
                else
                    ops[i] = result[1]
                    ops[i+1] = result[2]
                end
            end
            i += 1
        end
    end
    tuple(ops...)
end

@inline cas_src(op1, op2) = src(op1)
@inline cas_dest(op1, op2) = dest(op2)

@inline assert_compatible_compose_src_dest(op1, op2) = iscompatible(dest(op1), src(op2))

iterate(x::AbstractOperator) = (x, nothing)
iterate(x::AbstractOperator, ::Any) = nothing


function compose_and_simplify(op1::DictionaryOperator, op2::DictionaryOperator)
    assert_compatible_compose_src_dest(op1, op2)
    result = unsafe_compose_and_simplify(op1, op2)
    if result isa Tuple && result[1] isa Bool
        result[2]
    else
        tuple(result...)
    end
end

function _compose_and_simplify(op1::DictionaryOperator, op2::DictionaryOperator)
    assert_compatible_compose_src_dest(op1, op2)
    result = unsafe_compose_and_simplify(op1, op2)
    if result isa Tuple && result[1] isa Bool
        false, result[2]
    else
        true, result
    end
end


unsafe_compose_and_simplify(op1, op2) =
    unsafe_compose_and_simplify1(op1, op2)

unsafe_compose_and_simplify1(op1::DictionaryOperator, op2) =
        unsafe_compose_and_simplify2(op1, op2)

unsafe_compose_and_simplify2(op1, op2::DictionaryOperator) =
        default_unsafe_compose_and_simplify(op1, op2)

@inline default_unsafe_compose_and_simplify(op1, op2) = (false, (op1, op2))

unsafe_compose_and_simplify(op1::IdentityOperator, op2::IdentityOperator) = op1
unsafe_compose_and_simplify1(op1::IdentityOperator, op2) = op2
unsafe_compose_and_simplify2(op1, op2::IdentityOperator) = op1

# Move scalars to last place
unsafe_compose_and_simplify1(op1::ScalingOperator{T}, op2) where T =  op2, ScalingOperator(dest(op2), op1.A; T=T)

# Combine scalars to one / not type stable
function unsafe_compose_and_simplify(op1::ScalingOperator, op2::ScalingOperator)
    A = op2.A*op1.A
    if A.λ ≈ 1 && iscompatible(cas_src(op1,op2), cas_dest(op1,op2))
        IdentityOperator(cas_src(op1,op2), cas_dest(op1,op2); T = promote_type(eltype(op1),eltype(op2)))
    else
        ScalingOperator(cas_src(op1,op2), cas_dest(op1,op2), A; T = promote_type(eltype(op1),eltype(op2)))
    end
end

for (OP) in (:DiagonalOperator, :CirculantOperator)
    @eval begin
        unsafe_compose_and_simplify(op1::$OP, op2::$OP) = ArrayOperator(_checked_mul(unsafe_matrix(op2), unsafe_matrix(op1)), cas_src(op1,op2), cas_dest(op1,op2))
    end
end

function unsafe_compose_and_simplify(op1::IndexExtensionOperator, op2::IndexRestrictionOperator)
    if subindices(op1) == subindices(op2)
        IdentityOperator(cas_src(op1,op2), cas_dest(op1,op2))
    else
        default_unsafe_compose_and_simplify(op1, op2)
    end
end

unsafe_compose_and_simplify(op1::DiagonalOperator, op2::ScalingOperator) =
    DiagonalOperator(cas_src(op1,op2),cas_dest(op1,op2),Diagonal(diag(op1)*scalar(op2)))

unsafe_compose_and_simplify(op1::ScalingOperator, op2::DiagonalOperator) =
    DiagonalOperator(cas_src(op1,op2),cas_dest(op1,op2),Diagonal(diag(op2)*scalar(op1)))
