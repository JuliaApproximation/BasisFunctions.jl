
"""
A TensorProductOperator represents the tensor product of other operators.

struct TensorProductOperator{T} <: DictionaryOperator{T}
"""
struct TensorProductOperator{T} <: DictionaryOperator{T}
    src             ::  Dictionary
    dest            ::  Dictionary
    operators
    scratch
    src_scratch
    dest_scratch
end

# Generic functions for composite types:
components(op::TensorProductOperator) = op.operators
component(op::TensorProductOperator, j::Int) = op.operators[j]

factors(op::TensorProductOperator) = components(op)

TensorProductOperator(operators::AbstractOperator...; T=promote_type(map(eltype, operators)...)) =
    TensorProductOperator{T}(operators...)

function TensorProductOperator{T}(operators::AbstractOperator...) where {T}
    tp_src = tensorproduct(map(src, operators)...)
    tp_dest = tensorproduct(map(dest, operators)...)
    TensorProductOperator{T}(tp_src, tp_dest, operators...)
end

function TensorProductOperator{T}(tp_src::Dictionary, tp_dest::Dictionary, operators::AbstractOperator...) where {T}
    L = length(operators)
    # Scratch contains matrices of sufficient size to hold intermediate results
    # in the application of the tensor product operator.
    # Example, for L=3 scratch is a length (L-1)-tuple of matrices of size:
    # - [M1,N2,N3]
    # - [M1,M2,N3]
    # where operator J maps a set of length Nj to a set of length Mj.
    scratch_array = [ zeros(T, [length(dest(operators[k])) for k=1:j-1]..., [length(src(operators[k])) for k=j:L]...) for j=2:L]
    scratch = (scratch_array...,)

    # scr_scratch and dest_scratch are tuples of length len that contain preallocated
    # storage to hold a vector for source and destination for each operator
    src_scratch_array = [zeros(T, src(operators[j])) for j=1:L]
    src_scratch = (src_scratch_array...,)
    dest_scratch_array = [zeros(T, dest(operators[j])) for j=1:L]
    dest_scratch = (dest_scratch_array...,)
    TensorProductOperator{T}(tp_src, tp_dest, operators, scratch, src_scratch, dest_scratch)
end



function tensorproduct(ops::IdentityOperator...)
    tp_src = tensorproduct(map(src, ops)...)
    tp_dest = tensorproduct(map(dest, ops)...)
    IdentityOperator(tp_src, tp_dest)
end

# Element-wise src and dest functions
src(op::TensorProductOperator, j::Int) = src(component(op, j))
dest(op::TensorProductOperator, j::Int) = dest(component(op, j))


unsafe_wrap_operator(dict1::TensorProductDict, dict2::TensorProductDict, op::TensorProductOperator{T}) where {T} =
    TensorProductOperator{T}(dict1, dict2, map(wrap_operator, components(dict1), components(dict2), components(op))...)
unsafe_wrap_operator(dict1::Dictionary, dict2::Dictionary, op::TensorProductOperator{T}) where {T} =
    TensorProductOperator{T}(dict1, dict2, op.operators, op.scratch, op.src_scratch, op.dest_scratch)


#getindex(op::TensorProductOperator, j::Int) = component(op, j)
adjoint(op::TensorProductOperator) = TensorProductOperator(map(adjoint, components(op))...)
conj(op::TensorProductOperator) = TensorProductOperator(map(conj, components(op))...)

pinv(op::TensorProductOperator) = TensorProductOperator(map(pinv, components(op))...)

inv(op::TensorProductOperator) = TensorProductOperator(map(inv, components(op))...)

isinplace(op::TensorProductOperator) = all(map(isinplace, op.operators))
isdiag(op::TensorProductOperator) = all(map(isdiag, op.operators))

isidentity(op::TensorProductOperator) = all(map(isidentity, op.operators))

# Matrix(op::TensorProductOperator) = kron((Matrix(op) for op in reverse(components(op)))...)
sparse(op::TensorProductOperator) = kron((sparse(op) for op in reverse(components(op)))...)

unsafe_wrap_operator(src, dest, op::TensorProductOperator{T}) where T =
    TensorProductOperator{T}(src, dest, op.operators, op.scratch, op.src_scratch, op.dest_scratch)

# We reshape any incoming or outgoing coefficients into an array of the right size
function apply!(op::TensorProductOperator, coef_dest, coef_src::AbstractVector)
    @warn "Reshaping input of tensor product operator from vector to tensor" maxlog=4
    apply!(op, coef_dest, reshape(coef_src, size(src(op))))
end

function apply!(op::TensorProductOperator, coef_dest::Vector, coef_src)
    @warn "Reshaping output of tensor product operator from vector to tensor" maxlog=4
    apply!(op, reshape(coef_dest, size(dest(op))), coef_src)
end

function apply!(op::TensorProductOperator, coef_dest::Vector, coef_src::AbstractVector)
    @warn "Reshaping input and output of tensor product operator from vector to tensor" maxlog=4
    apply!(op, reshape(coef_dest, size(dest(op))), reshape(coef_src, size(src(op))))
end

apply_inplace!(op::TensorProductOperator, coef_srcdest::AbstractVector) =
    apply_inplace!(op, reshape(coef_srcdest, size(src(op))))


apply!(op::TensorProductOperator, coef_dest, coef_src) =
    apply_tensor!(op, coef_dest, coef_src, op.operators, op.scratch, op.src_scratch, op.dest_scratch)

apply_inplace!(op::TensorProductOperator, coef_srcdest) =
    apply_inplace_tensor!(op, coef_srcdest, op.operators, op.src_scratch)

function apply_tensor!(op, coef_dest, coef_src, operators::Tuple{A}, scratch, src_scratch, dest_scratch) where {A}
    println("One-element TensorProductOperators should not exist!")
    apply!(operators[1], coef_dest, coef_src)
end

function apply_inplace_tensor!(op, coef_srcdest, operators::Tuple{A}, src_scratch) where {A}
    println("One-element TensorProductOperators should not exist!")
    apply!(operators[1], coef_srcdest)
end

function apply_tensor!(op, coef_dest, coef_src, operators::Tuple{A,B}, scratch, src_scratch, dest_scratch) where {A,B}
    src1 = src_scratch[1]
    src2 = src_scratch[2]
    dest1 = dest_scratch[1]
    dest2 = dest_scratch[2]
    op1 = operators[1]
    op2 = operators[2]
    intermediate = scratch[1]

    for j in eachindex(src2)
        for i in eachindex(src1)
            src1[i] = coef_src[i,j]
        end
        apply!(op1, dest1, src1)
        for i in eachindex(dest1)
            intermediate[i,j] = dest1[i]
        end
    end
    for i in eachindex(dest1)
        for j in eachindex(src2)
            src2[j] = intermediate[i,j]
        end
        apply!(op2, dest2, src2)
        for j in eachindex(dest2)
            coef_dest[i,j] = dest2[j]
        end
    end
    coef_dest
end

function apply_inplace_tensor!(op, coef_srcdest, operators::Tuple{A,B}, src_scratch) where {A,B}
    src1 = src_scratch[1]
    src2 = src_scratch[2]
    op1 = operators[1]
    op2 = operators[2]

    for j in eachindex(src2)
        for i in eachindex(src1)
            src1[i] = coef_srcdest[i,j]
        end
        apply!(op1, src1)
        for i in eachindex(src1)
            coef_srcdest[i,j] = src1[i]
        end
    end
    for i in eachindex(src1)
        for j in eachindex(src2)
            src2[j] = coef_srcdest[i,j]
        end
        apply!(op2, src2)
        for j in eachindex(src2)
            coef_srcdest[i,j] = src2[j]
        end
    end
    coef_srcdest
end

function apply_tensor!(op, coef_dest, coef_src, operators::Tuple{A,B,C}, scratch, src_scratch, dest_scratch) where {A,B,C}
    src1 = src_scratch[1]
    src2 = src_scratch[2]
    src3 = src_scratch[3]
    dest1 = dest_scratch[1]
    dest2 = dest_scratch[2]
    dest3 = dest_scratch[3]
    op1 = operators[1]
    op2 = operators[2]
    op3 = operators[3]
    intermediate1 = scratch[1]
    intermediate2 = scratch[2]

    for k in eachindex(src3)
        for j in eachindex(src2)
            for i in eachindex(src1)
                src1[i] = coef_src[i,j,k]
            end
            apply!(op1, dest1, src1)
            for i in eachindex(dest1)
                intermediate1[i,j,k] = dest1[i]
            end
        end
    end
    for k in eachindex(src3)
        for i in eachindex(dest1)
            for j in eachindex(src2)
                src2[j] = intermediate1[i,j,k]
            end
            apply!(op2, dest2, src2)
            for j in eachindex(dest2)
                intermediate2[i,j,k] = dest2[j]
            end
        end
    end
    for i in eachindex(dest1)
        for j in eachindex(dest2)
            for k in eachindex(src3)
                src3[k] = intermediate2[i,j,k]
            end
            apply!(op3, dest3, src3)
            for k in eachindex(dest3)
                coef_dest[i,j,k] = dest3[k]
            end
        end
    end
    coef_dest
end

function apply_inplace_tensor!(op, coef_srcdest, operators::Tuple{A,B,C}, src_scratch) where {A,B,C}
    src1 = src_scratch[1]
    src2 = src_scratch[2]
    src3 = src_scratch[3]
    op1 = operators[1]
    op2 = operators[2]
    op3 = operators[3]

    for k in eachindex(src3)
        for j in eachindex(src2)
            for i in eachindex(src1)
                src1[i] = coef_srcdest[i,j,k]
            end
            apply!(op1, src1)
            for i in eachindex(src1)
                coef_srcdest[i,j,k] = src1[i]
            end
        end
    end
    for k in eachindex(src3)
        for i in eachindex(src1)
            for j in eachindex(src2)
                src2[j] = coef_srcdest[i,j,k]
            end
            apply!(op2, src2)
            for j in eachindex(src2)
                coef_srcdest[i,j,k] = src2[j]
            end
        end
    end
    for i in eachindex(src1)
        for j in eachindex(src2)
            for k in eachindex(src3)
                src3[k] = coef_srcdest[i,j,k]
            end
            apply!(op3, src3)
            for k in eachindex(src3)
                coef_srcdest[i,j,k] = src3[k]
            end
        end
    end
    coef_srcdest
end

function apply_tensor!(op, coef_dest, coef_src, operators::Tuple{A,B,C,D}, scratch, src_scratch, dest_scratch) where {A,B,C,D}
    src1 = src_scratch[1]
    src2 = src_scratch[2]
    src3 = src_scratch[3]
    src4 = src_scratch[4]
    dest1 = dest_scratch[1]
    dest2 = dest_scratch[2]
    dest3 = dest_scratch[3]
    dest4 = dest_scratch[4]
    op1 = operators[1]
    op2 = operators[2]
    op3 = operators[3]
    op4 = operators[4]
    intermediate1 = scratch[1]
    intermediate2 = scratch[2]
    intermediate3 = scratch[3]

    for l in eachindex(src4)
        for k in eachindex(src3)
            for j in eachindex(src2)
                for i in eachindex(src1)
                    src1[i] = coef_src[i,j,k,l]
                end
                apply!(op1, dest1, src1)
                for i in eachindex(dest1)
                    intermediate1[i,j,k,l] = dest1[i]
                end
            end
        end
    end
    for l in eachindex(src4)
        for k in eachindex(src3)
            for i in eachindex(dest1)
                for j in eachindex(src2)
                    src2[j] = intermediate1[i,j,k,l]
                end
                apply!(op2, dest2, src2)
                for j in eachindex(dest2)
                    intermediate2[i,j,k,l] = dest2[j]
                end
            end
        end
    end
    for l in eachindex(src4)
        for i in eachindex(dest1)
            for j in eachindex(dest2)
                for k in eachindex(src3)
                    src3[k] = intermediate2[i,j,k,l]
                end
                apply!(op3, dest3, src3)
                for k in eachindex(dest3)
                    intermediate3[i,j,k,l] = dest3[k]
                end
            end
        end
    end
    for i in eachindex(dest1)
        for j in eachindex(dest2)
            for k in eachindex(dest3)
                for l in eachindex(src4)
                    src4[l] = intermediate3[i,j,k,l]
                end
                apply!(op4, dest4, src4)
                for l in eachindex(dest4)
                    coef_dest[i,j,k,l] = dest4[l]
                end
            end
        end
    end
    coef_dest
end

function apply_inplace_tensor!(op, coef_srcdest, operators::Tuple{A,B,C,D}, src_scratch) where {A,B,C,D}
    src1 = src_scratch[1]
    src2 = src_scratch[2]
    src3 = src_scratch[3]
    src4 = src_scratch[4]
    op1 = operators[1]
    op2 = operators[2]
    op3 = operators[3]
    op4 = operators[4]

    for l in eachindex(src4)
        for k in eachindex(src3)
            for j in eachindex(src2)
                for i in eachindex(src1)
                    src1[i] = coef_srcdest[i,j,k,l]
                end
                apply!(op1, src1)
                for i in eachindex(src1)
                    coef_srcdest[i,j,k,l] = src1[i]
                end
            end
        end
    end
    for l in eachindex(src4)
        for k in eachindex(src3)
            for i in eachindex(dest1)
                for j in eachindex(src2)
                    src2[j] = coef_srcdest[i,j,k,l]
                end
                apply!(op2, src2)
                for j in eachindex(src2)
                    coef_srcdest[i,j,k,l] = src2[j]
                end
            end
        end
    end
    for l in eachindex(src4)
        for i in eachindex(dest1)
            for j in eachindex(dest2)
                for k in eachindex(src3)
                    src3[k] = coef_srcdest[i,j,k,l]
                end
                apply!(op3, src3)
                for k in eachindex(src3)
                    coef_srcdest[i,j,k,l] = src3[k]
                end
            end
        end
    end
    for i in eachindex(dest1)
        for j in eachindex(dest2)
            for k in eachindex(dest3)
                for l in eachindex(src4)
                    src4[l] = coef_srcdest[i,j,k,l]
                end
                apply!(op4, src4)
                for l in eachindex(src4)
                    coef_srcdest[i,j,k,l] = src4[l]
                end
            end
        end
    end
    coef_srcdest
end


# Pretty printing
Display.combinationsymbol(op::TensorProductOperator) = Display.Symbol('⊗')
Display.displaystencil(op::TensorProductOperator) = composite_displaystencil(op)
