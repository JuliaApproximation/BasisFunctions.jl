# tensorproductoperator.jl


"""
A TensorProductOperator represents the tensor product of other operators.

immutable TensorProductOperator{TO,ELT,SCRATCH,SRC_SCRATCH,DEST_SCRATCH} <: AbstractOperator{ELT}

Parameters:
- TO is a tuple of (operator) types.
- ELT is the element type of the operator.
- SCRATCH parameters describe the type of scratch storage space
"""
immutable TensorProductOperator{TO,ELT,SCRATCH,SRC_SCRATCH,DEST_SCRATCH} <: AbstractOperator{ELT}
    src             ::  FunctionSet
    dest            ::  FunctionSet
    operators       ::  TO
    scratch         ::  SCRATCH
    src_scratch     ::  SRC_SCRATCH
    dest_scratch    ::  DEST_SCRATCH
end
# TODO: try to remove some of the type parameters.

# Generic functions for composite types:
elements(op::TensorProductOperator) = op.operators
element(op::TensorProductOperator, j::Int) = op.operators[j]
composite_length(op::TensorProductOperator) = length(op.operators)

function TensorProductOperator(operators...)
    ELT = promote_type(map(eltype, operators)...)
    TO = typeof(operators)
    L = length(operators)
    tp_src = tensorproduct(map(src, operators)...)
    tp_dest = tensorproduct(map(dest, operators)...)

    # Scratch contains matrices of sufficient size to hold intermediate results
    # in the application of the tensor product operator.
    # Example, for L=3 scratch is a length (L-1)-tuple of matrices of size:
    # - [M1,N2,N3]
    # - [M1,M2,N3]
    # where operator J maps a set of length Nj to a set of length Mj.
    scratch_array = [ zeros(ELT, [length(dest(operators[k])) for k=1:j-1]..., [length(src(operators[k])) for k=j:L]...) for j=2:L]
    scratch = (scratch_array...)
    SCRATCH = typeof(scratch)

    # scr_scratch and dest_scratch are tuples of length len that contain preallocated
    # storage to hold a vector for source and destination for each operator
    src_scratch_array = [zeros(ELT, size(src(operators[j]))) for j=1:L]
    src_scratch = (src_scratch_array...)
    dest_scratch_array = [zeros(ELT, size(dest(operators[j]))) for j=1:L]
    dest_scratch = (dest_scratch_array...)
    SRC_SCRATCH = typeof(src_scratch)
    DEST_SCRATCH = typeof(dest_scratch)
    TensorProductOperator{TO,ELT,SCRATCH,SRC_SCRATCH,DEST_SCRATCH}(tp_src, tp_dest, operators, scratch, src_scratch, dest_scratch)
end


# Element-wise src and dest functions
src(op::TensorProductOperator, j::Int) = src(element(op,j))
dest(op::TensorProductOperator, j::Int) = dest(element(op,j))


# Element-wise and total size functions
size(op::TensorProductOperator, j::Int) = j == 1 ? prod(map(length, elements(dest(op)))) : prod(map(length, elements(src(op))))
size(op::TensorProductOperator) = (size(op,1), size(op,2))


#getindex(op::TensorProductOperator, j::Int) = element(op, j)

ctranspose(op::TensorProductOperator) = TensorProductOperator(map(ctranspose, elements(op))...)

inv(op::TensorProductOperator) = TensorProductOperator(map(inv, elements(op))...)

is_inplace(op::TensorProductOperator) = reduce(&, map(is_inplace, op.operators))
is_diagonal(op::TensorProductOperator) = reduce(&, map(is_diagonal, op.operators))


apply!(op::TensorProductOperator, coef_dest, coef_src) =
    apply_tensor!(op, coef_dest, coef_src, op.operators, op.scratch, op.src_scratch, op.dest_scratch)

apply_inplace!(op::TensorProductOperator, coef_srcdest) =
    apply_inplace_tensor!(op, coef_srcdest, op.operators, op.src_scratch)

function apply_tensor!(op, coef_dest, coef_src, operators::NTuple{1}, scratch, src_scratch, dest_scratch)
    println("One-element TensorProductOperators should not exist!")
    apply!(operators[1], coef_dest, coef_src)
end

function apply_inplace_tensor!(op, coef_srcdest, operators::NTuple{1}, src_scratch)
    println("One-element TensorProductOperators should not exist!")
    apply!(operators[1], coef_srcdest)
end

function apply_tensor!(op, coef_dest, coef_src, operators::NTuple{2}, scratch, src_scratch, dest_scratch)
    src1 = src_scratch[1]
    src2 = src_scratch[2]
    dest1 = dest_scratch[1]
    dest2 = dest_scratch[2]
    op1 = operators[1]
    op2 = operators[2]
    intermediate = scratch[1]

    # println("coef_src: ", size(coef_src))
    # println("coef_dest: ", size(coef_dest))
    # println("src1: ", size(src1))
    # println("src2: ", size(src2))
    # println("dest1: ", size(dest1))
    # println("dest2: ", size(dest2))
    # println("op1: ", size(op1))
    # println("op2: ", size(op2))
    # println("intermediate: ", size(intermediate))

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

function apply_inplace_tensor!(op, coef_srcdest, operators::NTuple{2}, src_scratch)
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

function apply_tensor!(op, coef_dest, coef_src, operators::NTuple{3}, scratch, src_scratch, dest_scratch)
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

    # println("coef_src: ", size(coef_src))
    # println("coef_dest: ", size(coef_dest))
    # println("src1: ", size(src1))
    # println("src2: ", size(src2))
    # println("dest1: ", size(dest1))
    # println("dest2: ", size(dest2))
    # println("op1: ", size(op1))
    # println("op2: ", size(op2))
    # println("intermediate: ", size(intermediate))

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

function apply_inplace_tensor!(op, coef_srcdest, operators::NTuple{3}, src_scratch)
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

# # Reshape the scratch space first to the right size
# # TODO: get rid of this by making scr_scratch the right size from the start
# function apply!{ELT,TO}(op::TensorProductOperator{ELT,TO,2}, coef_dest, coef_src)
#     src1 = reshape(op.src_scratch[1], size(BasisFunctions.src(op, 1)))
#     src2 = reshape(op.src_scratch[2], size(BasisFunctions.src(op, 2)))
#     dest1 = reshape(op.dest_scratch[1], size(BasisFunctions.dest(op, 1)))
#     dest2 = reshape(op.dest_scratch[2], size(BasisFunctions.dest(op, 2)))
#     apply_reshaped!(op, reshape(coef_dest, length(dest1), length(dest2)),
#         reshape(coef_src, length(src1), length(src2)),
#         src1, src2, dest1, dest2, op.scratch[1], element(op,1), element(op,2))
# end
#
# # Same for in-place variant
# function apply_inplace!{ELT,TO}(op::TensorProductOperator{ELT,TO,2}, coef_srcdest)
#     src1 = reshape(op.src_scratch[1], size(BasisFunctions.src(op, 1)))
#     src2 = reshape(op.src_scratch[2], size(BasisFunctions.src(op.operators[2])))
#     apply_inplace_reshaped!(op, reshape(coef_srcdest, size(dest)), src1, src2, element(op,1), element(op,2))
# end
#
# # TensorProduct with 2 elements
# function apply_reshaped!{ELT,TO}(op::TensorProductOperator{ELT,TO,2}, coef_dest, coef_src,
#     src1, src2, dest1, dest2, intermediate, op1, op2)
#     ## @assert size(dest) == size(coef_dest)
#     ## @assert size(src)  == size(coef_src)
#
#     M1,N1 = size(op1)
#     M2,N2 = size(op2)
#     # coef_src has size (N1,N2)
#     # coef_dest has size (M1,M2)
#     for j = 1:N2
#         for i = 1:N1
#             src1[i] = coef_src[i,j]
#         end
#         apply!(op1, dest1, src1)
#         for i = 1:M1
#             intermediate[i,j] = dest1[i]
#         end
#     end
#
#     for j = 1:M1
#         for i = 1:N2
#             src2[i] = intermediate[j,i]
#         end
#         apply!(op2, dest2, src2)
#         for i = 1:M2
#             coef_dest[j,i] = dest2[i]
#         end
#     end
#     coef_dest
# end
#
# # In-place variant
# function apply_inplace_reshaped!{ELT,TO}(op::TensorProductOperator{ELT,TO,2}, coef, src1, src2, op1, op2)
#     # There are cases where coef_srcdest is a linearized vector instead of a matrix.
#     M1,N1 = size(op1)
#     M2,N2 = size(op2)
#
#     # coef has size (N1,N2) = (M1,M2)
#     for j = 1:N2
#         for i = 1:N1
#             src1[i] = coef[i,j]
#         end
#         apply!(op1, src1)
#         for i = 1:M1
#             coef[i,j] = src1[i]
#         end
#     end
#
#     for j = 1:M1
#         for i = 1:N2
#             src2[i] = coef[j,i]
#         end
#         apply!(op2, src2)
#         for i = 1:M2
#             coef[j,i] = src2[i]
#         end
#     end
#     coef
# end
#

#
# # TensorProduct with 3 elements
# function apply!{ELT,TO}(op::TensorProductOperator{ELT,TO,3}, coef_dest, coef_src)
#     # @assert size(dest) == size(coef_dest)
#     # @assert size(src)  == size(coef_src)
#
#     M1,N1 = size(element(op,1))
#     M2,N2 = size(element(op,2))
#     M3,N3 = size(element(op,3))
#     # coef_src has size (N1,N2,N3)
#     # coef_dest has size (M1,M2,M3)
#
#     intermediate1 = op.scratch[1]
#     src_j = op.src_scratch[1]
#     dest_j = op.dest_scratch[1]
#     for j = 1:N2
#         for k = 1:N3
#             for i = 1:N1
#                 src_j[i] = coef_src[i,j,k]
#             end
#             apply!(element(op,1), dest_j, src_j)
#             for i = 1:M1
#                 intermediate1[i,j,k] = dest_j[i]
#             end
#         end
#     end
#
#     intermediate2 = op.scratch[2]
#     src_j = op.src_scratch[2]
#     dest_j = op.dest_scratch[2]
#     for i = 1:M1
#         for k = 1:N3
#             for j = 1:N2
#                 src_j[j] = intermediate1[i,j,k]
#             end
#             apply!(element(op,2), dest_j, src_j)
#             for j = 1:M2
#                 intermediate2[i,j,k] = dest_j[j]
#             end
#         end
#     end
#
#     src_j = op.src_scratch[3]
#     dest_j = op.dest_scratch[3]
#     for i = 1:M1
#         for j = 1:M2
#             for k = 1:N3
#                 src_j[k] = intermediate2[i,j,k]
#             end
#             apply!(element(op,3), dest_j, src_j)
#             for k = 1:M3
#                 coef_dest[i,j,k] = dest_j[k]
#             end
#         end
#     end
#     coef_dest
# end
#
# # In-place variant
# function apply_inplace!{ELT,TO}(op::TensorProductOperator{ELT,TO,3}, coef_srcdest)
#
#     # There are cases where coef_srcdest is a large linearized vector rather than a tensor
#     # TODO: remove the reshape as for the 2-element tensor product operator
# #    coef = reshape(coef_srcdest, size(dest))
#
#     M1,N1 = size(element(op,1))
#     M2,N2 = size(element(op,2))
#     M3,N3 = size(element(op,3))
#     # coef_srcdest has size (N1,N2,N3) = (M1,M2,M3)
#
#     src_j = op.src_scratch[1]
#     for j = 1:N2
#         for k = 1:N3
#             for i = 1:N1
#                 src_j[i] = coef_srcdest[i,j,k]
#             end
#             apply!(element(op,1), src_j)
#             for i = 1:M1
#                 coef_srcdest[i,j,k] = src_j[i]
#             end
#         end
#     end
#
#     src_j = op.src_scratch[2]
#     for i = 1:M1
#         for k = 1:N3
#             for j = 1:N2
#                 src_j[j] = coef_srcdest[i,j,k]
#             end
#             apply!(element(op,2), src_j)
#             for j = 1:M2
#                 coef_srcdest[i,j,k] = src_j[j]
#             end
#         end
#     end
#
#     src_j = op.src_scratch[3]
#     for i = 1:M1
#         for j = 1:M2
#             for k = 1:N3
#                 src_j[k] = coef_srcdest[i,j,k]
#             end
#             apply!(element(op,3), src_j)
#             for k = 1:M3
#                 coef_srcdest[i,j,k] = src_j[k]
#             end
#         end
#     end
#     coef_srcdest
# end
