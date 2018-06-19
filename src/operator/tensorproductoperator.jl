# tensorproductoperator.jl


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
elements(op::TensorProductOperator) = op.operators
element(op::TensorProductOperator, j::Int) = op.operators[j]

is_composite(op::TensorProductOperator) = true

function TensorProductOperator(operators...)
    T = promote_type(map(eltype, operators)...)
    L = length(operators)
    tp_src = tensorproduct(map(src, operators)...)
    tp_dest = tensorproduct(map(dest, operators)...)

    # Scratch contains matrices of sufficient size to hold intermediate results
    # in the application of the tensor product operator.
    # Example, for L=3 scratch is a length (L-1)-tuple of matrices of size:
    # - [M1,N2,N3]
    # - [M1,M2,N3]
    # where operator J maps a set of length Nj to a set of length Mj.
    scratch_array = [ zeros(T, [length(dest(operators[k])) for k=1:j-1]..., [length(src(operators[k])) for k=j:L]...) for j=2:L]
    scratch = (scratch_array...)

    # scr_scratch and dest_scratch are tuples of length len that contain preallocated
    # storage to hold a vector for source and destination for each operator
    src_scratch_array = [zeros(src(operators[j])) for j=1:L]
    src_scratch = (src_scratch_array...)
    dest_scratch_array = [zeros(dest(operators[j])) for j=1:L]
    dest_scratch = (dest_scratch_array...)
    TensorProductOperator{T}(tp_src, tp_dest, operators, scratch, src_scratch, dest_scratch)
end


# Element-wise src and dest functions
src(op::TensorProductOperator, j::Int) = src(element(op, j))
dest(op::TensorProductOperator, j::Int) = dest(element(op, j))


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

function apply_tensor!{A}(op, coef_dest, coef_src, operators::Tuple{A}, scratch, src_scratch, dest_scratch)
    println("One-element TensorProductOperators should not exist!")
    apply!(operators[1], coef_dest, coef_src)
end

function apply_inplace_tensor!{A}(op, coef_srcdest, operators::Tuple{A}, src_scratch)
    println("One-element TensorProductOperators should not exist!")
    apply!(operators[1], coef_srcdest)
end

function apply_tensor!{A,B}(op, coef_dest, coef_src, operators::Tuple{A,B}, scratch, src_scratch, dest_scratch)
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
    #
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

function apply_inplace_tensor!{A,B}(op, coef_srcdest, operators::Tuple{A,B}, src_scratch)
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

function apply_tensor!{A,B,C}(op, coef_dest, coef_src, operators::Tuple{A,B,C}, scratch, src_scratch, dest_scratch)
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

function apply_inplace_tensor!{A,B,C}(op, coef_srcdest, operators::Tuple{A,B,C}, src_scratch)
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

SparseOperator(op::TensorProductOperator; options...) =
    TensorProductOperator([SparseOperator(opi) for opi in elements(op)]...)
    
function stencil(op::TensorProductOperator)
    A = Any[]
    push!(A,element(op,1))
    for i=2:length(elements(op))
        push!(A," ⊗ ")
        push!(A,element(op,i))
    end
    A
end
