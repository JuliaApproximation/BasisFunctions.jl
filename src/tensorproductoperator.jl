# tensorproductoperator.jl


"""
A TensorProductOperator represents the tensor product of other operators.

immutable TensorProductOperator{ELT,TO,ON,SCRATCH,SRC,DEST} <: AbstractOperator{SRC,DEST}

Parameters:
- ELT is the element type of the operator.
- TO is a tuple of (operator) types.
- ON is the number of elements in TO
- SCRATCH is the type of the scratch space allocated at creation of the tensor-product.
  It represents a tuple of arrays of ELT, which will hold intermediate calculations.
- SRC and DEST are the (tensor product) source and destination of this operator.
"""
immutable TensorProductOperator{ELT,TO,ON,SCRATCH,SRC,DEST} <: AbstractOperator{SRC,DEST}
    operators       ::  TO
    src             ::  SRC
    dest            ::  DEST
    scratch         ::  SCRATCH
    src_scratch     ::  NTuple{ON,Array{ELT,1}}
    dest_scratch    ::  NTuple{ON,Array{ELT,1}}
    
end

TensorProductOperator(operators...) = TensorProductOperator(eltype(map(eltype,operators)...), operators...)

# Expand tensorproductoperators in a tuple of operators to their individual operators.
function flattenops(ops::AbstractOperator...)
    flattened = AbstractOperator[]
    for i = 1:length(ops)
        appendops(flattened, ops[i])
    end
    flattened = tuple(flattened...)
end

appendops(flattened::Array{AbstractOperator,1}, f::AbstractOperator) = append!(flattened, [f])

function appendops(flattened::Array{AbstractOperator,1}, f::TensorProductOperator)
    for j = 1:tp_length(f)
        append!(flattened, [operator(f, j)])
    end
end


function TensorProductOperator{ELT}(::Type{ELT}, ops...)
    operators = flattenops(ops...)
    TO = typeof(operators)
    ON = length(operators)
    tp_src = TensorProductSet(map(src, operators)...)
    tp_dest = TensorProductSet(map(dest, operators)...)
    SRC = typeof(tp_src)
    DEST = typeof(tp_dest)

    # Scratch contains matrices of sufficient size to hold intermediate results
    # in the application of the tensor product operator.
    # Example, for ON=3 scratch is a length (ON-1)-tuple of matrices of size:
    # - [M1,N2,N3]
    # - [M1,M2,N3]
    # where operator J maps a set of length Nj to a set of length Mj.
    scratch_array = [ zeros(ELT, [length(dest(operators[k])) for k=1:j-1]..., [length(src(operators[k])) for k=j:ON]...) for j=2:ON]
    scratch = (scratch_array...)
    SCRATCH = typeof(scratch)

    # scr_scratch and dest_scratch are tuples of length ON that contain preallocated
    # storage to hold a vector for source and destination for each operator
    src_scratch_array = [zeros(ELT, length(src(operators[j]))) for j=1:ON]
    src_scratch = (src_scratch_array...)
    dest_scratch_array = [zeros(ELT, length(dest(operators[j]))) for j=1:ON]
    dest_scratch = (dest_scratch_array...)
    TensorProductOperator{ELT,TO,ON,SCRATCH,SRC,DEST}(operators, tp_src, tp_dest, scratch, src_scratch, dest_scratch)
end

tensorproduct(op::AbstractOperator, n) = TensorProductOperator([op for i=1:n]...)

⊗(op::AbstractOperator, ops::AbstractOperator...) = TensorProductOperator(op, ops...)

numtype{ELT,TO,ON,SCRATCH,SRC,DEST}(::Type{TensorProductOperator{ELT,TO,ON,SCRATCH,SRC,DEST}}) = numtype(SRC)

eltype{ELT,TO,ON,SCRATCH,SRC,DEST}(::Type{TensorProductOperator{ELT,TO,ON,SCRATCH,SRC,DEST}}) = ELT

# Element-wise src and dest functions
src(op::TensorProductOperator, j::Int) = set(op.src, j)
dest(op::TensorProductOperator, j::Int) = set(op.dest, j)


# Element-wise and total size functions
size(op::TensorProductOperator, j::Int) = j == 1 ? prod(map(length, sets(dest(op)))) : prod(map(length, sets(src(op))))
size(op::TensorProductOperator) = (size(op,1), size(op,2))

tp_length{ELT,TO,ON,SCRATCH,SRC,DEST}(op::TensorProductOperator{ELT,TO,ON,SCRATCH,SRC,DEST}) = ON

# Retrieve the operators that make up this tensor product
operators(op::TensorProductOperator) = op.operators
operator(op::TensorProductOperator, j::Int) = op.operators[j]

getindex(op::TensorProductOperator, j::Int) = operator(op, j)

ctranspose(op::TensorProductOperator) = TensorProductOperator(map(ctranspose, operators(op)))

function is_inplace{ELT,TO,ON,SCRATCH, SRC,DEST}(::Type{TensorProductOperator{ELT,TO,ON,SCRATCH,SRC,DEST}})
    is_inplace(TO)
end

is_inplace{OP1 <: AbstractOperator}(TO::Type{Tuple{OP1}}) = is_inplace(OP1) 
is_inplace{OP1 <: AbstractOperator, OP2 <: AbstractOperator}(TO::Type{Tuple{OP1,OP2}}) = is_inplace(OP1) & is_inplace(OP2)
is_inplace{OP1 <: AbstractOperator, OP2 <: AbstractOperator, OP3 <: AbstractOperator}(TO::Type{Tuple{OP1,OP2,OP3}}) = is_inplace(OP1) & is_inplace(OP2) & is_inplace(OP3)
    
# It is much easier to implement different versions specific to the dimension
# of the tensor-product, than to provide an N-dimensional implementation...
# The one-element tensorproduct is particularly simple:
apply!{ELT,TO}(op::TensorProductOperator{ELT,TO,1}, dest, src, coef_dest, coef_src) = apply!(operator(op,1), coef_dest, coef_src)


# TensorProduct with 2 elements
function apply!{ELT,TO}(op::TensorProductOperator{ELT,TO,2}, dest, src, coef_dest, coef_src)
    ## @assert size(dest) == size(coef_dest)
    ## @assert size(src)  == size(coef_src)

    M1,N1 = size(op[1])
    M2,N2 = size(op[2])
    # coef_src has size (N1,N2)
    # coef_dest has size (M1,M2)
    intermediate = op.scratch[1]
    src_j = op.src_scratch[1]
    dest_j = op.dest_scratch[1]
    for j = 1:N2
        for i = 1:N1
            src_j[i] = coef_src[i,j]
        end
        apply!(op[1], dest_j, src_j)
        for i = 1:M1
            intermediate[i,j] = dest_j[i]
        end
    end

    src_j = op.src_scratch[2]
    dest_j = op.dest_scratch[2]
    for j = 1:M1
        for i = 1:N2
            src_j[i] = intermediate[j,i]
        end
        apply!(op[2], dest_j, src_j)
        for i = 1:M2
            coef_dest[j,i] = dest_j[i]
        end
    end
end

# In-place variant
function apply!{ELT,TO}(op::TensorProductOperator{ELT,TO,2}, dest, src, coef_srcdest)
    # There are cases where coef_srcdest is a linearized vector instead of a matrix.
    coef = reshape(coef_srcdest, size(dest))

    M1,N1 = size(op[1])
    M2,N2 = size(op[2])

    # coef_srcdest has size (N1,N2) = (M1,M2)
    src_j = op.src_scratch[1]
    for j = 1:N2
        for i = 1:N1
            src_j[i] = coef[i,j]
        end
        apply!(op[1], src_j)
        for i = 1:M1
            coef[i,j] = src_j[i]
        end
    end

    src_j = op.src_scratch[2]
    for j = 1:M1
        for i = 1:N2
            src_j[i] = coef[j,i]
        end
        apply!(op[2], src_j)
        for i = 1:M2
            coef[j,i] = src_j[i]
        end
    end
end



# TensorProduct with 3 elements
function apply!{ELT,TO}(op::TensorProductOperator{ELT,TO,3}, dest, src, coef_dest, coef_src)
    @assert size(dest) == size(coef_dest)
    @assert size(src)  == size(coef_src)

    M1,N1 = size(op[1])
    M2,N2 = size(op[2])
    M3,N3 = size(op[3])
    # coef_src has size (N1,N2,N3)
    # coef_dest has size (M1,M2,M3)

    intermediate1 = op.scratch[1]
    src_j = op.src_scratch[1]
    dest_j = op.dest_scratch[1]
    for j = 1:N2
        for k = 1:N3
            for i = 1:N1
                src_j[i] = coef_src[i,j,k]
            end
            apply!(op[1], dest_j, src_j)
            for i = 1:M1
                intermediate1[i,j,k] = dest_j[i]
            end
        end
    end

    intermediate2 = op.scratch[2]
    src_j = op.src_scratch[2]
    dest_j = op.dest_scratch[2]
    for i = 1:M1
        for k = 1:N3
            for j = 1:N2
                src_j[j] = intermediate1[i,j,k]
            end
            apply!(op[2], dest_j, src_j)
            for j = 1:M2
                intermediate2[i,j,k] = dest_j[j]
            end
        end
    end

    src_j = op.src_scratch[3]
    dest_j = op.dest_scratch[3]
    for i = 1:M1
        for j = 1:M2
            for k = 1:N3
                src_j[k] = intermediate2[i,j,k]
            end
            apply!(op[3], dest_j, src_j)
            for k = 1:M3
                coef_dest[i,j,k] = dest_j[k]
            end
        end
    end
end

# In-place variant
function apply!{ELT,TO}(op::TensorProductOperator{ELT,TO,3}, dest, src, coef_srcdest)

    # There are cases where coef_srcdest is a large linearized vector rather than a tensor
    coef = reshape(coef_srcdest, size(dest))

    M1,N1 = size(op[1])
    M2,N2 = size(op[2])
    M3,N3 = size(op[3])
    # coef_srcdest has size (N1,N2,N3) = (M1,M2,M3)

    src_j = op.src_scratch[1]
    for j = 1:N2
        for k = 1:N3
            for i = 1:N1
                src_j[i] = coef[i,j,k]
            end
            apply!(op[1], src_j)
            for i = 1:M1
                coef[i,j,k] = src_j[i]
            end
        end
    end

    src_j = op.src_scratch[2]
    for i = 1:M1
        for k = 1:N3
            for j = 1:N2
                src_j[j] = coef[i,j,k]
            end
            apply!(op[2], src_j)
            for j = 1:M2
                coef[i,j,k] = src_j[j]
            end
        end
    end

    src_j = op.src_scratch[3]
    for i = 1:M1
        for j = 1:M2
            for k = 1:N3
                src_j[k] = coef[i,j,k]
            end
            apply!(op[3], src_j)
            for k = 1:M3
                coef[i,j,k] = src_j[k]
            end
        end
    end
end

