# tensorproductoperator.jl


# A TensorProductOperator represents the tensor product of other operators.
# Parameter TO is a tuple of (operator) types.
# Parametes SRC and DEST are the (tensor product) source and destination of this operator.
immutable TensorProductOperator{TO,SRC,DEST} <: AbstractOperator{SRC,DEST}
    operators   ::  TO
    src         ::  SRC
    dest        ::  DEST
end

TensorProductOperator(operators...) = TensorProductOperator(operators, TensorProductSet(map(src, operators)...), TensorProductSet(map(dest, operators)...)

tensorproduct(o::Abstractoperator, n) = TensorProductOperator([o for i=1:n]...)

# Element-wise src and dest functions
src(o::TensorProductOperator, j::Int) = set(o.src, j)
dest(o::TensorProductOperator, j::Int) = set(o.dest, j)


size(o::TensorProductOperator, j::Int) = j == 1 ? prod(map(length, sets(src(o)))) : prod(map(length, sets(dest(o))))
size(o::TensorProductOperator) = (size(o,1), size(o,2))

operators(o::TensorProductOperator) = o.operators
operator(o::TensorProductOperator, j::Int) = o.operators[j]


