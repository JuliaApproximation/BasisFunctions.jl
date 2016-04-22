# tensorproducts.jl

# Use \otimes as notation for tensor product.
⊗(args...) = tensorproduct(args...)

# All tensor products are created using the generic 'tensorproduct' function.
# This function calls a suitable constructor for the tensor product.

tensorproduct() = nothing
for (BaseType,TPType) in [(:AbstractOperator,:TensorProductOperator),
           (:FunctionSet,:TensorProductSet),
           (:AbstractGrid, :TensorProductGrid)]
    # In order to avoid strange nested structures, we flatten the arguments
    @eval tensorproduct(args::$BaseType...) = $TPType(flatten($TPType, args...)...)
    @eval tensorproduct(arg::$BaseType, n::Int) = tensorproduct([arg for i in 1:n]...)
    # Disallow tensor products with just one argument
    @eval tensorproduct(arg::$BaseType) = arg

end

# Flatten a sequence of elements that may be recursively composite
# For example: a TensorProductSet of TensorProductSets will yield a list of each of the
# individual sets, like the leafs of a tree structure.
function flatten{T}(::Type{T}, elements...)
    flattened = []
    for element in elements
        append_flattened!(T, flattened, element)
    end
    flattened = tuple(flattened...)
end

function append_flattened!{T}(::Type{T}, flattened::Vector, element::T)
    for el in elements(element)
        append_flattened!(T, flattened, el)
    end
end

function append_flattened!{T}(::Type{T}, flattened::Vector, element)
    append!(flattened, [element])
end

# The routines below seem to fail in Julia 0.4.5. The compiler goes in an
# infinite loop in apply_composite! Should be fixed in 0.5. See #10340:
# https://github.com/JuliaLang/julia/issues/10340


# function tensorproduct(op1::AbstractOperator, op2::AbstractOperator)
#     if ndims(src(op1)) == 1 && ndims(src(op2)) == 1
#         d1 = dimension_operator(src(op1) ⊗ src(op2), dest(op1) ⊗ src(op2), op1, 1)
#         d2 = dimension_operator(dest(op1) ⊗ src(op2), dest(op1) ⊗ dest(op2), op2, 2)
#         compose(d1,d2)
#     else
#         TensorProductOperator(op1, op2)
#     end
# end
#
# function tensorproduct(op1::AbstractOperator, op2::AbstractOperator, op3::AbstractOperator; options...)
#     if ndims(src(op1)) == 1 && ndims(src(op2)) == 1 && ndims(src(op3)) == 1
#         d1 = dimension_operator(src(op1) ⊗ src(op2) ⊗ src(op3), dest(op1) ⊗ src(op2) ⊗ src(op3), op1, 1; options...)
#         d2 = dimension_operator(dest(op1) ⊗ src(op2) ⊗ src(op3), dest(op1) ⊗ dest(op2) ⊗ src(op3), op2, 2; options...)
#         d3 = dimension_operator(dest(op1) ⊗ dest(op2) ⊗ src(op3), dest(op1) ⊗ dest(op2) ⊗ dest(op3), op3, 3; options...)
#         compose(d1,d2,d3)
#     else
#         TensorProductOperator(op1, op2, op3)
#     end
# end
