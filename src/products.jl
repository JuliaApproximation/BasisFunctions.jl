# tensorproducts.jl

"""
Create a tensor product of the supplied arguments.

The function `tensorproduct` applies some simplifications and does not necessarily
return a Product type.

A `tensorproduct(a)` with just a single element returns `a`.

For integer `n`, `tensorproduct(a, n)` becomes `tensorproduct(a, a, ..., a)`.
A type-safe variant is `tensorproduct(a, Val{N})`.
"""
tensorproduct() = nothing

# Don't create a tensor product of just one element
tensorproduct(a::Tuple) = tensorproduct(a...)
tensorproduct(a::AbstractArray) = tensorproduct(a...)
tensorproduct(a) = a

# Create a tensor product with n times the same element
tensorproduct(a, n::Int) = tensorproduct(ntuple(t->a, n)...)

tensorproduct(a, ::Type{Val{N}}) where {N} = tensorproduct(ntuple(t->a, Val{N})...)

# Use \otimes as notation for tensor product.
⊗ = tensorproduct

# Flatten a sequence of elements that may be recursively composite
# For example: a ProductDomain of ProductDomains will yield a list of each of the
# individual domains, like the leafs of a tree structure.
function flatten{T}(::Type{T}, elements::Array, BaseType = Any)
    flattened = BaseType[]
    for element in elements
        append_flattened!(T, flattened, element)
    end
    flattened
end

flatten{T}(::Type{T}, elements...) = tuple(flatten(T, [el for el in elements])...)

function append_flattened!{T}(::Type{T}, flattened::Vector, element::T)
    for el in elements(element)
        append_flattened!(T, flattened, el)
    end
end

function append_flattened!{T}(::Type{T}, flattened::Vector, element)
    append!(flattened, [element])
end

# All tensor products are created using the generic 'tensorproduct' function.
# This function calls a suitable constructor for the tensor product.

for (BaseType,TPType) in [(:AbstractOperator,:TensorProductOperator),
           (:FunctionSet,:TensorProductSet)]
    # In order to avoid strange nested structures, we flatten the arguments
    @eval tensorproduct(args::$BaseType...) = $TPType(flatten($TPType, args...)...)
    @eval tensorproduct(arg::$BaseType, n::Int) = tensorproduct([arg for i in 1:n]...)
    # Disallow tensor products with just one argument
    @eval tensorproduct(arg::$BaseType) = arg

end

for (BaseType,TPType) in [ (:AbstractGrid, :ProductGrid)]
    # Override × for grids
    @eval cross(args::$BaseType...) = cartesianproduct(args...)
    # In order to avoid strange nested structures, we flatten the arguments
    @eval cartesianproduct(args::$BaseType...) = $TPType(flatten($TPType, args...)...)
    @eval cartesianproduct(arg::$BaseType, n::Int) = cartesianproduct([arg for i in 1:n]...)
    # Disallow cartesian products with just one argument
    @eval cartesianproduct(arg::$BaseType) = arg
end

function is_homogeneous(tp)
    T = typeof(element(tp, 1))
    reduce(&, map(el->typeof(el)==T, elements(tp)))
end

function basetype(tp)
    promote_type(map(typeof, elements(tp))...)
end


# The routines below seem to fail in Julia 0.4.5. The compiler goes in an
# infinite loop in apply_composite! Should be fixed in 0.5. See #10340:
# https://github.com/JuliaLang/julia/issues/10340


# function tensorproduct(op1::AbstractOperator, op2::AbstractOperator)
#     if dimension(src(op1)) == 1 && dimension(src(op2)) == 1
#         d1 = dimension_operator(src(op1) ⊗ src(op2), dest(op1) ⊗ src(op2), op1, 1)
#         d2 = dimension_operator(dest(op1) ⊗ src(op2), dest(op1) ⊗ dest(op2), op2, 2)
#         compose(d1,d2)
#     else
#         TensorProductOperator(op1, op2)
#     end
# end
#
# function tensorproduct(op1::AbstractOperator, op2::AbstractOperator, op3::AbstractOperator; options...)
#     if dimension(src(op1)) == 1 && dimension(src(op2)) == 1 && dimension(src(op3)) == 1
#         d1 = dimension_operator(src(op1) ⊗ src(op2) ⊗ src(op3), dest(op1) ⊗ src(op2) ⊗ src(op3), op1, 1; options...)
#         d2 = dimension_operator(dest(op1) ⊗ src(op2) ⊗ src(op3), dest(op1) ⊗ dest(op2) ⊗ src(op3), op2, 2; options...)
#         d3 = dimension_operator(dest(op1) ⊗ dest(op2) ⊗ src(op3), dest(op1) ⊗ dest(op2) ⊗ dest(op3), op3, 3; options...)
#         compose(d1,d2,d3)
#     else
#         TensorProductOperator(op1, op2, op3)
#     end
# end