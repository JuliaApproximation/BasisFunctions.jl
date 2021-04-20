
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
function flatten(::Type{T}, elements::Array, BaseType = Any) where {T}
    flattened = BaseType[]
    for element in elements
        append_flattened!(T, flattened, element)
    end
    flattened
end

flatten(T, elements...) = tuple(flatten(T, [el for el in elements])...)

function append_flattened!(::Type{T}, flattened::Vector, element::T) where {T}
    for el in components(element)
        append_flattened!(T, flattened, el)
    end
end

function append_flattened!(::Type{T}, flattened::Vector, element) where {T}
    append!(flattened, [element])
end

# All tensor products are created using the generic 'tensorproduct' function.
# This function calls a suitable constructor for the tensor product.

for (BaseType,TPType) in [(:DictionaryOperator,:TensorProductOperator),
           (:Dictionary,:TensorProductDict)]
    # In order to avoid strange nested structures, we flatten the arguments
    @eval tensorproduct(args::$BaseType...) = $TPType(flatten($TPType, args...)...)
    @eval tensorproduct(arg::$BaseType, n::Int) = tensorproduct([arg for i in 1:n]...)
    # Disallow tensor products with just one argument
    @eval tensorproduct(arg::$BaseType) = arg

end

function ishomogeneous(tp)
    T = typeof(component(tp, 1))
    mapreduce(el->typeof(el)==T, &, components(tp))
end

basetype(tp) = promote_type(map(typeof, components(tp))...)
