# generic_operators.jl


# In this file we define the interface for the following generic functions:
#
# Extension and restriction:
# - extension_operator
# - restriction_operator
#
# Approximation:
# - interpolation_operator
# - approximation_operator
# - leastsquares_operator
# - evaluation_operator
# - transform_operator (with transform_pre_operator and transform_post_operator)
#
# Calculus:
# - differentiation_operator
# - antidifferentation_operator

# These operators are also defined for TensorProductSet's.


############################
# Extension and restriction
############################

# A function set can often be extended to a similar function set of a different size.
# Extension and restriction operators always have a source and destination set.
# For introspection, you can ask a set what a suitable extensize size would be,
# and use a resized set as the destination of the operator:
#
#   larger_size = extension_size(some_set)
#   larger_set = resize(some_set, larger_size)
#   E = extension_operator(some_set, larger_set)
#
# and similarly for restriction. This is the default if no destination set is
# supplied to extension_operator.
#
# If a set does not need to store data for its extension operator, it can simply
# choose to implement the action of the Extension operator, e.g.:
#
# apply!(op::Extension, s1::SomeSet, s2::SomeSet, coef_dest, coef_src) = ...

immutable Extension{SRC <: FunctionSet,DEST <: FunctionSet,ELT} <: AbstractOperator{ELT}
    src     ::  SRC
    dest    ::  DEST
end

Extension(src::FunctionSet, dest::FunctionSet) =
    Extension{typeof(src),typeof(dest),op_eltype(src,dest)}(src, dest)

"""
An extension operator is an operator that can be used to extend a representation in a set s1 to a
representation in a larger set s2. The default extension operator is of type Extension with s1 and
s2 as source and destination.
"""
function extension_operator(s1::FunctionSet, s2::FunctionSet; options...)
    @assert has_extension(s1)
    Extension(s1, s2)
end

"""
Return a suitable length to extend to, for example one such that the corresponding grids are nested
and function evaluations can be shared. The default is twice the length of the current set.
"""
extension_size(s::FunctionSet) = 2*length(s)

extend(s::FunctionSet) = resize(s, extension_size(s))

extension_operator(s1::FunctionSet; options...) =
    extension_operator(s1, extend(s1); options...)


immutable Restriction{SRC <: FunctionSet,DEST <: FunctionSet,ELT} <: AbstractOperator{ELT}
    src     ::  SRC
    dest    ::  DEST
end

Restriction(src::FunctionSet, dest::FunctionSet) =
    Restriction{typeof(src),typeof(dest),op_eltype(src,dest)}(src, dest)


"""
A restriction operator does the opposite of what the extension operator does: it restricts
a representation in a set s1 to a representation in a smaller set s2. Loss of accuracy may result
from the restriction. The default restriction_operator is of type Restriction with sets s1 and
s2 as source and destination.
"""
restriction_operator(s1::FunctionSet, s2::FunctionSet; options...) = Restriction(s1, s2)

"""
Return a suitable length to restrict to, for example one such that the corresponding grids are nested
and function evaluations can be shared. The default is half the length of the current set.
"""
restriction_size(s::FunctionSet) = length(s)>>1

restrict(s::FunctionSet) = resize(s, restriction_size(s))

restriction_operator(s1::FunctionSet; options...) =
    restriction_operator(s1, restrict(s1); options...)


ctranspose(op::Extension) = restriction_operator(dest(op), src(op))

ctranspose(op::Restriction) = extension_operator(dest(op), src(op))






#################################################################
# Approximation: transformation, interpolation and evaluation
#################################################################

# A function set can have several associated transforms. The default transform is
# associated with the grid of the set, e.g. the FFT and the DCT for Chebyshev expansions
# which convert between coefficient space and value space. In this case, the
# transform maps coefficients to a DiscreteGridSpace.
#
# transform_operator takes two arguments, a source and destination set, in order
# to allow for different transforms.
#
# We assume that the transform itself is unitary. In order to compute an approximation
# to a function from function values, the transform is typically preceded and followed
# by an additional computation (e.g. the first Chebyshev coefficiet is halved after the DCT).
# These additional operations are achieved by the operator returned by the
# transform_pre_operator and transform_post_operator functions: pre acts on the
# coefficients of the source space, post on the coefficients of the dest space.


# The default transform space is the space associated with the grid of the set
transform_set(set::FunctionSet; options...) = DiscreteGridSpace(grid(set), eltype(set))

for op in (:transform_operator, :transform_pre_operator, :transform_post_operator)
    # Convenience functions: automatically convert a grid to a DiscreteGridSpace
    @eval $op(src::AbstractGrid, dest::FunctionSet; options...) =
        $op(DiscreteGridSpace(src, eltype(dest)), dest; options...)
    @eval $op(src::FunctionSet, dest::AbstractGrid; options...) =
        $op(src, DiscreteGridSpace(dest, eltype(src)); options...)
    # With only one argument, use the default transform space
    @eval $op(src::FunctionSet; options...) =
        $op(src, transform_set(src; options...); options...)
end

# Pre and post operations are identity by default.
transform_pre_operator(src::FunctionSet, dest::FunctionSet; options...) =
    IdentityOperator(src, dest)

transform_post_operator(src::FunctionSet, dest::FunctionSet; options...) =
    IdentityOperator(src, dest)

# Return all three of them in a tuple
transform_operators(sets::FunctionSet...; options...) =
    (transform_pre_operator(sets...; options...),
     transform_operator(sets...; options...),
     transform_post_operator(sets...; options...))

# Return the full operation: Post * Trans * Pre
function full_transform_operator(sets::FunctionSet...; options...)
    Pre,T,Post = transform_operators(sets...; options...)
    Post * T * Pre
end

## Interpolation and least squares

# Compute the interpolation matrix of the given basis on the given set of points
# (a grid or any iterable set of points)
function evaluation_matrix(set::FunctionSet, pts)
    T = promote_type(eltype(set), numtype(pts))
    a = Array(T, length(pts), length(set))
    evaluation_matrix!(a, set, pts)
end

function evaluation_matrix!(a::AbstractMatrix, set::FunctionSet, pts)
    @assert size(a,1) == length(pts)
    @assert size(a,2) == length(set)

    for (j,ϕ) in enumerate(set), (i,x) in enumerate(pts)
        a[i,j] = ϕ(x)
    end
    a
end

function interpolation_matrix(set::FunctionSet, pts)
    @assert length(set) == length(pts)
    evaluation_matrix(set, pts)
end

function leastsquares_matrix(set::FunctionSet, pts)
    @assert length(set) <= length(pts)
    evaluation_matrix(set, pts)
end


interpolation_operator(set::FunctionSet; options...) = interpolation_operator(set, grid(set); options...)

interpolation_operator(set::FunctionSet, grid::AbstractGrid; options...) =
    interpolation_operator(set, DiscreteGridSpace(grid, eltype(set)); options...)

# Interpolate set in the grid of dgs
function interpolation_operator(set::FunctionSet, dgs::DiscreteGridSpace; options...)
    if has_grid(set) && grid(set) == grid(dgs) && has_transform(set, dgs)
        full_transform_operator(dgs, set; options...)
    else
        SolverOperator(dgs, set, qrfact(evaluation_matrix(set, grid(dgs))))
    end
end


function interpolate(set::FunctionSet, pts, f)
    A = evaluation_matrix(set, pts)
    B = eltype(A)[f(x...) for x in pts]
    SetExpansion(set, A\B)
end

function leastsquares_operator(set::FunctionSet; samplingfactor = 2, options...)
    if has_grid(set)
        set2 = resize(set, samplingfactor*length(set))
        ls_grid = grid(set2)
    else
        ls_grid = EquispacedGrid(samplingfactor*length(set), left(set), right(set))
    end
    leastsquares_operator(set, ls_grid; options...)
end

leastsquares_operator(set::FunctionSet, grid::AbstractGrid; options...) =
    leastsquares_operator(set, DiscreteGridSpace(grid, eltype(set)); options...)

function leastsquares_operator(set::FunctionSet, dgs::DiscreteGridSpace; options...)
    if has_grid(set)
        larger_set = resize(set, size(dgs))
        if grid(larger_set) == grid(dgs) && has_transform(larger_set, dgs)
            R = restriction_operator(larger_set, set; options...)
            T = full_transform_operator(dgs, larger_set; options...)
            return R * T
        end
    end
    SolverOperator(dgs, set, qrfact(evaluation_matrix(set, grid(dgs))))
end



evaluation_operator(s::FunctionSet; options...) = evaluation_operator(s, grid(s); options...)

evaluation_operator(s::FunctionSet, g::AbstractGrid; options...) = evaluation_operator(s, DiscreteGridSpace(g, eltype(s)); options...)

# Evaluate s in the grid of dgs
function evaluation_operator(s::FunctionSet, dgs::DiscreteGridSpace; options...)
    if has_transform(s, dgs)
        if length(s) == length(dgs)
            full_transform_operator(s, dgs; options...)
        elseif length(s)<length(dgs)
            slarge = resize(s, length(dgs))
            evaluation_operator(slarge, dgs; options...) * extension_operator(s, slarge; options...)
        else
            # This might be faster implemented by:
            #   - finding an integer n so that nlength(dgs)>length(s)
            #   - resorting to the above evaluation + extension
            #   - subsampling by factor n
            MultiplicationOperator(s, dgs, evaluation_matrix(s, grid(dgs)))
        end
    else
        MultiplicationOperator(s, dgs, evaluation_matrix(s, grid(dgs)))
    end
end

default_approximation_operator = leastsquares_operator

"""
The approximation_operator function returns an operator that can be used to approximate
a function in the function set. This operator maps a grid to a set of coefficients.
"""
function approximation_operator(b::FunctionSet; options...)
    if is_basis(b)
        interpolation_operator(b; options...)
    else
        default_approximation_operator(b; options...)
    end
end



# Automatically sample a function if an operator is applied to it with a
# source that has a grid
(*)(op::AbstractOperator, f::Function) = op * sample(grid(src(op)), f, eltype(src(op)))

approximate(s::FunctionSet, f::Function; options...) =
    SetExpansion(s, approximation_operator(s; options...) * f)




####################
# Differentiation/AntiDifferentiation
####################

"""
The differentiation operator of a set maps an expansion in the set to an expansion of its derivative.
The result of this operation may be an expansion in a different set. A function set can have different
differentiation operators, with different result sets.
For example, an expansion of Chebyshev polynomials up to degree n may map to polynomials up to degree n,
or to polynomials up to degree n-1.
"""
immutable Differentiation{SRC <: FunctionSet,DEST <: FunctionSet,ELT} <: AbstractOperator{ELT}
    src     ::  SRC
    dest    ::  DEST
    order   ::  Int
end

Differentiation(src::FunctionSet, dest::FunctionSet, order) =
    Differentiation{typeof(src),typeof(dest),op_eltype(src,dest)}(src, dest, order)


order(op::Differentiation) = op.order


"""
The differentation_operator function returns an operator that can be used to differentiate
a function in the function set, with the result as an expansion in a second set.
"""
function differentiation_operator(s1::FunctionSet, s2::FunctionSet, order; options...)
    @assert has_derivative(s1)
    Differentiation(s1, s2, order)
end

# Default if no order is specified
function differentiation_operator(s1::FunctionSet1d, order=1; options...)
    s2 = derivative_set(s1, order)
    differentiation_operator(s1, s2, order; options...)
end
differentiation_operator(s1::FunctionSet; dim=1, options...) = differentiation_operator(s1, dimension_tuple(ndims(s1), 1))
function differentiation_operator(s1::FunctionSet, order; options...)
    s2 = derivative_set(s1, order)
    differentiation_operator(s1, s2, order; options...)
end


"""
The antidifferentiation operator of a set maps an expansion in the set to an expansion of its
antiderivative. The result of this operation may be an expansion in a different set. A function set
can have different antidifferentiation operators, with different result sets.
"""
immutable AntiDifferentiation{SRC <: FunctionSet,DEST <: FunctionSet,ELT} <: AbstractOperator{ELT}
    src     ::  SRC
    dest    ::  DEST
    order   ::  Int
end

AntiDifferentiation(src::FunctionSet, dest::FunctionSet, order) =
    AntiDifferentiation{typeof(src),typeof(dest),op_eltype(src,dest)}(src, dest, order)

order(op::AntiDifferentiation) = op.order

# The default antidifferentiation implementation is differentiation with a negative order (such as for Fourier)
# If the destination contains a DC coefficient, it is zero by default.

"""
The antidifferentiation_operator function returns an operator that can be used to find the antiderivative
of a function in the function set, with the result an expansion in a second set.
"""
function antidifferentiation_operator(s1::FunctionSet, s2::FunctionSet, order; options...)
    @assert has_antiderivative(s1)
    AntiDifferentiation(s1, s2, order)
end

# Default if no order is specified
function antidifferentiation_operator(s1::FunctionSet1d; options...)
    order = 1
    s2 = antiderivative_set(s1, order)
    antidifferentiation_operator(s1, s2, order; options...)
end

function antidifferentiation_operator(s1::FunctionSet; dim=1, options...)
    order = dimension_tuple(ndims(s1), dim)
    s2 = antiderivative_set(s1, order)
    antidifferentiation_operator(s1, s2, order; options...)
end





#####################################
# Operators for tensor product sets
#####################################



# We make a special case for transform operators, so that they can be intercepted in case a multidimensional
# transform is available for a specific basis.
transform_operator(s1::TensorProductSet, s2::TensorProductSet; options...) =
    transform_operator_tensor(s1, s2, elements(s1)..., elements(s2)...; options...)

transform_operator_tensor(s1, s2, s1_set1, s1_set2, s2_set1, s2_set2; options...) =
    tensorproduct(transform_operator(s1_set1, s2_set1; options...),
        transform_operator(s1_set2, s2_set2; options...))

transform_operator_tensor(s1, s2, s1_set1, s1_set2, s1_set3,
                                  s2_set1, s2_set2, s2_set3; options...) =
    tensorproduct(transform_operator(s1_set1, s2_set1; options...),
        transform_operator(s1_set2, s2_set2; options...),
        transform_operator(s1_set3, s2_set3; options...))

transform_operator_tensor(s1, s2, s1_set1, s1_set2, s1_set3, s1_set4,
                                  s2_set1, s2_set2, s2_set3, s2_set4; options...) =
    tensorproduct(transform_operator(s1_set1, s2_set1; options...),
        transform_operator(s1_set2, s2_set2; options...),
        transform_operator(s1_set3, s2_set3; options...),
        transform_operator(s1_set4, s2_set4; options...))

for op in (:extension_operator, :restriction_operator, :evaluation_operator,
            :interpolation_operator, :leastsquares_operator,
            :transform_pre_operator, :transform_post_operator)
    @eval $op(s1::TensorProductSet, s2::TensorProductSet; options...) =
        tensorproduct([$op(element(s1,i),element(s2, i); options...) for i in 1:composite_length(s1)]...)
end

for op in (:approximation_operator, )
    @eval $op(s::TensorProductSet; options...) =
        tensorproduct([$op(element(s,i); options...) for i in 1:composite_length(s)]...)
end

for op in (:differentiation_operator, :antidifferentiation_operator)
    # TODO: this assumes that the number of elements of the tensor product equals the dimension
    @eval function $op(s1::TensorProductSet, s2::TensorProductSet, order::NTuple; options...)
        tensorproduct([$op(element(s1,i), element(s2,i), order[i]; options...) for i in 1:composite_length(s1)]...)
    end
end
