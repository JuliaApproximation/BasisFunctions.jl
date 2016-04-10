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
# - transform_operator
# - transform_normalization_operator
# - normalization_operator
#
# Calculus:
# - differentiation_operator
# - antidifferentation_operator

# These operators are also defined for TensorProductSet's.


############################
# Extension and restriction
############################



immutable Extension{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

"""
An extension operator is an operator that can be used to extend a representation in a set s1 to a
representation in a larger set s2. The default extension operator is of type Extension with s1 and
s2 as source and destination.
"""
extension_operator(s1::FunctionSet, s2::FunctionSet; options...) = Extension(s1, s2)

"""
Return a suitable length to extend to, for example one such that the corresponding grids are nested
and function evaluations can be shared. The default is twice the length of the current set.
"""
extension_size(s::FunctionSet) = 2*length(s)

extension_size(s::TensorProductSet) = map(extension_size, sets(s))

extend(s::FunctionSet) = resize(s, extension_size(s))

immutable Restriction{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end


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

ctranspose(op::Extension) = restriction_operator(dest(op), src(op))

ctranspose(op::Restriction) = extension_operator(dest(op), src(op))




#################
# Normalization
#################


immutable NormalizationOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

normalization_operator(src::FunctionSet; options...) = NormalizationOperator(src, normalize(src))

is_inplace{O <: NormalizationOperator}(::Type{O}) = True

is_diagonal{O <: NormalizationOperator}(::Type{O}) = True




#################################################################
# Approximation: transformation, interpolation and evaluation
#################################################################


"""
A function set can implement the apply! method of a suitable TransformOperator for any known
unitary transform.
Example: a discrete transform from a set of samples on a grid to a set of expansion coefficients.
"""
immutable TransformOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

# The default transform from src to dest is a TransformOperator. This may be overridden for specific source and destinations.
transform_operator(src, dest; options...) = TransformOperator(src, dest)

# Convenience functions: automatically convert a grid to a DiscreteGridSpace
transform_operator(src::AbstractGrid, dest::FunctionSet; options...) =
    transform_operator(DiscreteGridSpace(src, eltype(dest)), dest; options...)
transform_operator(src::FunctionSet, dest::AbstractGrid; options...) =
    transform_operator(src, DiscreteGridSpace(dest, eltype(src)); options...)

ctranspose(op::TransformOperator) = transform_operator(dest(op), src(op))

# Assume transform is unitary
inv(op::TransformOperator) = ctranspose(op)


## Interpolation and least squares

# Compute the interpolation matrix of the given basis on the given set of points (a grid or any iterable set of points)
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
        transform_normalization_operator(set; options...) * transform_operator(dgs, set; options...)
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
        larger_set = resize(set, length(dgs))
        if grid(larger_set) == grid(dgs) && has_transform(larger_set, dgs)
            R = restriction_operator(larger_set, set; options...)
            return R * transform_normalization_operator(larger_set; options...) * transform_operator(dgs, larger_set; options...)
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
            transform_operator(s, dgs; options...) * inv(transform_normalization_operator(s; options...))
        elseif length(s)<length(dgs)
            slarge = resize(s, length(dgs))
            evaluation_operator(slarge, dgs; options...) * extension_operator(s, slarge; options...)
        else
            # This might be faster implemented by:
            #   - finding an integer n so that nlength(dgs)>length(s)
            #   - resorting to the above evaluation + extension
            #   - subsampling by factor n
            MatrixOperator(interpolation_matrix(s, grid(dgs)), s, dgs)
        end
    else
        MatrixOperator(evaluation_matrix(s, grid(dgs)), s, dgs)
    end
end

"""
The approximation_operator function returns an operator that can be used to approximate
a function in the function set. This operator maps a grid to a set of coefficients.
"""
approximation_operator(b::FunctionSet; options...) = _approximation_operator(b, is_basis(b); options...)

# The default approximation for a basis is interpolation, for other sets it is least squares.
_approximation_operator(b, isbasis::True; options...) = interpolation_operator(b; options...)
_approximation_operator(b, isbasis::False; options...) = leastsquares_operator(b; options...)



# Automatically sample a function if an operator is applied to it with a
# source that has a grid
(*)(op::AbstractOperator, f::Function) = op * sample(grid(src(op)), f, eltype(src(op)))

approximate(s::FunctionSet, f::Function; options...) = SetExpansion(s, approximation_operator(s; options...) * f)




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
immutable Differentiation{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
    order   ::  Int
end

Differentiation{SRC <: FunctionSet, DEST <: FunctionSet}(src::SRC, dest::DEST = src, order = 1) =
    Differentiation{SRC,DEST}(src, dest, 1, 1)


order(op::Differentiation) = op.order


"""
The differentation_operator function returns an operator that can be used to differentiate
a function in the function set, with the result as an expansion in a second set.
"""
differentiation_operator(s1::FunctionSet, s2::FunctionSet, order = 1; options...) = Differentiation(s1, s2, order)

# With this definition below, the user may specify a single set and a variable, with or without an order
differentiation_operator(s1::FunctionSet, order = 1; options...) =
    differentiation_operator(s1, derivative_set(s1, order), order; options...)


"""
The antidifferentiation operator of a set maps an expansion in the set to an expansion of its
antiderivative. The result of this operation may be an expansion in a different set. A function set
can have different antidifferentiation operators, with different result sets.
"""
immutable AntiDifferentiation{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
    order   ::  Int
end

AntiDifferentiation{SRC <: FunctionSet, DEST <: FunctionSet}(src::SRC, dest::DEST = src, order = 1) =
    AntiDifferentiation{SRC,DEST}(src, dest, 1)

order(op::AntiDifferentiation) = op.order

# The default antidifferentiation implementation is differentiation with a negative order (such as for Fourier)
# If the destination contains a DC coefficient, it is zero by default.

"""
The antidifferentiation_operator function returns an operator that can be used to find the antiderivative
of a function in the function set, with the result an expansion in a second set.
"""
antidifferentiation_operator(s1::FunctionSet, s2::FunctionSet, order = 1; options...) =
    AntiDifferentiation(s1, s2, order)

# With this definition below, the user may specify a single set and a variable, with or without an order
antidifferentiation_operator(s1::FunctionSet, order = 1; options...) =
    antidifferentiation_operator(s1, antiderivative_set(s1, order), order; options...)




#####################################
# Operators for tensor product sets
#####################################

# We make a special case for transform operators, so that they can be intercepted in case a multidimensional
# transform is available for a specific basis.
transform_operator{TS1,TS2,SN}(s1::TensorProductSet{TS1,SN,2}, s2::TensorProductSet{TS2,SN,2}; options...) =
    transform_operator_tensor(s1, s2, set(s1, 1), set(s1, 2), set(s2, 1), set(s2, 2); options...)

transform_operator{TS1,TS2,SN}(s1::TensorProductSet{TS1,SN,3}, s2::TensorProductSet{TS2,SN,3}; options...) =
    transform_operator_tensor(s1, s2, set(s1, 1), set(s1, 2), set(s1, 3), set(s2, 1), set(s2, 2), set(s2, 3); options...)

transform_operator_tensor(s1, s2, s1_set1, s1_set2, s2_set1, s2_set2; options...) =
    TensorProductOperator(transform_operator(s1_set1, s2_set1; options...), transform_operator(s1_set2, s2_set2; options...))

transform_operator_tensor(s1, s2, s1_set1, s1_set2, s1_set3, s2_set1, s2_set2, s2_set3; options...) =
    TensorProductOperator(transform_operator(s1_set1, s2_set1; options...), transform_operator(s1_set2, s2_set2; options...),
        transform_operator(s1_set3, s2_set3; options...))

for op in (:extension_operator, :restriction_operator, :transform_operator, :evaluation_operator,
            :interpolation_operator, :leastsquares_operator)
    @eval $op{TS1,TS2,SN,LEN}(s1::TensorProductSet{TS1,SN,LEN}, s2::TensorProductSet{TS2,SN,LEN}; options...) =
        TensorProductOperator([$op(set(s1,i),set(s2, i); options...) for i in 1:LEN]...)
end

for op in (:approximation_operator, :normalization_operator, :transform_normalization_operator)
    @eval $op{TS,SN,LEN}(s::TensorProductSet{TS,SN,LEN}; options...) =
        TensorProductOperator([$op(set(s,i); options...) for i in 1:LEN]...)
end

for op in (:differentiation_operator, :antidifferentiation_operator)
    @eval function $op{TS1,TS2, SN,LEN}(s1::TensorProductSet{TS1,SN,LEN}, s2::TensorProductSet{TS2,SN,LEN}, order::NTuple{LEN}; options...)
        TensorProductOperator([$op(set(s1,i), set(s2,i), order[i]; options...) for i in 1:LEN]...)
    end
end

## # Overly complicated routines to select a single variable and order from a tensorproductset.
## for op in (:differentiation_operator, :antidifferentiation_operator)
##     @eval function $op{TS1,TS2,SN,LEN}(s1::TensorProductSet{TS1,SN,LEN}, s2::TensorProductSet{TS2,SN,LEN}, var::Int, order::Int)
##         operators = map(i->$op(set(s1,i),set(s2,i),1,0),1:LEN)
##         setindex = minimum(find(cumsum([SN...]).>(var-1)))
##         varadjusted = var-cumsum([0; SN...])[setindex]
##         operators[setindex] = $op(set(s1,setindex),set(s2,setindex),varadjusted,order)
##         TensorProductOperator(operators...)
##     end
## end

## for op in (:differentiation_operator, :antidifferentiation_operator)
##     @eval function $op{TS1,SN,LEN}(s1::TensorProductSet{TS1,SN,LEN}, var::Int, order::Int)
##         operators = AbstractOperator[IdentityOperator(set(s1,i)) for i in 1:LEN]
##         setindex = minimum(find(cumsum([SN...]).>(var-1)))
##         varadjusted = var-cumsum([0; SN...])[setindex]
##         operators[setindex] = $op(set(s1,setindex),varadjusted,order)
##         TensorProductOperator(operators...)
##     end
## end
