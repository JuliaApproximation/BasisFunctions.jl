# dimop.jl

using BasisFunctions.Slices

const VIEW_COPY = 1
const VIEW_SUB = 2
const VIEW_VIEW = 3

const VIEW_DEFAULT = VIEW_COPY

"""
A DimensionOperator applies a given operator along one dimension in a higher-dimensional
data set. For example, an operator can be applied to each row (dim=1) or column (dim=2)
of a 2D array of coefficients.

A tensor product operation can be implemented as a sequence of DimensionOperator's.
"""
# Parameter VIEW determines the view type:
# 1: make a copy
# 2: use sub
# 3: use ArrayViews.view
# The first is the default.
struct DimensionOperator{VIEW,ELT} <: AbstractOperator{ELT}
    src             ::  Span
    dest            ::  Span
    op              ::  AbstractOperator{ELT}
    dim             ::  Int
    scratch_src     ::  AbstractArray{ELT}
    scratch_dest    ::  AbstractArray{ELT}

    function DimensionOperator{VIEW,ELT}(set_src::Span, set_dest::Span, op::AbstractOperator, dim::Int) where {VIEW,ELT}
        scratch_src = zeros(src(op))
        scratch_dest = zeros(dest(op))
        new(set_src, set_dest, op, dim, scratch_src, scratch_dest)
    end
end


DimensionOperator(src::Span, dest::Span, op, dim, viewtype) =
    DimensionOperator{viewtype,eltype(op)}(src, dest, op, dim)

is_inplace(op::DimensionOperator) = is_inplace(op.op)

# Generic function to create a DimensionOperator
# This function can be intercepted for operators that have a more efficient implementation.
dimension_operator(src, dest, op::AbstractOperator, dim; viewtype = VIEW_DEFAULT, options...) =
    DimensionOperator(src, dest, op, dim, viewtype)


function apply_inplace!(op::DimensionOperator, coef_srcdest)
    apply_dim_inplace!(op, coef_srcdest, op.op, op.dim)
end

function apply_dim_inplace!(dimop::DimensionOperator, coef_srcdest, op, dim,
    scratch_src = dimop.scratch_src)

    for slice in Slices.eachslice(coef_srcdest, dim)
        copy!(scratch_src, coef_srcdest, slice)
        apply!(op, scratch_src)
        copy!(coef_srcdest, slice, scratch_src)
    end
    coef_srcdest
end

function apply!(op::DimensionOperator, coef_dest, coef_src)
    apply_dim!(op, coef_dest, coef_src, op.op, op.dim)
end


function copy!(a::AbstractVector, b::AbstractArray, slice::Slices.SliceIndex)
    for i in eachindex(a)
        a[i] = b[slice,i]
    end
end

function copy!(a::AbstractArray, slice::Slices.SliceIndex, b::AbstractVector)
    for i in eachindex(b)
        a[slice,i] = b[i]
    end
end

function apply_dim!(dimop::DimensionOperator{1}, coef_dest, coef_src, op::AbstractOperator, dim,
    scratch_dest = dimop.scratch_dest,
    scratch_src = dimop.scratch_src)

    for (s_slice,d_slice) in Slices.joint(Slices.eachslice(coef_src, dim), Slices.eachslice(coef_dest, dim))
        copy!(scratch_src, coef_src, s_slice)
        apply!(op, scratch_dest, scratch_src)
        copy!(coef_dest, d_slice, scratch_dest)
    end
    coef_dest
end

function apply_dim!(dimop::DimensionOperator{2}, coef_dest, coef_src, op::AbstractOperator, dim)
    for (s_slice,d_slice) in Slices.joint(Slices.eachslice(coef_src, dim), Slices.eachslice(coef_dest, dim))
        src_view = sub(coef_src, s_slice)
        dest_view = sub(coef_dest, d_slice)
        apply!(op, dest_view, src_view)
    end
    coef_dest
end

function apply_dim!(dimop::DimensionOperator{3}, coef_dest, coef_src, op::AbstractOperator, dim)
    for (s_slice,d_slice) in Slices.joint(Slices.eachslice(coef_src, dim), Slices.eachslice(coef_dest, dim))
        src_view = view(coef_src, s_slice)
        dest_view = view(coef_dest, d_slice)
        apply!(op, dest_view, src_view)
    end
    coef_dest
end



"Replace the j-th set of a tensor product set with a different one."
replace(tpset::TensorProductSet, j, s) = tensorproduct([element(tpset, i) for i in 1:j-1]..., s, [element(tpset, i) for i in j+1:nb_elements(tpset)]...)

inv{VIEW}(op::DimensionOperator{VIEW}) = DimensionOperator(replace(src(op), op.dim, element(dest(op), op.dim)),
    replace(dest(op), op.dim, element(src(op), op.dim)), inv(op.op), op.dim, VIEW)

ctranspose{VIEW}(op::DimensionOperator{VIEW}) = DimensionOperator(replace(src(op), op.dim, element(dest(op), op.dim)),
    replace(dest(op), op.dim, element(src(op), op.dim)), ctranspose(op.op), op.dim, VIEW)
