# evaluation.jl

#####################
# Generic evaluation
#####################

# Compute the evaluation matrix of the given dict on the given set of points
# (a grid or any iterable set of points)
function evaluation_matrix(dict::Dictionary, pts)
    a = Array{codomaintype(dict)}(length(pts), length(dict))
    evaluation_matrix!(a, dict, pts)
end

function evaluation_matrix!(a::AbstractMatrix, dict::Dictionary, pts)
    @assert size(a,1) == length(pts)
    @assert size(a,2) == length(dict)

    for (j,ϕ) in enumerate(dict), (i,x) in enumerate(pts)
        a[i,j] = ϕ(x)
    end
    a
end

# By default we evaluate on the associated grid (if any, otherwise this gives an error)
# TODO: try to get rid of the oversampling again, that should go to FrameFun
evaluation_operator(s::Span; oversampling = default_oversampling(dictionary(s)), options...) =
    evaluation_operator(s, oversampled_grid(dictionary(s), oversampling); options...)

# Convert a grid to a DiscreteGridSpace
evaluation_operator(s::Span, grid::AbstractGrid; options...) =
    evaluation_operator(s, gridspace(s, grid); options...)

# Convert a linear range to an equispaced grid
evaluation_operator(s::Span, r::LinSpace; options...) =
    evaluation_operator(s, EquispacedGrid(r))

# For easier dispatch, if the destination is a DiscreteGridSpace we add the grid as parameter
evaluation_operator(s::Span, dgs::DiscreteGridSpace; options...) =
    grid_evaluation_operator(s, dgs, grid(dgs); options...)

default_evaluation_operator(s::Span, dgs::DiscreteGridSpace; options...) =
    MultiplicationOperator(s, dgs, evaluation_matrix(dictionary(s), grid(dgs)))

# Evaluate s in the grid of dgs
# We try to see if any fast transform is available
function grid_evaluation_operator(s::Span, dgs::DiscreteGridSpace, grid::AbstractGrid; options...)
    fs = dictionary(s)
    if has_transform(s)
        if has_transform(s, dgs)
            full_transform_operator(s, dgs; options...)
        elseif length(s) < length(dgs)
            if dimension(s) == 1
                slarge = resize(s, length(dgs))
                (has_transform(slarge, dgs) && has_extension(s)) && return (full_transform_operator(slarge, dgs; options...) * extension_operator(s, slarge; options...))
            # The basis should at least be resizeable to the dimensions of the grid
            elseif dimension(s) == length(size(dgs))
                slarge = resize(s, size(dgs))
                has_transform(slarge, dgs) && return (full_transform_operator(slarge, dgs; options...) * extension_operator(s, slarge; options...))
            end
            return default_evaluation_operator(s, dgs; options...)
        else
            # This might be faster implemented by:
            #   - finding an integer n so that nlength(dgs)>length(s)
            #   - resorting to the above evaluation + extension
            #   - subsampling by factor n
            default_evaluation_operator(s, dgs; options...)
        end
    else
        default_evaluation_operator(s, dgs; options...)
    end
end

# Try to do efficient evaluation also for subgrids
function grid_evaluation_operator(s::Span, dgs::DiscreteGridSpace, subgrid::AbstractSubGrid; options...)
    # We make no attempt if the set has no associated grid
    if has_grid(s)
        # Is the associated grid of the same type as the supergrid at hand?
        if typeof(grid(s)) == typeof(supergrid(subgrid))
            # It is: we can use the evaluation operator of the supergrid
            super_dgs = gridspace(s, supergrid(subgrid))
            E = evaluation_operator(s, super_dgs; options...)
            R = restriction_operator(super_dgs, dgs; options...)
            R*E
        else
            default_evaluation_operator(s, dgs; options...)
        end
    else
        default_evaluation_operator(s, dgs; options...)
    end
end

# By default we evaluate on the associated grid (if any, otherwise this gives an error)
discrete_dual_evaluation_operator(s::Span; oversampling = default_oversampling(dictionary(s)), options...) =
    grid_evaluation_operator(s, gridspace(s, oversampled_grid(dictionary(s), oversampling)), oversampled_grid(dictionary(s), oversampling); options...)*DiscreteDualGram(s; oversampling=oversampling)
