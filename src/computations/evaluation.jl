
#####################
# Generic evaluation
#####################

const BF_WARNSLOW = true

# Compute the evaluation matrix of the given dict on the given set of points
# (a grid or any iterable set of points)
function evaluation_matrix(dict::Dictionary, pts; T = codomaintype(dict))
    a = Array{T}(undef, length(pts), length(dict))
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

function dense_evaluation_operator(s::Dictionary, gb::GridBasis;
            T = op_eltype(s,gb), options...)
    A = evaluation_matrix(s, grid(gb); T=T)
    ArrayOperator(A, s, gb)
end

evaluation_operator(dict::Dictionary, grid::AbstractGrid; options...) =
    evaluation_operator(dict, GridBasis(dict, grid); options...)

evaluation_operator(dict::Dictionary, grid::AbstractSubGrid; T = coefficienttype(dict), options...) =
     restriction_operator(GridBasis{T}(supergrid(grid)), GridBasis{T}(grid); T=T, options...) * evaluation_operator(dict, supergrid(grid); T=T, options...)

evaluation_operator(dict::Dictionary, gb::GridBasis;
            T = op_eltype(dict, gb), options...) =
    grid_evaluation_operator(dict, gb, grid(gb); T=T, options...)

function grid_evaluation_operator(dict::Dictionary, gb::GridBasis, grid;
            T = op_eltype(dict, gb), options...)
    @debug "No fast evaluation available in $grid, using dense evaluation matrix instead."
    dense_evaluation_operator(dict, gb; T=T, options...)
end

function resize_and_transform(dict::Dictionary, gb::GridBasis, grid; options...)
    if size(dict) == size(grid)
        transform_to_grid(dict, gb, grid; options...)
    elseif length(grid) > length(dict)
        dlarge = resize(dict, size(grid))
        transform_to_grid(dlarge, gb, grid; options...) * extension_operator(dict, dlarge; options...)
    else
        @debug "Resize and transform: dictionary evaluated in small grid"
        dense_evaluation_operator(dict, gb; options...)
    end
end
