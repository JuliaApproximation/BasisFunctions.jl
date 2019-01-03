
#####################
# Generic evaluation
#####################

# Compute the evaluation matrix of the given dict on the given set of points
# (a grid or any iterable set of points)
function evaluation_matrix(dict::Dictionary, pts, T = codomaintype(dict))
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

# By default we evaluate on the associated grid (if any, otherwise this gives an error)
# TODO: try to get rid of the oversampling again, that should go to FrameFun
evaluation_operator(s::Dictionary; oversampling = default_oversampling(s), options...) =
    evaluation_operator(s, oversampled_grid(s, oversampling); options...)

# Convert a grid to a GridBasis
evaluation_operator(s::Dictionary, grid::AbstractGrid; options...) =
    evaluation_operator(s, GridBasis(s, grid); options...)

# Convert a linear range to an equispaced grid
evaluation_operator(s::Dictionary, r::LinRange; options...) =
    evaluation_operator(s, EquispacedGrid(r))

# For easier dispatch, if the destination is a GridBasis we add the grid as parameter
evaluation_operator(s::Dictionary, dgs::GridBasis; options...) =
    grid_evaluation_operator(s, dgs, grid(dgs); options...)

function dense_evaluation_operator(s::Dictionary, dgs::GridBasis;
            T = op_eltype(s,dgs), options...)
    A = evaluation_matrix(s, grid(dgs), T)
    MultiplicationOperator(s, dgs, A)
end

# Evaluate s in the grid of dgs
# We try to see if any fast transform is available
# We need to intercept grid_evaluation_operator both for DerivedDicts and ComplexifiedDicts, hence the select_ routine.
grid_evaluation_operator(s::Dictionary, dgs::GridBasis, grid::AbstractGrid; options...) =
    select_grid_evaluation_operator(s, dgs, grid; options...)

function select_grid_evaluation_operator(s::Dictionary, dgs::GridBasis, grid::AbstractGrid;
            warnslow = false, options...)
    if has_transform(s)
        if has_transform(s, dgs)
            return transform_operator(s, dgs; options...)
        elseif length(s) < length(dgs)
            if has_extension(s)
                if dimension(s) == 1
                    slarge = resize(s, length(dgs))
                    if has_transform(slarge, dgs)
                        return transform_operator(slarge, dgs; options...) * extension_operator(s, slarge; options...)
                    else
                        if warnslow
                            println("Has transform, but no match after resize, dim 1")
                            println(slarge)
                            println(dgs)
                            @warn "Slow evaluation operator selected"
                        end
                        return dense_evaluation_operator(s, dgs; options...)
                    end
                # The basis should at least be resizeable to the dimensions of the grid
                elseif dimension(s) == length(size(dgs))
                    slarge = resize(s, size(dgs))
                    if has_transform(slarge, dgs)
                        return transform_operator(slarge, dgs; options...) * extension_operator(s, slarge; options...)
                    else
                        if warnslow
                            println("Has transform, but no match after resize, dim > 1")
                            println(slarge)
                            println(dgs)
                            @warn "Slow evaluation operator selected"
                        end
                        return dense_evaluation_operator(s, dgs; options...)
                    end
                else
                    if warnslow
                        println("Apparent dimension mismatch")
                        println(s)
                        println(dgs)
                        @warn "Slow evaluation operator selected"
                    end
                    return dense_evaluation_operator(s, dgs; options...)
                end
            else
                if warnslow
                    println("No extension available")
                    println(s)
                    println(dgs)
                    @warn "Slow evaluation operator selected"
                end
                return dense_evaluation_operator(s, dgs; options...)
            end
        else
            # This might be faster implemented by:
            #   - finding an integer n so that nlength(dgs)>length(s)
            #   - resorting to the above evaluation + extension
            #   - subsampling by factor n
            if warnslow
                println("Length of grid smaller than length of dictionary")
                println(s)
                println(dgs)
                @warn "Slow evaluation operator selected"
            end
            return dense_evaluation_operator(s, dgs; options...)
        end
    else
        if warnslow
            println("No transform available")
            println(s)
            println(dgs)
            @warn "Slow evaluation operator selected"
        end
        return dense_evaluation_operator(s, dgs; options...)
    end
    @warn "This code path should not be reached. Case missed?"
    println(s)
    println(dgs)
    dense_evaluation_operator(s, dgs; options...)
end

grid_evaluation_operator(s::Dictionary, dgs::GridBasis, subgrid::AbstractSubGrid; options...) =
    _grid_evaluation_operator(s, dgs, subgrid; options...)

# Try to do efficient evaluation also for subgrids
function _grid_evaluation_operator(s::Dictionary, dgs::GridBasis, subgrid::AbstractSubGrid; options...)
    # We make no attempt if the set has no associated grid
    if has_interpolationgrid(s)
        # Is the associated grid of the same type as the supergrid at hand?
        if typeof(interpolation_grid(s)) == typeof(supergrid(subgrid))
            # It is: we can use the evaluation operator of the supergrid
            super_dgs = GridBasis(s, supergrid(subgrid))
            E = evaluation_operator(s, super_dgs; options...)
            R = restriction_operator(super_dgs, dgs; options...)
            R*E
        else
            dense_evaluation_operator(s, dgs; options...)
        end
    else
        dense_evaluation_operator(s, dgs; options...)
    end
end

# By default we evaluate on the associated grid (if any, otherwise this gives an error)
discrete_dual_evaluation_operator(s::Dictionary; oversampling = default_oversampling(s), options...) =
    grid_evaluation_operator(s, GridBasis(s, oversampled_grid(s, oversampling)), oversampled_grid(s, oversampling); options...)*DiscreteDualGram(s; oversampling=oversampling)


new_evaluation_operator(dict::Dictionary, grid::AbstractGrid; options...) =
    new_evaluation_operator(dict, GridBasis(dict, grid); options...)

new_evaluation_operator(dict::Dictionary, grid::AbstractSubGrid; T = coefficienttype(dict), options...) =
     restriction_operator(GridBasis{T}(supergrid(grid)), GridBasis{T}(grid); options...) * new_evaluation_operator(dict, supergrid(grid); options...)

new_evaluation_operator(dict::Dictionary, gb::GridBasis;
            T = op_eltype(dict, gb), options...) =
    new_evaluation_operator(dict, gb, grid(gb); T=T, options...)


function resize_and_transform(dict::Dictionary, gb::GridBasis, grid;
            warnslow = false, options...)
    if size(dict) == size(grid)
        transform_to_grid(dict, gb, grid; options...)
    elseif length(grid) > length(dict)
        dlarge = resize(dict, size(grid))
        transform_to_grid(dlarge, gb, grid; options...) * extension_operator(dict, dlarge; options...)
    else
        if warnslow
            @warn "Resize and transform: dictionary evaluated in small grid"
        end
        dense_evaluation_operator(dict, gb; options...)
    end
end
