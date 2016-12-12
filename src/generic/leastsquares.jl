# leastsquares.jl

########################
# Generic least squares
########################


function leastsquares_matrix(set::FunctionSet, pts)
    @assert length(set) <= length(pts)
    evaluation_matrix(set, pts)
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
            R * T
        else
            default_leastsquares_operator(set, dgs; options...)
        end
    else
        default_leastsquares_operator(set, dgs; options...)
    end
end

function default_leastsquares_operator(set::FunctionSet, dgs::DiscreteGridSpace; options...)
    SolverOperator(dgs, set, qrfact(evaluation_matrix(set, grid(dgs))))
end
