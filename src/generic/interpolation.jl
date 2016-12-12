# interpolation.jl

########################
# Generic interpolation
########################

function interpolation_matrix(set::FunctionSet, pts)
    @assert length(set) == length(pts)
    evaluation_matrix(set, pts)
end

interpolation_operator(set::FunctionSet; options...) =
    interpolation_operator(set, grid(set); options...)

interpolation_operator(set::FunctionSet, grid::AbstractGrid; options...) =
    interpolation_operator(set, gridspace(set, grid); options...)

# Interpolate set in the grid of dgs
function interpolation_operator(set::FunctionSet, dgs::DiscreteGridSpace; options...)
    if has_grid(set) && grid(set) == grid(dgs) && has_transform(set, dgs)
        full_transform_operator(dgs, set; options...)
    else
        default_interpolation_operator(set, dgs; options...)
    end
end

function default_interpolation_operator(set::FunctionSet, dgs::DiscreteGridSpace; options...)
    SolverOperator(dgs, set, qrfact(evaluation_matrix(set, grid(dgs))))
end

function interpolate(set::FunctionSet, pts, f)
    A = interpolation_matrix(set, pts)
    B = eltype(A)[f(x...) for x in pts]
    SetExpansion(set, A\B)
end
