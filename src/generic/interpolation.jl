# interpolation.jl

########################
# Generic interpolation
########################

function interpolation_matrix(set::FunctionSet, pts)
    @assert length(set) == length(pts)
    evaluation_matrix(set, pts)
end

interpolation_operator(s::Span; options...) =
    interpolation_operator(s, grid(s); options...)

interpolation_operator(s::Span, grid::AbstractGrid; options...) =
    interpolation_operator(s, gridspace(s, grid); options...)

# Interpolate set in the grid of dgs
function interpolation_operator(s::Span, dgs::DiscreteGridSpace; options...)
    if has_grid(s) && grid(s) == grid(dgs) && has_transform(s, dgs)
        full_transform_operator(dgs, s; options...)
    else
        default_interpolation_operator(s, dgs; options...)
    end
end

function default_interpolation_operator(s::Span, dgs::DiscreteGridSpace; options...)
    SolverOperator(dgs, s, qrfact(evaluation_matrix(set(s), grid(dgs))))
end

function interpolate(s::Span, pts, f)
    A = interpolation_matrix(set(s), pts)
    B = coeftype(s)[f(x...) for x in pts]
    SetExpansion(set(s), A\B)
end
