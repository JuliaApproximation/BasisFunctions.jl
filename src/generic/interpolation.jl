# interpolation.jl

########################
# Generic interpolation
########################

function interpolation_matrix(dict::Dictionary, pts)
    @assert length(dict) == length(pts)
    evaluation_matrix(dict, pts)
end

interpolation_operator(s::Span; options...) =
    interpolation_operator(s, grid(s); options...)

interpolation_operator(s::Span, grid::AbstractGrid; options...) =
    interpolation_operator(s, gridspace(s, grid); options...)

# Interpolate dict in the grid of dgs
function interpolation_operator(s::Span, dgs::DiscreteGridSpace; options...)
    if has_grid(s) && grid(s) == grid(dgs) && has_transform(s, dgs)
        full_transform_operator(dgs, s; options...)
    else
        default_interpolation_operator(s, dgs; options...)
    end
end

function default_interpolation_operator(s::Span, dgs::DiscreteGridSpace; options...)
    SolverOperator(dgs, s, qrfact(evaluation_matrix(dictionary(s), grid(dgs))))
end

function interpolate(s::Span, pts, f)
    A = interpolation_matrix(dictionary(s), pts)
    B = coeftype(s)[f(x...) for x in pts]
    Expansion(dictionary(s), A\B)
end
