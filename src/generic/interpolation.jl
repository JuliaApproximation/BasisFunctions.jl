# interpolation.jl

########################
# Generic interpolation
########################

function interpolation_matrix(dict::Dictionary, pts)
    @assert length(dict) == length(pts)
    evaluation_matrix(dict, pts)
end

interpolation_operator(s::Dictionary; options...) =
    interpolation_operator(s, grid(s); options...)

interpolation_operator(s::Dictionary, grid::AbstractGrid; options...) =
    interpolation_operator(s, gridbasis(s, grid); options...)

# Interpolate dict in the grid of dgs
function interpolation_operator(s::Dictionary, dgs::GridBasis; options...)
    if has_grid(s) && grid(s) == grid(dgs) && has_transform(s, dgs)
        full_transform_operator(dgs, s; options...)
    else
        default_interpolation_operator(s, dgs; options...)
    end
end

default_interpolation_operator(s::Dictionary, dgs::GridBasis; options...) = (VERSION < v"0.7-") ?
    SolverOperator(dgs, s, qrfact(evaluation_matrix(s, grid(dgs)))) :
    SolverOperator(dgs, s, qr(evaluation_matrix(s, grid(dgs))))


function interpolate(s::Dictionary, pts, f)
    A = interpolation_matrix(s, pts)
    B = coeftype(s)[f(x...) for x in pts]
    Expansion(s, A\B)
end
