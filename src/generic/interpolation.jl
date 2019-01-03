
########################
# Generic interpolation
########################

function interpolation_matrix(dict::Dictionary, pts)
    @assert length(dict) == length(pts)
    evaluation_matrix(dict, pts)
end

interpolation_operator(s::Dictionary; options...) =
    interpolation_operator(s, interpolation_grid(s); options...)

interpolation_operator(s::Dictionary, grid::AbstractGrid; options...) =
    interpolation_operator(s, GridBasis(s, grid); options...)

# Interpolate dict in the grid of dgs
function interpolation_operator(s::Dictionary, dgs::GridBasis; options...)
    if has_interpolationgrid(s) && interpolation_grid(s) == grid(dgs) && has_transform(s, dgs)
        transform_operator(dgs, s; options...)
    else
        default_interpolation_operator(s, dgs; options...)
    end
end

default_interpolation_operator(s::Dictionary, dgs::GridBasis; options...) =
    QR_solver(MultiplicationOperator(s, dgs, evaluation_matrix(s, grid(dgs))))


function interpolate(s::Dictionary, pts, f)
    A = interpolation_matrix(s, pts)
    B = coefficienttype(s)[f(x...) for x in pts]
    Expansion(s, A\B)
end
