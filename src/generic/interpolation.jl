
########################
# Generic interpolation
########################

function interpolation_matrix(dict::Dictionary, pts; T=coefficienttype(dict))
    @assert length(dict) == length(pts)
    evaluation_matrix(dict, pts; T=T)
end

interpolation_operator(s::Dictionary; options...) =
    interpolation_operator(s, interpolation_grid(s); options...)

interpolation_operator(s::Dictionary, grid::AbstractGrid; options...) =
    interpolation_operator(s, GridBasis(s, grid); options...)

# Interpolate dict in the grid of dgs
function interpolation_operator(s::Dictionary, dgs::GridBasis; options...)
    if hasinterpolationgrid(s) && interpolation_grid(s) == grid(dgs) && hastransform(s, dgs)
        transform_operator(dgs, s; options...)
    else
        default_interpolation_operator(s, dgs; options...)
    end
end

default_interpolation_operator(s::Dictionary, dgs::GridBasis; T=op_eltype(s,dgs), options...) =
    QR_solver(ArrayOperator(evaluation_matrix(s, grid(dgs); T=T), s, dgs))


function interpolate(s::Dictionary, pts, f; T=coefficienttype(s))
    A = interpolation_matrix(s, pts; T=T)
    B = coefficienttype(s)[f(x...) for x in pts]
    Expansion(s, A\B)
end
