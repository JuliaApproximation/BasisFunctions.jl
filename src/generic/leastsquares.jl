# leastsquares.jl

########################
# Generic least squares
########################


function leastsquares_matrix(dict::Dictionary, pts)
    @assert length(dict) <= length(pts)
    evaluation_matrix(dict, pts)
end

function leastsquares_operator(s::Dictionary; samplingfactor = 2, options...)
    if has_grid(s)
        dict2 = resize(s, samplingfactor*length(s))
        ls_grid = grid(dict2)
    else
        ls_grid = EquispacedGrid(samplingfactor*length(s), support(s))
    end
    leastsquares_operator(s, ls_grid; options...)
end

leastsquares_operator(s::Dictionary, grid::AbstractGrid; options...) =
    leastsquares_operator(s, gridbasis(grid, coeftype(s)); options...)

function leastsquares_operator(s::Dictionary, dgs::GridBasis; options...)
    if has_grid(s)
        larger_dict = resize(s, size(dgs))
        if grid(larger_dict) == grid(dgs) && has_transform(larger_dict, dgs)
            R = restriction_operator(larger_dict, s; options...)
            T = full_transform_operator(dgs, larger_s; options...)
            R * T
        else
            default_leastsquares_operator(s, dgs; options...)
        end
    else
        default_leastsquares_operator(s, dgs; options...)
    end
end

default_leastsquares_operator(s::Dictionary, dgs::GridBasis; options...) =
    QR_solver(MultiplicationOperator(s, dgs, leastsquares_matrix(s, grid(dgs))))
