# leastsquares.jl

########################
# Generic least squares
########################


function leastsquares_matrix(set::FunctionSet, pts)
    @assert length(set) <= length(pts)
    evaluation_matrix(set, pts)
end

function leastsquares_operator(s::Span; samplingfactor = 2, options...)
    if has_grid(s)
        set2 = resize(s, samplingfactor*length(s))
        ls_grid = grid(set2)
    else
        ls_grid = EquispacedGrid(samplingfactor*length(s), left(set(s)), right(set(s)))
    end
    leastsquares_operator(s, ls_grid; options...)
end

leastsquares_operator(s::Span, grid::AbstractGrid; options...) =
    leastsquares_operator(s, DiscreteGridSpace(grid, coeftype(s)); options...)

function leastsquares_operator(s::Span, dgs::DiscreteGridSpace; options...)
    if has_grid(s)
        larger_set = resize(s, size(dgs))
        if grid(larger_set) == grid(dgs) && has_transform(larger_set, dgs)
            R = restriction_operator(larger_set, s; options...)
            T = full_transform_operator(dgs, larger_s; options...)
            R * T
        else
            default_leastsquares_operator(s, dgs; options...)
        end
    else
        default_leastsquares_operator(s, dgs; options...)
    end
end

function default_leastsquares_operator(s::Span, dgs::DiscreteGridSpace; options...)
    SolverOperator(dgs, s, qrfact(evaluation_matrix(set(s), grid(dgs))))
end
