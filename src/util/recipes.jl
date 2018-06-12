
## Plotting recipe for a setexpansion:
# - determine the plotting grid(depending on the basis)
# - evaluate the expansion in the gridpoints (fast if possible)
# - postprocess the data

@recipe function f(S::Expansion; plot_complex = false, n=200)
    legend --> false
    # title --> "Expansion"
    grid = plotgrid(dictionary(S), n)
    vals = plot_complex ? S(grid) : real(S(grid))
    grid, postprocess(dictionary(S), grid, vals)
end

# When a target function is provided, plot the error
# dispatch on dimension to set the logscale
@recipe function f(S::Expansion, target::Function; n=200)
    grid = plotgrid(dictionary(S), n)
    # Determine the return type so we know where to sample
    origvals = sample(grid, target, eltype(S))
    vals = abs.(origvals - S(grid))
    dictionary(S), grid, postprocess(dictionary(S), grid, vals)
end
# 1D error plot
@recipe function f(S::Dictionary1d, grid::AbstractGrid, vals)
    title --> "Error"
    legend --> false
    yscale --> :log10
    ylims --> [.1eps(real(codomaintype(S))),Inf]
    vals[vals.==0] = eps(real(codomaintype(S)))^2
    # yaxis --> (:log10, (,Inf))
    grid, vals
end
# 2D error plot
@recipe function f(S::Dictionary2d, grid::AbstractGrid, vals)
    title --> "Error (log)"
    seriestype --> :heatmap
    grid, log10.(real(vals))
end


# Plot a vector of values on a 1D grid
@recipe function f(grid::AbstractGrid1d, vals)
    size --> (800,400)
    collect(grid), vals
end

# Plot a matrix of values on a 2D equispaced grid
@recipe function f(grid::AbstractGrid2d, vals)
    seriestype --> :surface
    size --> (500,400)
    xrange = linspace(leftendpoint(grid)[1],rightendpoint(grid)[1],size(grid,1))
    yrange = linspace(leftendpoint(grid)[2],rightendpoint(grid)[2],size(grid,2))
    xrange, yrange, vals'
end

# Plot an Nd grid
@recipe function f(grid::AbstractGrid)
    seriestype --> :scatter
    size --> (500,400)
    legend --> false
    broadcast(x->tuple(x...),collect(grid))
end

# Plot a 1D grid
@recipe function f(grid::AbstractGrid1d)
    seriestype --> :scatter
    yticks --> []
    ylims --> [-1 1]
    size --> (500,200)
    legend --> false
    collect(grid), zeros(size(grid))
end

# Plot a Dictionary
@recipe function f(F::Dictionary; plot_complex = false, n=200)
    for i in eachindex(F)
        @series begin
            grid = plotgrid(F[i],n)
            vals = plot_complex? F[i](grid) : real(F[i](grid))
            grid, postprocess(F[i],grid,vals)
        end
    end
    nothing
end

#
# For regular Expansions, no postprocessing is needed
postprocess(S::Dictionary, grid, vals) = vals

# For function subsets, revert to the underlying Dictionary for postprocessing
postprocess(S::Subdictionary, grid, vals) = postprocess(superdict(S), grid, vals)

## Plotting grids
# Always plot on equispaced grids for the best plotting resolution
plotgrid(S::Dictionary1d, n) = rescale(PeriodicEquispacedGrid(n,domaintype(S)),infimum(support(S)),supremum(support(S)))

# NOTE: This only supoorts multi-dimensional tensor product dicts.
plotgrid(F::Dictionary{Tuple{S1,S2},T}, n) where {S1,S2,T} = plotgrid(element(F,1),n)×plotgrid(element(F,2),n)
plotgrid(F::Subdictionary, n) = plotgrid(superdict(F), n)
## Split complex plots in real and imaginary parts
# 1D
@recipe function f(A::AbstractArray{S}, B::Array{Complex{T}}) where {S<:Real, T<:Real}
    # Force double layout
    layout := 2
    # Legend is useless here
    legend --> false
    @series begin
        subplot := 1
        title := "Real"
        A,real(B)
    end
    @series begin
        subplot :=2
        title := "Imaginary"
        A,imag(B)
    end
    nothing
end

# 2D
@recipe function f(A::LinSpace{S},B::LinSpace{S},C::Array{T}) where {S<:Real,T<:Complex}
    # Force double layout
    layout := 2
    # Legend is useless here
    legend --> false
    @series begin
        subplot := 1
        title := "Real"
        A,B,real(C)
    end
    @series begin
        subplot :=2
        title := "Imaginary"
        A,B,imag(C)
    end
    nothing
end
