
## Plotting recipe for a setexpansion:
# - determine the plotting grid(depending on the basis)
# - evaluate the expansion in the gridpoints (fast if possible)
# - postprocess the data

@recipe function f(S::SetExpansion; n=200)
    legend --> :false
    title --> "SetExpansion"
    grid = plotgrid(set(S), n)
    vals = real(S(grid))
    grid, postprocess(set(S), grid, vals)
end

# When a target function is provided, plot the error
# dispatch on dimension to set the logscale
@recipe function f(S::SetExpansion, target::Function; n=200)
    grid = plotgrid(set(S), n)
    origvals = reshape(eltype(S)[real(target(x...)) for x in grid],size(grid))
    vals = abs(origvals - S(grid))
    set(S), grid, postprocess(set(S), grid, vals)
end
# 1D error plot
@recipe function f(S::FunctionSet{1}, grid::AbstractGrid, vals)
    title --> "Error"
    legend --> false
    yscale --> :log10
    grid, vals
end
# 2D error plot
@recipe function f(S::FunctionSet{2}, grid::AbstractGrid, vals)
    title --> "Error (log)"
    seriestype --> :heatmap
    grid, log10(real(vals))
end


# Plot a vector of values on a 1D grid
@recipe function f(grid::AbstractGrid{1}, vals)
    size --> (800,400)
    collect(grid), vals
end

# Plot a matrix of values on a 2D equispaced grid
@recipe function f(grid::AbstractGrid{2}, vals)
    seriestype --> :surface
    size --> (500,400)
    xrange = linspace(left(grid)[1],right(grid)[1],size(grid,1))
    yrange = linspace(left(grid)[2],right(grid)[2],size(grid,2))
    xrange, yrange, vals'
end

# 
# For regular SetExpansions, no postprocessing is needed 
postprocess(S::FunctionSet, grid, vals) = vals

# For the 1D case, we can just use the standard set grid
plotgrid(S::FunctionSet{1}, n) = grid(resize(S, n))

# For the 2D case a rectangular grid is needed for efficient plotting.
plotgrid(S::FunctionSet{2}, n) = rescale(PeriodicEquispacedGrid(n),left(S,1),right(S,1))⊗rescale(PeriodicEquispacedGrid(n),left(S,2),right(S,2))

