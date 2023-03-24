
## Plotting recipe for a setexpansion:
# - determine the plotting grid(depending on the basis)
# - evaluate the expansion in the gridpoints (fast if possible)
# - postprocess the data

@recipe function f(S::Expansion; plot_complex = false, n=200)
    legend --> false
    # title --> "Expansion"
    grid = plotgrid(dictionary(S), n)
    vals = plot_complex ? S.(grid) : real(S.(grid))
    grid, postprocess(dictionary(S), grid, vals)
end

# When a target function is provided, plot the error
# dispatch on dimension to set the logscale
@recipe function f(S::Expansion, target::Function; n=200)
    grid = plotgrid(dictionary(S), n)
    # Determine the return type so we know where to sample
    origvals = sample(grid, target, codomaintype(S))
    vals = abs.(origvals - S.(grid))
    dictionary(S), grid, postprocess(dictionary(S), grid, vals)
end
# 1D error plot
@recipe function f(S::Dictionary1d, grid::AbstractGrid, vals)
    title --> "Error"
    legend --> false
    yscale --> :log10
    ylims --> [.1eps(real(codomaintype(S))),Inf]
    vals[vals.==0] .= eps(real(codomaintype(S)))^2
    # yaxis --> (:log10, (,Inf))
    grid, vals
end
# 2D error plot
@recipe function f(S::Dictionary2d, grid::AbstractGrid, vals)
    title --> "Error (log)"
    seriestype --> :heatmap
    grid, log10.(real(vals))
end

# Plot a Dictionary
@recipe function f(F::Dictionary; plot_complex = false, n=200)
    for i in eachindex(F)
        @series begin
            grid = plotgrid(F[i],n)
            z = F[i].(grid)
            vals = plot_complex ? z : real(z)
            grid, postprocess(F[i],grid,vals)
        end
    end
    nothing
end

# Plot a basis function
@recipe function f(φ::AbstractBasisFunction; plot_complex = false, n=200)
    @series begin
        grid = plotgrid(φ,n)
        z = φ.(grid)
        vals = plot_complex ? z : real(z)
        grid, postprocess(φ,grid,vals)
    end
    nothing
end

#
# For regular Expansions, no postprocessing is needed
function postprocess(S::Dictionary, grid, vals)
    D = support(S)
    I = grid .∉ Ref(D)
    vals[I] .= Inf
    vals
end

# For function subsets, revert to the underlying Dictionary for postprocessing
postprocess(S::Subdictionary, grid, vals) = postprocess(superdict(S), grid, vals)
postprocess(φ::AbstractBasisFunction, grid, vals) = postprocess(dictionary(φ), grid, vals)

## Plotting grids
# Always plot on equispaced grids for the best plotting resolution
plotgrid(S::Dictionary1d, n) = EquispacedGrid(n, support(S))
# look at first element by default
plotgrid(S::MultiDict, n) = plotgrid(component(S,1), n)
plotgrid(S::DerivedDict, n) = plotgrid(superdict(S),n)
# NOTE: This only supports multi-dimensional tensor product dicts.
plotgrid(F::TensorProductDict2, n) = plotgrid(component(F,1),n)×plotgrid(component(F,2),n)
plotgrid(F::OperatedDict, n) = plotgrid(superdict(F), n)
plotgrid(F::Subdictionary, n) = plotgrid(superdict(F), n)
plotgrid(φ::AbstractBasisFunction, n) = plotgrid(dictionary(φ), n)

# generic fallback for suitable plotting grids
plotgrid(S::Dictionary, n) = plotgrid_domain(n, support(S))
plotgrid_domain(n::Int, domain::AbstractInterval) = EquispacedGrid(n, domain)
plotgrid_domain(n::Int, domain::ProductDomain) =
    productgrid(map(d->plotgrid_domain(n, d), factors(domain))...)
function plotgrid_domain(n::Int, domain)
    box = boundingbox(domain)
    g = plotgrid_domain(n, box)
    MaskedGrid(g, domain)
end

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
        A,real.(B)
    end
    @series begin
        subplot :=2
        title := "Imaginary"
        A,imag.(B)
    end
    nothing
end

# 2D
@recipe function f(A::LinRange{S},B::LinRange{S},C::Array{T}) where {S<:Real,T<:Complex}
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


@recipe function f(dom::Domain2d; n=300, distance=false, xlim=[-1,1], ylim=[-1,1])
    seriescolor --> :tempo
     seriestype --> :heatmap
    aspect_ratio --> 1
    colorbar --> false
    # xrange = linspace(xlim[1],xlim[2],n)
    xrange = EquispacedGrid(n,xlim[1],xlim[2])
    # yrange = linspace(ylim[1],ylim[2],n)
    yrange = EquispacedGrid(n,ylim[1],ylim[2])
    # grid = [SVector(i,j) for i in xrange , j in yrange]
    grid = xrange×yrange

    plotdata = distance ? dist.(grid, Ref(dom)) : in.(grid, Ref(dom))
    collect(xrange),collect(yrange),plotdata'
end
