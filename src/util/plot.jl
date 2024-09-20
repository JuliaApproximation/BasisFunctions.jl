
"Return a grid suitable for plotting a function in the given dictionary."
function plotgrid end
"Process plotting data before it is visualized."
function plot_postprocess end

# For regular Expansions, no postprocessing is needed
function plot_postprocess(S::Dictionary, grid, vals)
    D = support(S)
    I = grid .∉ Ref(D)
    vals[I] .= Inf
    vals
end

# For function subsets, revert to the underlying Dictionary for postprocessing
plot_postprocess(S::Subdictionary, grid, vals) = plot_postprocess(superdict(S), grid, vals)
plot_postprocess(φ::AbstractBasisFunction, grid, vals) = plot_postprocess(dictionary(φ), grid, vals)

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
