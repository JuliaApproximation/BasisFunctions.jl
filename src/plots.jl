# plots.jl

function plot(f::SetExpansion)
    fset = set(f)
    grid = plot_grid(fset)
    data = f(grid)
    plot(grid, data)
end

plot_grid(set::FunctionSet{1}) = EquispacedGrid(100, left(set), right(set))

plot_grid{N}(set::FunctionSet{N}) = TensorProductGrid(tuple([EquispacedGrid(100, left(set,dim), right(set,dim)) for dim = 1:N]...))


plot(grid::AbstractGrid1d, data::Array) = gadflyplot(grid, data)

function gadflyplot{T <: Real}(grid::AbstractGrid1d, data::Array{T})
#    require("GadFly")
    Main.Gadfly.plot(x = range(grid), y = data)
end

gadflyplot{T <: Real}(grid::AbstractGrid1d, data::Array{Complex{T}}) = gadflyplot(grid, real(data))


plot(grid::AbstractGrid2d, data::Array) = pyplot(grid, data)

function pyplot{T <: Real}(grid::AbstractGrid{2}, data::Array{T})
    Main.PyPlot.surf(range(grid,1), range(grid, 2), data)
end

pyplot{T <: Real}(grid::AbstractGrid, data::Array{Complex{T}}) = pyplot(grid, real(data))


