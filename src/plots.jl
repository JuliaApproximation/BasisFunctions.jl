# plots.jl

## function plot(f::SetExpansion)
##     fset = set(f)
##     grid = plot_grid(fset)
##     data = f(grid)
##     plot(grid, data)
## end

## plot_grid(set::FunctionSet{1}) = EquispacedGrid(100, left(set), right(set))

## plot_grid{N}(set::FunctionSet{N}) = TensorProductGrid(tuple([EquispacedGrid(100, left(set,dim), right(set,dim)) for dim = 1:N]...))


## plot(grid::AbstractGrid1d, data::Array) = gadflyplot(grid, data)

## function gadflyplot{T <: Real}(grid::AbstractGrid1d, data::Array{T})
## #    require("GadFly")
##     Main.Gadfly.plot(x = range(grid), y = data)
## end

## gadflyplot{T <: Real}(grid::AbstractGrid1d, data::Array{Complex{T}}) = gadflyplot(grid, real(data))


## plot(grid::AbstractGrid2d, data::Array) = pyplot(grid, data)

## function pyplot{T <: Real}(grid::AbstractGrid{2}, data::Array{T})
##     Main.PyPlot.surf(range(grid,1), range(grid, 2), data)
## end

## pyplot{T <: Real}(grid::AbstractGrid, data::Array{Complex{T}}) = pyplot(grid, real(data))


function plot{S <: FunctionSet{2}}(s::SetExpansion{S} ; n=1000)
    B = boundingbox(domain(set(s)))
    Tgrid = equispaced_grid(B,n)
    Mgrid=MaskedGrid(Tgrid, domain(set(s)))
    data = convert(Array{Float64},real(s(Mgrid)))
    x=[Mgrid[i][1] for i = 1:length(Mgrid)]
    y=[Mgrid[i][2] for i = 1:length(Mgrid)]
    Main.PyPlot.plot_trisurf(x,y,data)
end

function plot_image{S <: FunctionSet{2}}(s::SetExpansion{S};n=300, vmin=0, vmax=0)
    Tgrid = grid(similar(s,eltype(f),(n,n)))
    Z = evalgrid(Tgrid, d)
    data = real(expansion(f)(Tgrid))
    # If no explicit scaling information is provided, scale to the data
    if vmin == vmax
        vmin = minimum(data)
        vmax = maximum(data)
    end
    plot_image(data, Tgrid, vmin, vmax)
end

function plot_image{T,TG,GN,ID}(data::AbstractArray{T,2},grid::TensorProductGrid{TG,GN,ID,2}, vmin, vmax)
    X = convert(Array{Float64},real(apply((x,y)->x,Complex{Float64},Tgrid)))
    Y = convert(Array{Float64},real(apply((x,y)->y,Complex{Float64},Tgrid)))
    Main.PyPlot.pcolormesh(X,Y,data,vmin=vmin, vmax=vmax, shading = "gouraud")
    
    Main.PyPlot.axis("scaled")
    Main.PyPlot.xlim([left(Tgrid)[1], right(Tgrid)[1]])
    Main.PyPlot.ylim([left(Tgrid)[2], right(Tgrid)[2]])
    Main.PyPlot.colorbar()
end


function plot_image{S <: FunctionSet{2}}(s::SetExpansion{S}, g::Function; n=300)
    Tgrid = grid(similar(s,eltype(f),(n,n)))
    Z = evalgrid(Tgrid, d)
    data = apply(g,Complex{Float64},Tgrid)
    plot_image(data, Tgrid)
end

function plot_error{S <: FunctionSet{2}}(s::SetExpansion{S}, g::Function; n=300)
    Tgrid = grid(similar(s,eltype(f),(n,n)))
    Z = evalgrid(Tgrid, d)
    data = real(expansion(f)(Tgrid))-apply(g,Complex{Float64},Tgrid)
    plot_image(log10(abs(data)),Tgrid, -16, 1)
end
function plot_grid(grid::AbstractGrid2d)
    x=[grid[i][1] for i = 1:length(grid)]
    y=[grid[i][2] for i = 1:length(grid)]
    Main.PyPlot.plot(x,y,linestyle="none",marker="o",color="black",mec="black",markersize=1.5)
    Main.PyPlot.axis("equal")
end

function plot_grid(grid::AbstractGrid3d)
    x=[grid[i][1] for i = 1:length(grid)]
    y=[grid[i][2] for i = 1:length(grid)]
    z=[grid[i][3] for i = 1:length(grid)]
    Main.PyPlot.plot3D(x,y,z,linestyle="none",marker="o",color="blue")
    Main.PyPlot.axis("equal")
end

function plot_grid{TG,GN,ID}(grid::TensorProductGrid{TG,GN,ID,2})
    dom = Cube(left(grid),right(grid))
    Mgrid = MaskedGrid(grid,dom)
    plot_grid(Mgrid)
end

function plot_expansion{S <: FunctionSet{1}}(s::SetExpansion{S}; n=201, repeats=0)
    G = grid(similar(set(s),eltype(s),n))
    data = convert(Array{Float64},real(s(G)))
    x = convert(Array{Float64},apply(x->x,eltype(s),G))
    for i=-repeats:repeats
        Main.PyPlot.plot(x+i*(right(G)-left(G)),data,linestyle="dashed",color="blue")
    end
    Main.PyPlot.plot(x,data,color="blue")
    Main.PyPlot.title("Extension (Full)")
end


function apply(f::Function, return_type, g::AbstractGrid)
    result = Array(return_type, size(g))
    call!(f, result, g)
    result
end

function call!{N}(f::Function, result::AbstractArray, g::AbstractGrid{N})
    for i in eachindex(g)
        result[i] = f(getindex(g, i)...)
    end
end
