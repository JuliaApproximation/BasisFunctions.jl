# plots.jl

function plot{S <: FunctionSet1d}(s::SetExpansion{S}; n=201, repeats=0, color="blue", alpha=1.0, title = "")
    g = grid(resize(set(s), n))
    data = map(Float64, real(s(g)))
    x = collect(g)
    if repeats > 0
        for i = -repeats:repeats
            plot(x+i*(right(g)-left(g)), data, linestyle="dashed", color=color, alpha=alpha)
        end
    end
    plot(x, data, color=color, alpha=alpha)
    PyPlot.title(title)
end

function plot{S <: FunctionSet2d}(s::SetExpansion{S}; n=201)
    Tgrid = grid(resize(set(s), (n,n)))
    vals = s(Tgrid)
    data = Float64[real(v) for v in vals]
    pts = collect(Tgrid)
    x = [p[1] for p in pts]
    y = [p[2] for p in pts]
    plot_trisurf(x,y,data[:])
end

function plot_image{S <: FunctionSet{2}}(s::SetExpansion{S}; n=30)
    Tgrid = grid(resize(set(s), (n,n)))
    data = real(s(Tgrid))
    plot_image(data, Tgrid)
end

function plot_error{S <: FunctionSet2d}(s::SetExpansion{S}, f_orig::Function; n=100)
    Tgrid = grid(resize(s, (n,n)))
    data = real(s(Tgrid))
    data_orig = map(f_orig, Tgrid)
    plot_image(data, Tgrid)
end


function plot_image{T,TG}(data::AbstractArray{T,2}, grid::TensorProductGrid{TG,2},
    vmin = minimum(data), vmax = maximum(data))

    pts = collect(grid)
    x = [p[1] for p in pts]
    y = [p[2] for p in pts]
    X = reshape(x, size(grid))
    Y = reshape(y, size(grid))
    # X = convert(Array{Float64},real(apply((x,y)->x,Complex{Float64},grid)))
    # Y = convert(Array{Float64},real(apply((x,y)->y,Complex{Float64},grid)))
    pcolormesh(X,Y,data,vmin=vmin, vmax=vmax, shading = "gouraud")

    axis("scaled")
    xlim([left(grid)[1], right(grid)[1]])
    ylim([left(grid)[2], right(grid)[2]])
    colorbar()
end



function plot_grid(grid::AbstractGrid2d)
    pts = collect(grid)
    x = [p[1] for p in pts]
    y = [p[2] for p in pts]
    plot(x,y,linestyle="none",marker="o",color="black",mec="black",markersize=1.5)
    axis("equal")
end

function plot_grid(grid::AbstractGrid3d)
    pts = collect(grid)
    x = [p[1] for p in pts]
    y = [p[2] for p in pts]
    z = [p[3] for p in pts]
    plot3D(x,y,z,linestyle="none",marker="o",color="blue")
    axis("equal")
end
