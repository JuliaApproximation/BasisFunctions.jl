
function test_interval_grid(grid::AbstractGrid, show_timings=false)
    test_generic_grid(grid, show_timings=show_timings)
    T = eltype(grid)
    g1 = rescale(rescale(grid, -T(10), T(3)), leftendpoint(grid), rightendpoint(grid))
    @test 1+leftendpoint(g1) ≈ 1+leftendpoint(grid) && 1+rightendpoint(g1) ≈ 1+rightendpoint(grid)
end

function test_generic_grid(grid; show_timings=false)
    L = length(grid)

    T = eltype(grid)
    FT = float_type(T)

    grid_iterator(grid)

    # Test two types of iterations over a grid.
    # Make sure there are L elements. Do some computation on each point,
    # and make sure the results are the same.
    (l1,sum1) = grid_iterator1(grid)
    @test l1 == L
    (l2,sum2) = grid_iterator2(grid)
    @test l2 == L
    @test sum1 ≈ sum2

    if typeof(grid) <: AbstractGrid1d
        # Make sure that 1d grids return points, not vectors with one point.
        @test eltype(grid) <: Number
    end

    if hasextension(grid)
        g_ext = extend(grid, 2)
        for i in 1:length(grid)
            @test grid[i] ≈ g_ext[2i-1]
        end
    end

    if show_timings
        t = @timed grid_iterator1(grid)
        t = @timed grid_iterator1(grid)
        print_with_color(:blue, "Eachindex: ")
        println(t[3], " bytes in ", t[2], " seconds")
        t = @timed grid_iterator2(grid)
        print_with_color(:blue, "Each x in grid: ")
        println(t[3], " bytes in ", t[2], " seconds")
    end
end

function grid_iterator(grid)
    for (i,j) in zip(1:length(grid), eachindex(grid))
        @test BasisFunctions.unsafe_getindex(grid, i) == grid[i] == grid[j]
    end
end

function grid_iterator1(grid)
    l = 0
    s = zero(float_type(eltype(grid)))
    for i in eachindex(grid)
        x = grid[i]
        l += 1
        s += sum(x)
    end
    (l,s)
end

function grid_iterator2(grid)
    l = 0
    s = zero(float_type(eltype(grid)))
    for x in grid
        l += 1
        s += sum(x)
    end
    (l,s)
end
