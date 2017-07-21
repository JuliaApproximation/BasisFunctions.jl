# test_generic_grids.jl

#####
# Grids
#####

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


function test_generic_grid(grid)
    #println("Grid: ", typeof(grid))
    L = length(grid)

    T = eltype(grid)
    FT = float_type(T)

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

    if has_extension(grid)
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

function test_grids(T)

    ## Equispaced grids
    len = 50
    a = -T(1.2)
    b = T(3.5)
    g1 = EquispacedGrid(len, a, b)
    g2 = PeriodicEquispacedGrid(len, a-1, b+1)
    g3 = g1 × g2
    g4 = g1 × g3

    test_generic_grid(g1)
    test_generic_grid(g2)
    test_generic_grid(g3)
    test_generic_grid(g4)

    # Test a subgrid
    g5 = g1[10:20]
    @test g5[1] == g1[10]
    @test g5[11] == g1[20]
    @test length(g5) == 20-10+1
    test_generic_grid(g5)
    g6 = g1[10:2:20]
    @test g6[2] == g1[12]
    @test length(g6) == 6

    g = EquispacedGrid(len, a, b)
    idx = 5
    @test g[idx] ≈ a + (idx-1) * (b-a)/(len-1)
    @test g[len] ≈ b
    @test_throws BoundsError g[len+1] == b

    # Test iterations
    (l,s) = grid_iterator1(g)
    @assert s ≈ len * (a+b)/2

    ## Periodic equispaced grids
    len = 120
    a = -T(1.2)
    b = T(3.5)
    g = PeriodicEquispacedGrid(len, a, b)

    idx = 5
    @test g[idx] ≈ a + (idx-1) * (b-a)/len
    @test g[len] ≈ b - stepsize(g)
    @test_throws BoundsError g[len+1] == b

    (l,s) = grid_iterator1(g)
    @test s ≈ (len-1)*(a+b)/2 + a

    (l,s) = grid_iterator2(g)
    @test s ≈ (len-1)*(a+b)/2 + a

    ## Tensor product grids
    len = 120
    g1 = PeriodicEquispacedGrid(len, -one(T), one(T))
    g2 = EquispacedGrid(len, -one(T), one(T))
    g = g1 × g2
    @test length(g) == length(g1) * length(g2)
    @test size(g) == (length(g1),length(g2))

    idx1 = 5
    idx2 = 9
    x1 = g1[idx1]
    x2 = g2[idx2]
    x = g[idx1,idx2]
    @test x[1] ≈ x1
    @test x[2] ≈ x2

    (l,s) = grid_iterator1(g)
    @test s ≈ -len

    (l,s) = grid_iterator2(g)
    @test s ≈ -len

    # Test a tensor of a tensor
    g3 = g × g2
    idx1 = 5
    idx2 = 7
    idx3 = 4
    x = g3[idx1,idx2,idx3]
    x1 = g1[idx1]
    x2 = g2[idx2]
    x3 = g2[idx3]
    @test x[1] ≈ x1
    @test x[2] ≈ x2
    @test x[3] ≈ x3

    # Test a mapped grid
    m = interval_map(T(0), T(1), T(2), T(3))
    # Make a MappedGrid by hand because mapped_grid would simplify
    mg1 = MappedGrid(PeriodicEquispacedGrid(30, T(0), T(1)), m)
    test_generic_grid(mg1)
    # Does mapped_grid simplify?
    mg2 = mapped_grid(PeriodicEquispacedGrid(30, T(0), T(1)), m)
    @test typeof(mg2) <: PeriodicEquispacedGrid
    @test leftendpoint(mg2) ≈ T(2)
    @test rightendpoint(mg2) ≈ T(3)

    # Apply a second map and check whether everything simplified
    m2 = interval_map(T(2), T(3), T(4), T(5))
    mg3 = mapped_grid(mg1, m2)
    @test leftendpoint(mg3) ≈ T(4)
    @test rightendpoint(mg3) ≈ T(5)
    @test typeof(supergrid(mg3)) <: PeriodicEquispacedGrid

    # Scattered grid
    pts = map(T, rand(10))
    sg = ScatteredGrid(pts)
    test_generic_grid(sg)
end
