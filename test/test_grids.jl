using BasisFunctions, BasisFunctions.Test, DomainSets

interval_grids = (EquispacedGrid, PeriodicEquispacedGrid, MidpointEquispacedGrid, ChebyshevNodes, ChebyshevExtremae)

using Test, LinearAlgebra

types = (Float64,BigFloat)

function test_grids(T)
    ## Equispaced grids
    len = 21
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
    len = 11
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
    pts = rand(T, 10)
    sg = ScatteredGrid(pts)
    test_generic_grid(sg)
end

for T in types
    delimit(string(T))
    for GRID in interval_grids
        @testset "$(rpad(string(GRID),80))" begin
            g = instantiate(GRID,10,T)
            test_interval_grid(g)
        end
    end

    @testset "$(rpad("Specific grid tests",80))" begin
        test_grids(T)
    end
end

function test_subgrids()
    delimit("Grid functionality")
    n = 20
    grid1 = EquispacedGrid(n, -1.0, 1.0)
    subgrid2 = IndexSubGrid(grid1, 4:12)

    G1 = EquispacedGrid(n, -1.0, 1.0)
    G2 = EquispacedGrid(n, -1.0, 1.0)
    ProductG = G1 × G2

    C = disk(1.0)
    @testset begin


        G1s = IndexSubGrid(G1,2:4)
        G2s = IndexSubGrid(G2,3:5)
        ProductGs = G1s × G2s
        @test G1s[1] == G1[2]
        @test G2s[1] == G2[3]
        @test ProductGs[1,1] == [G1[2],G2[3]]
    end

    # Generic tests for the subgrids
    @testset begin
        grid,subgrid = (grid1,subgrid2)
        cnt = 0
        for i in 1:length(grid)
            if issubindex(i, subgrid)
                cnt += 1
            end
        end
        @test cnt == length(subgrid)

        space = GridBasis(grid)
        subspace = GridBasis(subgrid)
        R = restriction_operator(space, subspace)
        E = extension_operator(subspace, space)

        e = random_expansion(subspace)
        e_ext = E * e
        # Are the elements in the right place?
        cnt = 0
        diff = 0.0
        for i in 1:length(grid)
            if issubindex(i, subgrid)
                cnt += 1
                diff += abs(e[cnt] - e_ext[i])
            else
                # all other entries should be zero
                diff += abs(e_ext[i])
            end
        end
        @test diff < 1e-6

        e_rest = R * e_ext
        @test sum([abs(e[i]-e_rest[i]) for i in 1:length(e)]) < 1e-6
    end
end


test_subgrids()
