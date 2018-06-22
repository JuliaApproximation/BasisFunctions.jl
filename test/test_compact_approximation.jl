using BasisFunctions, WaveletsCopy, Base.Test, StaticArrays

using BasisFunctions: coefficient_index_range_of_overlapping_elements, ModCartesianRange, grid_index_range_in_element_support
function test_coefficient_index_range_of_overlapping_elements()
    L = 5
    B = BSplineTranslatesBasis(1<<L,2)
    S = ScalingBasis(cdf33, L)
    for x in [0, 0.01, 0.99]
        g = ScatteredGrid([x])
        @test sort(find(evaluation_matrix(S, g).!=0)) == sort(collect(coefficient_index_range_of_overlapping_elements(S, g[1])))
        @test sort(find(evaluation_matrix(B, g).!=0)) == sort(collect(coefficient_index_range_of_overlapping_elements(B, g[1])))
    end
    B = B⊗B
    S = S⊗S
    for x in [0, 0.01, 0.99]
        g = ScatteredGrid([x])
        g = g×g
        @test sort(find(evaluation_matrix(S, g).!=0)) == sort([sub2ind(size(S), i.I...) for i in coefficient_index_range_of_overlapping_elements(S, g[1])])
        @test sort(find(evaluation_matrix(B, g).!=0)) == sort([sub2ind(size(B), i.I...) for i in coefficient_index_range_of_overlapping_elements(B, g[1])])
    end
end

function test_spline_approximation(T)

    B = BSplineTranslatesBasis(10,3,T)⊗BSplineTranslatesBasis(15,5,T)

    x = [SVector(1e-4,1e-5),SVector(.23,.94)]
    indexranges = coefficient_index_range_of_overlapping_elements.(B,x)
    for (t,indexrange) in zip(x,indexranges)
        for i in indexrange
            @test B[i](t) > 0
        end
    end

    B = BSplineTranslatesBasis(10,3,T)⊗BSplineTranslatesBasis(15,5,T)⊗BSplineTranslatesBasis(5,1,T)

    x = [SVector(1e-4,1e-5,1e-4),SVector(.23,.94,.93)]
    indexranges = coefficient_index_range_of_overlapping_elements.(B,x)
    for (t,indexrange) in zip(x,indexranges)
        for i in indexrange
            @test B[i](t) > 0
        end
    end


    # Select the points that are in the support of the function
    B = BSplineTranslatesBasis(10,3,T)⊗BSplineTranslatesBasis(15,5,T)
    g = BasisFunctions.grid(B)
    set = map(x->CartesianIndex(ind2sub(size(B), x)), [1,length(B)-1])
    indices = grid_index_range_in_element_support.(B,g,set)

    @test reduce(&,true,[B[set[j]](x...) for j in 1:length(set) for x in [g[i] for i in indices[j]]] .> 0)

    # Select the points that are in the support of the function
    B = BSplineTranslatesBasis(10,3,T)⊗BSplineTranslatesBasis(15,5,T)⊗BSplineTranslatesBasis(5,1,T)
    g = BasisFunctions.grid(B)
    set = map(x->CartesianIndex(ind2sub(size(B), x)), [1,length(B)-1])
    indices = grid_index_range_in_element_support.(B,g,set)
    @test reduce(&,true,[B[set[j]](x...) for j in 1:length(set) for x in [g[i] for i in indices[j]]] .> 0)
end

using BasisFunctions: grid_index_mask_in_element_support, coefficient_index_mask_of_overlapping_elements, coefficient_indices_of_overlapping_elements
function test_index_masks()
    B = BSplineTranslatesBasis(10,3)⊗BSplineTranslatesBasis(15,5)
    g = grid(B)
    indices = [rand(1:length(B)) for i in 1:10]
    cindices = map(x->CartesianIndex(ind2sub(size(B), x)), indices)
    mask = grid_index_mask_in_element_support(B, g, cindices)

    for i in eachindex(g)
        evaluate_not_zero =  false
        for j in indices
            if abs(B[j](g[i])) > 1e-10
                evaluate_not_zero = true;
                break;
            end
        end
        @test evaluate_not_zero == mask[i]
    end

    m = BitArray(size(B))
    fill!(m, 0)
    for i in cindices
        m[i] = 1
    end
    mask = grid_index_mask_in_element_support(B, g, m)
    for i in eachindex(g)
        evaluate_not_zero =  false
        for j in indices
            if abs(B[j](g[i])) > 1e-10
                evaluate_not_zero = true;
                break;
            end
        end
        @test evaluate_not_zero == mask[i]
    end


    B = BSplineTranslatesBasis(10,3)⊗BSplineTranslatesBasis(15,5)

    g = ScatteredGrid([SVector(1e-4,1e-5),SVector(.23,.94)])
    indexmask = coefficient_index_mask_of_overlapping_elements(B,g)
    for i in eachindex(B)
        if indexmask[i]
            @test norm(B[i](g)) > 0
        else
            @test norm(B[i](g)) ==0
        end
    end
    B = BSplineTranslatesBasis(10,3)⊗BSplineTranslatesBasis(15,5)⊗BSplineTranslatesBasis(5,1)

    g = ScatteredGrid([SVector(1e-4,1e-5,1e-4),SVector(.23,.94,.93)])
    indexmask = coefficient_index_mask_of_overlapping_elements(B,g)
    for  i in eachindex(B)
        if indexmask[i]
            @test norm(B[i](g)) > 0
        else
            @test norm(B[i](g)) ==0
        end
    end
    indices = coefficient_indices_of_overlapping_elements(B, g)
    m = BitArray(size(B))
    fill!(m, 0)
    m[indices] = 1
    @test m==indexmask

end

function test_scaling_platform()
    platform = scaling_platform([4,5], [db3,db3], 2)
    B = primal(platform, 1)
    e = rand(B)
    @test B == ScalingBasis(db3, 4)⊗ScalingBasis(db3, 5)
    @test dual(platform, 1) == BasisFunctions.wavelet_dual(B)
    @test sampler(platform, 1)==GridSamplingOperator( BasisFunctions.oversampled_grid(B, 2))
    @test dual_sampler(platform, 1).sampler==sampler(platform, 1)
    e = rand(src(dual_sampler(platform, 1).weight))
    @test dual_sampler(platform, 1).weight*e≈BasisFunctions.WeightOperator(primal(platform, 1), [2,2], [0,0])*e
    @test BasisFunctions.Zt(platform, 1)*(sampler(platform, 1)*((x,y)->1.))≈ones(B)/sqrt(length(B))
    e = rand(src(BasisFunctions.A(platform, 1)))
    @test BasisFunctions.A(platform, 1)*e≈evaluation_operator(B, BasisFunctions.oversampled_grid(B, 2))*e

    platform = scaling_platform([4,5], [db3,cdf13], 2)
    B = primal(platform, 2)
    @test B == ScalingBasis(db3, 5)⊗ScalingBasis(cdf13, 6)
    @test dual(platform, 2) == BasisFunctions.wavelet_dual(B)
    @test sampler(platform, 2)==GridSamplingOperator( BasisFunctions.oversampled_grid(B, 2))
    @test dual_sampler(platform, 2).sampler==sampler(platform, 2)
    e = rand(src(dual_sampler(platform, 2).weight))
    @test dual_sampler(platform, 2).weight*e≈BasisFunctions.WeightOperator(primal(platform, 2), [2,2], [0,0])*e
    @test BasisFunctions.Zt(platform, 2)*(sampler(platform, 2)*((x,y)->1.))≈ones(B)/sqrt(length(B))
    e = rand(src(BasisFunctions.A(platform, 2)))
    @test BasisFunctions.A(platform, 2)*e≈evaluation_operator(B, BasisFunctions.oversampled_grid(B, 2))*e

    platform = scaling_platform([4,5], [db3,cdf13], 4)
    B = primal(platform, 2)
    @test B == ScalingBasis(db3, 5)⊗ScalingBasis(cdf13, 6)
    @test dual(platform, 2) == BasisFunctions.wavelet_dual(B)
    @test sampler(platform, 2)==GridSamplingOperator( BasisFunctions.oversampled_grid(B, 4))
    @test dual_sampler(platform, 2).sampler==sampler(platform, 2)
    e = rand(src(dual_sampler(platform, 2).weight))
    @test dual_sampler(platform, 2).weight*e≈BasisFunctions.WeightOperator(primal(platform, 2), [2,2], [1,1])*e
    @test BasisFunctions.Zt(platform, 2)*(sampler(platform, 2)*((x,y)->1.))≈ones(B)/sqrt(length(B))
    e = rand(src(BasisFunctions.A(platform, 2)))
    @test BasisFunctions.A(platform, 2)*e≈evaluation_operator(B, BasisFunctions.oversampled_grid(B, 4))*e

end

@testset "Spline util (1)" begin test_coefficient_index_range_of_overlapping_elements() end
@testset "Spline util (2)" begin test_index_masks() end
@testset "Spline approx (float64)" begin test_spline_approximation(Float64) end
@testset "Spline approx (BigFloat)" begin test_spline_approximation(BigFloat) end
@testset "Scaling platform" begin test_scaling_platform() end
