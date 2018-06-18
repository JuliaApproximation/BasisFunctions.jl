using Base.Test
using BasisFunctions
using StaticArrays
using Domains
function test_generic_periodicbsplinebasis(T)

    for B in (BSplineTranslatesBasis, SymBSplineTranslatesBasis,)
        tol = sqrt(eps(real(T)))
        n = 5
        b = B(n,3, T)
        @test leftendpoint(support(b)) == 0
        @test rightendpoint(support(b))==1

        @test length(b)==5
        @test BasisFunctions.degree(b)==3
        @test is_basis(b)
        @test BasisFunctions.is_biorthogonal(b)
        @test !BasisFunctions.is_orthogonal(b)
        @test !BasisFunctions.is_orthonormal(b)
        @test !has_unitary_transform(b)

        @test BasisFunctions.left_of_compact_function(b) <= 0 <= BasisFunctions.right_of_compact_function(b)
        @test 0 < BasisFunctions.right_of_compact_function(b) - BasisFunctions.left_of_compact_function(b) < BasisFunctions.period(b)

        @test instantiate(B, 4, Float16)==B(4,3,Float16)
        @test promote_domaintype(b, Float16)==B(n,BasisFunctions.degree(b),Float16)
        @test promote_domaintype(b, complex(Float64))==B(n,BasisFunctions.degree(b),Complex128)
        @test resize(b, 20)==B(20,BasisFunctions.degree(b),T)
        @test BasisFunctions.grid(b)==PeriodicEquispacedGrid(n,0,1)
        @test BasisFunctions.period(b)==T(1)
        @test BasisFunctions.stepsize(b)==T(1//5)

        n = 3
        b=B(n,1,T)
        @test abs(sum(BasisFunctions.grammatrix(b) - [2//3 1//6 1//6; 1//6 2//3 1//6;1//6 1//6 2//3]//n)) < tol
        @test abs(sum(BasisFunctions.dualgrammatrix(b) - [5/3 -1/3 -1/3; -1/3 5/3 -1/3; -1/3 -1/3 5/3]*n)) < tol

        n = 8
        b=B(n,0,T)
        t = linspace(T(0),T(1))
        fp = map(x->BasisFunctions.unsafe_eval_element(b,1,x),t)
        fd = 1/n*map(x->BasisFunctions.eval_dualelement(b,1,x),t)
        @test fp≈fd
    end
end

using BasisFunctions: interval_index, coefficient_index_range_of_overlapping_elements
function test_translatedbsplines(T)
    B = BSplineTranslatesBasis(10,1)
    x = [1e-4,.23,.94]
    @test [interval_index(B,t)[1] for t in x] == [1,3,10]
    indices = coefficient_index_range_of_overlapping_elements.(B,x)
    @test reduce(&,true,[B[i].(x[j]) for j in 1:length(x) for i in indices[j]].>0)

    tol = sqrt(eps(real(T)))
    n = 5
    bb = BSplineTranslatesBasis(n, 1, T; scaled=true)
    b = BSplineTranslatesBasis(n, 1, T)
    e = rand(n)
    @test norm(Gram(b)*e-Gram(bb)*e/n) < tol

    b = BSplineTranslatesBasis(n,3, T)

    @test BasisFunctions.name(b) == "Set of translates of a function (B spline of degree 3)"

    @test infimum(support(b,1))≈ 0
    @test infimum(support(b,2))≈1//5
    @test infimum(support(b,3))≈0
    @test infimum(support(b,4))≈0
    @test infimum(support(b,5))≈0
    @test supremum(support(b,1))≈4//5
    @test supremum(support(b,2))≈1
    @test supremum(support(b,3))≈1
    @test supremum(support(b,4))≈1
    @test supremum(support(b,5))≈1

    t = .001
    @test in_support(b,1,.5)
    @test !in_support(b,1,.8+t)
    @test !in_support(b,1,1.-t)
    @test in_support(b,3,.2-t)
    @test in_support(b,3,.4+t)
    @test !in_support(b,3,.2+t)
    @test !in_support(b,3,.4-t)

    @test BasisFunctions.compatible_grid(b, grid(b))
    @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0,1))
    @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n+1,0,1))
    @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0.1,1))
    @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0,1.1))

    grid(BSplineTranslatesBasis(n,2, T)) == MidpointEquispacedGrid(n,0,1)
    @test BasisFunctions.degree(BSplineTranslatesBasis(5,2, T)) == 2
    b = BSplineTranslatesBasis(n,2,T)
    @test BasisFunctions.compatible_grid(b, grid(b))
    @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0,1))
    @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n+1,0,1))
    @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0.1,1))
    @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0,1.1))

    # Test extension_operator and invertability of restriction_operator w.r.t. extension_operator.
    n = 8
    for degree in 0:3
        b = BSplineTranslatesBasis(n, degree, T)
        basis_ext = extend(b)
        r = restriction_operator(basis_ext, b)
        e = extension_operator(b, basis_ext)
        @test abs(sum(eye(n)-matrix(r*e))) < tol

        grid_ext = grid(basis_ext)
        L = evaluation_operator(b, grid_ext)
        e = random_expansion(b)
        z = L*e
        L2 = evaluation_operator(basis_ext, grid_ext) * extension_operator(b, basis_ext)
        z2 = L2*e
        @test maximum(abs.(z-z2)) < tol
    end


    if T == Float64
        for K in 0:3
            for s2 in 5:6
                s1 = s2<<1
                b1 = BSplineTranslatesBasis(s1,K,T)
                b2 = BSplineTranslatesBasis(s2,K,T)

                e1 = random_expansion(b1)
                e2 = random_expansion(b2)

                @test coefficients(BasisFunctions.default_evaluation_operator(b2, gridspace(b2))*e2) ≈ coefficients(grid_evaluation_operator(b2, gridspace(b2), grid(b2))*e2)
                @test coefficients(BasisFunctions.default_evaluation_operator(b2, gridspace(b1))*e2) ≈ coefficients(grid_evaluation_operator(b2, gridspace(b1), grid(b1))*e2)
                @test coefficients(BasisFunctions.default_evaluation_operator(b1, gridspace(b1))*e1) ≈ coefficients(grid_evaluation_operator(b1, gridspace(b1), grid(b1))*e1)
                @test coefficients(BasisFunctions.default_evaluation_operator(b1, gridspace(b2))*e1) ≈ coefficients(grid_evaluation_operator(b1, gridspace(b2), grid(b2))*e1)

                mr = matrix(restriction_operator(b1, b2))
                me = matrix(extension_operator(b2, b1))
                pinvme = pinv(me)
                r = rand(size(pinvme,2))
                @test pinvme*r ≈ mr*r
            end
        end

        for K in 0:3
            for s2 in 5:6
                s1 = s2<<1
                b1 = BSplineTranslatesBasis(s1,K,T; scaled=true)
                b2 = BSplineTranslatesBasis(s2,K,T; scaled=true)

                e1 = random_expansion(b1)
                e2 = random_expansion(b2)

                @test coefficients(BasisFunctions.default_evaluation_operator(b2, gridspace(b2))*e2) ≈ coefficients(grid_evaluation_operator(b2, gridspace(b2), grid(b2))*e2)
                @test coefficients(BasisFunctions.default_evaluation_operator(b2, gridspace(b1))*e2) ≈ coefficients(grid_evaluation_operator(b2, gridspace(b1), grid(b1))*e2)
                @test coefficients(BasisFunctions.default_evaluation_operator(b1, gridspace(b1))*e1) ≈ coefficients(grid_evaluation_operator(b1, gridspace(b1), grid(b1))*e1)
                @test coefficients(BasisFunctions.default_evaluation_operator(b1, gridspace(b2))*e1) ≈ coefficients(grid_evaluation_operator(b1, gridspace(b2), grid(b2))*e1)

                mr = matrix(restriction_operator(b1, b2))
                me = matrix(extension_operator(b2, b1))
                pinvme = pinv(me)
                r = rand(size(pinvme,2))
                @test pinvme*r ≈ mr*r
            end
        end
    end

    @test_throws AssertionError restriction_operator(BSplineTranslatesBasis(4,0,T), BSplineTranslatesBasis(3,0,T))
    @test_throws AssertionError extension_operator(BSplineTranslatesBasis(4,0,T), BSplineTranslatesBasis(6,0,T))
end


function test_translatedsymmetricbsplines(T)
    tol = sqrt(eps(real(T)))
    n = 5

    b = SymBSplineTranslatesBasis(n,1,T)
    bb = BSplineTranslatesBasis(n,1,T)
    @test norm((Gram(b)-Gram(bb))*rand(n)) < tol

    b = SymBSplineTranslatesBasis(n,3, T)

    @test BasisFunctions.name(b) == "Set of translates of a function (symmetric B spline of degree 3)"


    @test infimum(support(b,1))≈ -4//10
    @test infimum(support(b,2))≈ -2//10
    @test infimum(support(b,3))≈ 0//10
    @test infimum(support(b,4))≈ 2//10
    @test infimum(support(b,5))≈ 4//10
    @test supremum(support(b,1))≈4//10
    @test supremum(support(b,2))≈6//10
    @test supremum(support(b,3))≈8//10
    @test supremum(support(b,4))≈10//10
    @test supremum(support(b,5))≈12//10

    t = .001
    @test in_support(b,1,.0)
    @test !in_support(b,1,.4+t)
    @test !in_support(b,1,.6-t)
    @test in_support(b,3,.8-t)
    @test in_support(b,3,.0+t)
    @test !in_support(b,3,.8+t)
    @test !in_support(b,3,.0-t)

    grid(SymBSplineTranslatesBasis(n,2, T)) == EquispacedGrid(n,0,1)
    @test BasisFunctions.compatible_grid(b, grid(b))
    @test BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0,1))
    @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n+1,0,1))
    @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0.1,1))
    @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0,1.1))

    @test BasisFunctions.compatible_grid(b, grid(b))
    @test BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0,1))
    @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n+1,0,1))
    @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0.1,1))
    @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0,1.1))

    # Test extension_operator and invertability of restriction_operator w.r.t. extension_operator.
    n = 8
    for degree in 1:2:3
        b = SymBSplineTranslatesBasis(n, degree, T)
        basis_ext = extend(b)
        r = restriction_operator(basis_ext, b)
        e = extension_operator(b, basis_ext)
        @test abs(sum(eye(n)-matrix(r*e))) < tol

        grid_ext = grid(basis_ext)
        L = evaluation_operator(b, grid_ext)
        e = random_expansion(b)
        z = L*e
        L2 = evaluation_operator(basis_ext, grid_ext) * extension_operator(b, basis_ext)
        z2 = L2*e
        @test maximum(abs.(z-z2)) < tol
    end


    for K in 1:2:3
        for s2 in 5:6
            s1 = s2<<1
            b1 = SymBSplineTranslatesBasis(s1,K)
            b2 = SymBSplineTranslatesBasis(s2,K)

            e1 = random_expansion(b1)
            e2 = random_expansion(b2)

            @test coefficients(BasisFunctions.default_evaluation_operator(b2, gridspace(b2))*e2) ≈ coefficients(grid_evaluation_operator(b2, gridspace(b2), grid(b2))*e2)
            @test coefficients(BasisFunctions.default_evaluation_operator(b2, gridspace(b1))*e2) ≈ coefficients(grid_evaluation_operator(b2, gridspace(b1), grid(b1))*e2)
            @test coefficients(BasisFunctions.default_evaluation_operator(b1, gridspace(b1))*e1) ≈ coefficients(grid_evaluation_operator(b1, gridspace(b1), grid(b1))*e1)
            @test coefficients(BasisFunctions.default_evaluation_operator(b1, gridspace(b2))*e1) ≈ coefficients(grid_evaluation_operator(b1, gridspace(b2), grid(b2))*e1)

            mr = matrix(restriction_operator(b1, b2))
            me = matrix(extension_operator(b2, b1))
            pinvme = pinv(me)
            r = rand(size(pinvme,2))
            @test pinvme*r ≈ mr*r
        end
    end

    @test_throws AssertionError restriction_operator(SymBSplineTranslatesBasis(4,1), SymBSplineTranslatesBasis(3,1))
    @test_throws AssertionError extension_operator(SymBSplineTranslatesBasis(4,1), SymBSplineTranslatesBasis(6,1))
end

# function test_orthonormalsplinebasis(T)
#     b = OrthonormalSplineBasis(5,2,Float64)
#     b = OrthonormalSplineBasis(5,2,T)
#     @test name(b) == "Set of translates of a function (B spline of degree 2) (orthonormalized)"
#     @test instantiate(OrthonormalSplineBasis,5)==OrthonormalSplineBasis(5,3)
#
#     G = sqrt(DualGram(Span(b.superdict)))
#     e = zeros(eltype(G),size(G,1))
#     e[1] = 1
#     @test BasisFunctions.coefficients(b) ≈ G*e
#
#     d = BasisFunctions.primalgramcolumn(Span(b); abstol=1e-3)
#     @test d ≈ e
#     @test typeof(Gram(Span(b))) <: IdentityOperator
#
#     n = 8
#     for degree in 0:3
#         b = Span(OrthonormalSplineBasis(n, degree, T))
#         basis_ext = extend(b)
#         r = restriction_operator(basis_ext, b)
#         e = extension_operator(b, basis_ext)
#         @test eye(n) ≈ matrix(r*e)
#
#         grid_ext = grid(basis_ext)
#         L = evaluation_operator(b, grid_ext)
#         e = random_expansion(b)
#         z = L*e
#         L2 = evaluation_operator(basis_ext, grid_ext) * extension_operator(b, basis_ext)
#         z2 = L2*e
#         @test 1+maximum(abs.(z-z2)) ≈ T(1)
#     end
# end

# function test_discrete_orthonormalsplinebasis(T)
#     b = DiscreteOrthonormalSplineBasis(5,2,Float64)
#     b = DiscreteOrthonormalSplineBasis(5,2,T)
#     @test name(b) == "Set of translates of a function (B spline of degree 2) (orthonormalized, discrete)"
#     @test instantiate(DiscreteOrthonormalSplineBasis,5)==DiscreteOrthonormalSplineBasis(5,3)
#
#     E = evaluation_operator(Span(b))
#     e = zeros(eltype(E),size(E,1))
#     e[1] = 1
#     @test (E'*E*e) ≈ 5*e
#     @test typeof(E) <: BasisFunctions.MultiplicationOperator
#
#     n = 8
#     for degree in 0:3
#         b = Span(DiscreteOrthonormalSplineBasis(n, degree, T))
#         basis_ext = extend(b)
#         r = restriction_operator(basis_ext, b)
#         e = extension_operator(b, basis_ext)
#         @test eye(n) ≈ matrix(r*e)
#
#         grid_ext = grid(basis_ext)
#         L = evaluation_operator(b, grid_ext)
#         e = random_expansion(b)
#         z = L*e
#         L2 = evaluation_operator(basis_ext, grid_ext) * extension_operator(b, basis_ext)
#         z2 = L2*e
#         @test 1+maximum(abs.(z-z2)) ≈ T(1)
#     end
#     for dgr in 0:4, oversampling in 1:4, n in 10:11
#         b = DiscreteOrthonormalSplineBasis(n,dgr,T; oversampling=oversampling)
#
#         e = zeros(T,n); e[1] = 1
#         @test sqrt(DiscreteDualGram(Span(BSplineTranslatesBasis(n,dgr,T)); oversampling=oversampling))*e≈BasisFunctions.coefficients(b)
#         @test DiscreteGram(Span(b); oversampling=oversampling)*e≈e
#     end
# end


using QuadGK

function test_dualsplinebasis(T)
    n = 10; degree = 2;
    tol = max(sqrt(eps(real(T))), 1e-16)
    b = BSplineTranslatesBasis(n,degree)
    bb = BasisFunctions.dual(b; reltol=tol, abstol=tol)
    @test dual(bb) == b
    e = coefficients(random_expansion(b))

    @test Gram(b; abstol=tol, reltol=tol)*e ≈ DualGram(bb; abstol=tol, reltol=tol)*e
    @test Gram(bb; abstol=tol, reltol=tol)*e ≈ DualGram(b; abstol=tol, reltol=tol)*e
    @test BasisFunctions.dualgramcolumn(b; reltol=tol, abstol=tol) ≈ BasisFunctions.coefficients(bb)
    @test QuadGK.quadgk(x->b[1](x)*bb[1](x),infimum(support(b)), supremum(support(b)); reltol=tol, abstol=tol)[1] - T(1) < sqrt(tol)
    @test QuadGK.quadgk(x->b[1](x)*bb[2](x),infimum(support(b)), supremum(support(b)); reltol=tol, abstol=tol)[1] - T(1) < sqrt(tol)
end

function test_discrete_dualsplinebasis(T)
    for degree in 0:4, n in 10:11, oversampling=1:3
        b = BSplineTranslatesBasis(n,degree, T)
        d = discrete_dual(b; oversampling=oversampling)

        e = zeros(T,n); e[1] = 1
        @test DiscreteDualGram(b; oversampling=oversampling)*e≈BasisFunctions.coefficients(d)
        E1 = matrix(discrete_dual_evaluation_operator(b; oversampling=oversampling))
        oss = []
        # For even degrees points on a coarse grid do not overlap with those on e fine grid.
        if isodd(degree)
            oss = [1, oversampling]
        else
            oss = [oversampling,]
        end
        for os in oss
            E2 = matrix(evaluation_operator(d; oversampling = os))
            E2test = E1[1:Int(oversampling/os):end,:]
            @test E2test*e ≈ E2*e
        end
    end
end

function test_bspline_platform(T)
    #  1D
    init = 4
    for oversampling in [1,2,4],  degree in 2:3, i in [1,3]
        platform = BasisFunctions.bspline_platform(T, init, degree, oversampling)

        P = primal(platform,i)
        D = dual(platform,i)
        S = BasisFunctions.sampler(platform,i)

        B = P
        g = BasisFunctions.oversampled_grid(B, oversampling)
        E = CirculantOperator(evaluation_matrix(B[1],g)[:])*IndexExtensionOperator(B,gridspace(g),1:oversampling:length(g))
        G = CirculantOperator(E'E*[1,zeros(length(g)-1)...]/length(B))
        DG = BasisFunctions.wrap_operator(B, B, inv(G))

        e = map(T,rand(length(B)))
        @test evaluation_operator(D,g)*e≈evaluation_matrix(D,g)*e
        @test evaluation_operator(D,g)*e≈evaluation_operator(P,g)*(matrix(DG)*e)
        @test evaluation_operator(D,g)'*evaluation_operator(P,g)*e ≈length(B)e
        @test S*exp≈exp.(g)
    end

    #  ND
    init = [3,4]
    degree = [1,3]
    T = Float64
    oversampling = 2
    for oversampling in [1,2,4], i in [1,2]
        platform = BasisFunctions.bspline_platform(T, init, degree, oversampling)

        P = primal(platform,i)
        D = dual(platform,i)
        S = BasisFunctions.sampler(platform,i)


        B = P
        B1, B2 = elements(P)
        g1 = BasisFunctions.oversampled_grid(B1,oversampling)
        g2 = BasisFunctions.oversampled_grid(B2,oversampling)
        g = g1×g2

        E1 = CirculantOperator(evaluation_matrix(B1[1],g1)[:])*IndexExtensionOperator(B1,gridspace(g1),1:oversampling:length(g1))
        E2 = CirculantOperator(evaluation_matrix(B2[1],g2)[:])*IndexExtensionOperator(B2,gridspace(g2),1:oversampling:length(g2))

        G1 = CirculantOperator(E1'E1*[1,zeros(length(g1)-1)...]/length(B1));
        G2 = CirculantOperator(E2'E2*[1,zeros(length(g2)-1)...]/length(B2));
        G = G1⊗G2

        DG = BasisFunctions.wrap_operator(B, B, inv(G))

        e = map(T,rand(size(B)...))
        @test (evaluation_operator(D,g)*e)[:]≈evaluation_matrix(D,g)*e[:]
        @test (evaluation_operator(D,g)*e)[:]≈(evaluation_operator(P,g)*reshape(matrix(DG)*e[:],size(P)...))[:]
        @test evaluation_operator(D,g)'*evaluation_operator(P,g)*e ≈length(B)e
        f = (x,y)->exp(x*y)
        @test S*f≈f.(g)
    end

    init = [3,3]
    degree = [1,3]
    oversampling = 2
    center = @SVector [.5,.5]
    domain  = disk(.3,center)
    platform=BasisFunctions.bspline_platform(T, init, degree, oversampling)
    i = 2
    B = primal(platform,i)
    D = dual(platform,i)
    g = grid(sampler(platform,i))
    Aop = BasisFunctions.A(platform, i)
    Zop = BasisFunctions.Zt(platform, i)'


    e = map(T,rand(size(B)...))
    @test evaluation_operator(B, g)*e≈Aop*e
    @test evaluation_operator(D, g)*e≈Zop*e*length(D)
    @test BasisFunctions.Zt(platform, i)*Aop*e ≈ e
end

function test_sparsity_speed(T)
    for d in 0:4
        B = BSplineTranslatesBasis(1<<10, d, T)
        E1 = evaluation_operator(B; sparse = false)
        E2 = evaluation_operator(B; sparse = true)
        typeof(E1)
        @test typeof(E1) <: CirculantOperator
        @test typeof(E2) <: MultiplicationOperator
        x = zeros(src(E1))
        b = zeros(dest(E1))
        # compilation
        tic()
        for i in 1:100
            apply!(E1, x, b)
        end
        t1=toq()
        tic()
        for i in 1:100
            apply!(E2, x, b)
        end
        t2=toq()

        tic()
        for i in 1:100
            apply!(E1, x, b)
        end
        t1=toq()
        tic()
        for i in 1:100
            apply!(E2, x, b)
        end
        t2=toq()
        #check whether sparsity is still the good default
        @test t2 < 10*t1
    end
end


# exit()

# @testset begin test_discrete_dualsplinebasis(BigFloat) end
# @testset begin test_dualsplinebasis(BigFloat) end
#
# @testset begin test_translatedbsplines(BigFloat) end
# @testset begin test_translatedsymmetricbsplines(BigFloat) end
# @testset begin test_generic_periodicbsplinebasis(BigFloat) end
# @testset begin test_bspline_platform(BigFloat) end
# @testset begin test_sparsity_speed(BigFloat) end



# @testset begin test_discrete_orthonormalsplinebasis(BigFloat) end
# @testset begin test_orthonormalsplinebasis(BigFloat) end


# using Plots
# using BasisFunctions
# n = 7
# k = 0
# b = BSplineTranslatesBasis(n,k)
# t = linspace(-0,2,200)
# f = map(x->unsafe_eval_element(b,1,x), t)
# using Plots
# plot!(t,f)
# @which unsafe_eval_element(b,1,.25)
# BasisFunctions.fun(b)
# f = map(BasisFunctions.fun(b), t)
# plot!(t,f)
# f = map(x->BasisFunctions.eval_expansion(b,ones(n),x),t)
# # f = map(x->BasisFunctions.eval_expansion(b,[1,0,0,0,0,0,0,0,0,0],x),t)
# plot!(t,f,ylims=[-n,2n])
#
