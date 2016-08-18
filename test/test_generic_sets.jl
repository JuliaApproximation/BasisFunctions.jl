# test_generic_sets.jl

function test_generic_set_interface(basis, SET = typeof(basis))
    ELT = eltype(basis)
    T = numtype(basis)
    n = length(basis)

    @test typeof(basis) <: SET

    if is_basis(basis)
        @test is_frame(basis)
    end
    if is_orthogonal(basis)
        @test is_biorthogonal(basis)
    end

    # Test type promotion
    ELT2 = widen(ELT)
    basis2 = promote_eltype(basis, ELT2)
    if typeof(basis2) <: OperatedSet
        # This test fails for an OperatedSet, because the eltype of an operator can not yet be promoted
        # @test_skip eltype(basis2) == ELT2
    else
        @test eltype(basis2) == ELT2
    end


    ## Test dimensions
    @test index_dim(basis) == index_dim(SET)
    @test index_dim(basis) == length(size(basis))
    s = size(basis)
    for i = 1:length(s)
        @test size(basis, i) == s[i]
    end
    @test index_dim(basis) <= ndims(basis)

    ## Test iteration over the set
    l = 0
    equality = true
    for i in eachindex(basis)
        l += 1

        # Is conversion from logical to native index and back a bijection?
        idxn = natural_index(basis, i)
        idx = logical_index(basis, idxn)
        equality = equality & (idx == i)
    end
    @test l == n
    @test equality

    ## Does indexing work as intended?
    idx = random_index(basis)
    bf = basis[idx]
    @test set(bf) == basis

    x = fixed_point_in_domain(basis)
    @test bf(x) ≈ call_set(basis, idx, x)

    # Create a random expansion in the basis to test expansion interface
    e = random_expansion(basis)
    coef = coefficients(e)

    ## Does evaluating an expansion equal the sum of coefficients times basis function calls?
    x = fixed_point_in_domain(basis)
    @test e(x) ≈ sum([coef[i] * basis[i](x) for i in eachindex(coef)])

    ## Test evaluation on an array
    ARRAY_TYPE = typeof(fixed_point_in_domain(basis))
    x_array = ARRAY_TYPE[random_point_in_domain(basis) for i in 1:10]
    z = e(x_array)
    @test  z ≈ ELT[ e(x_array[i]) for i in eachindex(x_array) ]


    ## Verify evaluation on the associated grid
    if BF.has_grid(basis)

        grid1 = grid(basis)
        @test length(grid1) == length(basis)

        z1 = e(grid1)
        @test z1 ≈ ELT[ e(grid1[i]) for i in eachindex(grid1) ]
        E = evaluation_operator(set(e), DiscreteGridSpace(set(e)) )
        z2 = E * coefficients(e)
        @test z1 ≈ z2
    end

    ## Test output type of calling function

    types_correct = true
    for x in [ fixed_point_in_domain(basis) rationalize(point_in_domain(basis, 0.5)) ]
        for idx in [1 2 n>>1 n-1 n]
            z = call_set(basis, idx, x)
            types_correct = types_correct & (typeof(z) == ELT)
        end
    end
    @test types_correct

    if ndims(basis) == 1

        # Test evaluation on a different grid on the support of the basis
        a = left(basis)
        b = right(basis)

        if isinf(a)
            a = -T(1)
        end
        if isinf(b)
            b = T(1)
        end

        grid2 = EquispacedGrid(n+3, T(a), T(b))
        z = e(grid2)
        @test z ≈ ELT[ e(grid2[i]) for i in eachindex(grid2) ]

        # Test evaluation on a grid of a single basis function
        idx = random_index(basis)
        z = basis[idx](grid2)
        @test  z ≈ ELT[ basis[idx](grid2[i]) for i in eachindex(grid2) ]

    end

    ## Test extensions
    if BF.has_extension(basis)
        n2 = extension_size(basis)
        basis2 = resize(basis, n2)
        E = extension_operator(basis, basis2)
        e1 = random_expansion(basis)
        e2 = E * e1
        x1 = point_in_domain(basis, 1/2)
        @test e1(x1) ≈ e2(x1)
        x2 = point_in_domain(basis, 0.3)
        @test e1(x2) ≈ e2(x2)

        R = restriction_operator(basis2, basis)
        e3 = R * e2
        @test e2(x1) ≈ e3(x1)
        @test e2(x2) ≈ e3(x2)
    end

    ## Test derivatives
    if BF.has_derivative(basis)
        D = differentiation_operator(basis)
        @test basis == src(D)
        diff_dest = dest(D)

        coef1 = random_expansion(basis)
        coef2 = D*coef
        e1 = SetExpansion(basis, coef)
        e2 = SetExpansion(diff_dest, coef2)

        x = fixed_point_in_domain(basis)
        delta = sqrt(eps(T))/10
        @test abs( (e1(x+delta)-e1(x))/delta - e2(x) ) / abs(e2(x)) < 2000delta
    end

    ## Test antiderivatives
    if BF.has_antiderivative(basis)
        D = antidifferentiation_operator(basis)
        @test basis == src(D)
        diff_dest = dest(D)

        coef1 = random_expansion(basis)
        coef2 = D*coef
        e1 = SetExpansion(basis, coef)
        e2 = SetExpansion(diff_dest, coef2)

        x = fixed_point_in_domain(basis)
        delta = sqrt(eps(T))/10

        @test abs( (e2(x+delta)-e2(x))/delta - e1(x) ) / abs(e1(x)) < 2000delta
    end

    ## Test associated transform
    if BF.has_transform(basis)
        # Check whether it is unitary
        g = grid(basis)
        t = transform_operator(g, basis)
        it = transform_operator(basis, g)
        A = matrix(t)
        if T == Float64
            @test cond(A) ≈ 1
        else
            #@test_skip cond(A) ≈ 1
        end
        AI = matrix(it)
        if T == Float64
            @test cond(AI) ≈ 1
        else
            #@test_skip cond(AI) ≈ 1
        end

        # Verify the normalization operator and its inverse
        n = transform_normalization_operator(basis)
        # - try interpolation using transform+normalization
        x = zeros(ELT, size(basis))
        for i in eachindex(x)
            x[i] = rand()
        end
        e = SetExpansion(basis, n*t*x)
        @test maximum(abs(e(g)-x)) < sqrt(eps(T))
        # - verify the inverse of the normalization
        ni = inv(n)
        @test maximum(abs(x - ni*n*x)) < sqrt(eps(T))
        @test maximum(abs(x - n*ni*x)) < sqrt(eps(T))
    end

    ## Test interpolation operator on a suitable interpolation grid
    if is_basis(basis) == True()
        g = suitable_interpolation_grid(basis)
        I = interpolation_operator(basis, g)
        x = zeros(ELT, size(basis))
        for i in eachindex(x)
            x[i] = rand()
        end
        e = SetExpansion(basis, I*x)
        @test maximum(abs(e(g)-x)) < 100sqrt(eps(T))
    end

    ## Test evaluation operator
    g = suitable_interpolation_grid(basis)
    E = evaluation_operator(basis, g)
    e = random_expansion(basis)
    y = E*e
    @test maximum([abs(e(g[i])-y[i]) for i in eachindex(g)]) < sqrt(eps(T))
end


#####
# Tensor sets
#####

function test_tensor_sets(T)

    a = FourierBasis(12)
    b = FourierBasis(13)
    c = FourierBasis(11)
    d = TensorProductSet(a,b,c)

    bf = d[3,4,5]
    x1 = T(2//10)
    x2 = T(3//10)
    x3 = T(4//10)
    @test bf(x1, x2, x3) ≈ call_set(a, 3, x1) * call_set(b, 4, x2) * call_set(c, 5, x3)

    # Can you iterate over the product set?
    z = zero(T)
    i = 0
    for f in d
        z += f(x1, x2, x3)
        i = i+1
    end
    @test i == length(d)
    @test abs(-0.211 - z) < 0.01

    z = zero(T)
    l = 0
    for i in eachindex(d)
        f = d[i]
        z += f(x1, x2, x3)
        l = l+1
    end
    @test l == length(d)
    @test abs(-0.211 - z) < 0.01
end
