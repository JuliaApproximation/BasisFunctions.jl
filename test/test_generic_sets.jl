# test_generic_sets.jl

# We try to test approximation for all function sets, except those that
# are currently known to fail for lack of an implementation.
supports_approximation(s::FunctionSet) = true
# Laguerre and Hermite fail because they have unbounded support.
supports_approximation(s::LaguerreBasis) = false
supports_approximation(s::HermiteBasis) = false

# It is difficult to do approximation in subsets generically
supports_approximation(s::FunctionSubSet) = false
supports_approximation(s::OperatedSet) = false

supports_approximation(s::TensorProductSet) =
    reduce(&, map(supports_approximation, elements(s)))

# Pick a simple function to approximate
suitable_function(s::FunctionSet1d) = exp

# Make a simple periodic function for Fourier
suitable_function(s::FourierBasis) =  x->1/(10+cos(2*pi*x))
suitable_function(s::PeriodicSplineBasis) =  x->1/(10+cos(2*pi*x))
suitable_function(s::CosineSeries) =  x->1/(10+cos(2*pi*x))

# Make a tensor product of suitable functions
function suitable_function(s::TensorProductSet)
    if ndims(s) == 2
        f1 = suitable_function(element(s,1))
        f2 = suitable_function(element(s,2))
        (x,y) -> f1(x)*f2(y)
    elseif ndims(s) == 3
        f1 = suitable_function(element(s,1))
        f2 = suitable_function(element(s,2))
        f3 = suitable_function(element(s,3))
        (x,y,z) -> f1(x)*f2(y)*f3(z)
    end
    # We should never get here
end

# Make a suitable function by undoing the map
function suitable_function(s::MappedSet)
    f = suitable_function(set(s))
    m = inv(mapping(s))
    x -> f(m*x)
end

function suitable_function(s::AugmentedSet)
    f = suitable_function(set(s))
    g = fun(s)
    x -> g(x) * f(x)
end



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
    s = size(basis)
    for i = 1:length(s)
        @test size(basis, i) == s[i]
    end

    ## Test iteration over the set
    equality = true
    l = 0
    for i in eachindex(basis)
        l += 1
        # Is conversion from logical to native index and back a bijection?
        idxn = native_index(basis, l)
        idx = linear_index(basis, idxn)
        equality = equality & (idx == l)
    end
    @test l == n
    @test equality

    ## Does indexing work as intended?
    idx = random_index(basis)
    bf = basis[idx]
    # Test below disabled because the assumption that the set of a basis function
    # is always the set that was indexed is false, e.g., for multisets.
    #@test set(bf) == basis

    @test endof(basis) == length(basis)
    # Is a boundserror thrown when the index is too large?
    @test try
        basis[length(basis)+1]
        false
    catch
        true
    end

    # Error thrown for a range that is out of range?
    @test try
        basis[1:length(basis)+1]
        false
    catch
        true
    end

    x = fixed_point_in_domain(basis)
    @test bf(x) ≈ eval_set_element(basis, idx, x)

    x_outside = point_outside_domain(basis)
    @test bf(x_outside) == 0
    @test isnan(eval_set_element(basis, idx, x_outside, T(NaN)))


    # Create a random expansion in the basis to test expansion interface
    e = random_expansion(basis)
    coef = coefficients(e)

    ## Does evaluating an expansion equal the sum of coefficients times basis function calls?
    x = fixed_point_in_domain(basis)
    @test e(x) ≈ sum([coef[i] * basis[i](x) for i in eachindex(coef)])

    ## Test evaluation on an array
    ARRAY_TYPE = typeof(fixed_point_in_domain(basis))
    x_array = ARRAY_TYPE[random_point_in_domain(basis) for i in 1:10]
    z = map(e, x_array)
    @test  z ≈ ELT[ e(x_array[i]) for i in eachindex(x_array) ]

    # Test linearization of coefficients
    linear_coefs = zeros(eltype(basis), length(basis))
    BasisFunctions.linearize_coefficients!(basis, linear_coefs, coef)
    coef2 = BasisFunctions.delinearize_coefficients(basis, linear_coefs)
    @test coef ≈ coef2

    ## Verify evaluation on the associated grid
    if BF.has_grid(basis)

        grid1 = grid(basis)
        @test length(grid1) == length(basis)

        z1 = e(grid1)
        z2 = [ e(grid1[i]) for i in eachindex(grid1) ]
        @test z1 ≈ z2
        E = evaluation_operator(set(e), DiscreteGridSpace(set(e)) )
        z3 = E * coefficients(e)
        @test z1 ≈ z3
    end

    ## Test output type of calling function

    types_correct = true
    # The comma in the line below is important, otherwise the two static vectors
    # are combined into a statix matrix.
    for x in [ fixed_point_in_domain(basis), rationalize(point_in_domain(basis, 0.5)) ]
        for idx in [1 2 n>>1 n-1 n]
            z = eval_set_element(basis, idx, x)
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
        for dim in 1:ndims(basis)
            D = differentiation_operator(basis; dim=dim)
            @test basis == src(D)
            diff_dest = dest(D)

            coef1 = random_expansion(basis)
            coef2 = D*coef
            e1 = SetExpansion(basis, coef)
            e2 = SetExpansion(diff_dest, coef2)

            x = fixed_point_in_domain(basis)
            delta = sqrt(eps(T))/10
            N = ndims(basis)
            if N > 1
                unit_vector = zeros(T, ndims(basis))
                unit_vector[dim] = 1
                x2 = x + SVector{N}(delta*unit_vector)
            else
                x2 = x+delta
            end
            @test abs( (e1(x2)-e1(x))/delta - e2(x) ) / abs(e2(x)) < 2000*sqrt(eps(T))
        end
    end

    ## Test antiderivatives
    if BF.has_antiderivative(basis)
        for dim in 1:ndims(basis)
            D = antidifferentiation_operator(basis; dim=dim)
            @test basis == src(D)
            antidiff_dest = dest(D)

            coef1 = random_expansion(basis)
            coef2 = D*coef
            e1 = SetExpansion(basis, coef)
            e2 = SetExpansion(antidiff_dest, coef2)

            x = fixed_point_in_domain(basis)
            delta = sqrt(eps(T))/10
            N = ndims(basis)
            if N > 1
                unit_vector = zeros(T, ndims(basis))
                unit_vector[dim] = 1
                x2 = x + SVector{N}(delta*unit_vector)
            else
                x2 = x+delta
            end
            @test abs( (e2(x2)-e2(x))/delta - e1(x) ) / abs(e1(x)) < 2000*sqrt(eps(T))
        end
    end

    ## Test associated transform
    if BF.has_transform(basis)
        # We have to look into this test
        @test has_transform(basis) == has_transform(basis, DiscreteGridSpace(grid(basis)))
        # Check whether it is unitary
        tbasis = transform_set(basis)
        t = transform_operator(tbasis, basis)
        it = transform_operator(basis, tbasis)
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

        # Verify the pre and post operators and their inverses
        pre1 = transform_operator_pre(tbasis, basis)
        post1 = transform_operator_post(tbasis, basis)
        pre2 = transform_operator_pre(basis, tbasis)
        post2 = transform_operator_post(basis, tbasis)
        # - try interpolation using transform+pre/post-normalization
        x = coefficients(random_expansion(tbasis))
        e = SetExpansion(basis, (post1*t*pre1)*x)
        g = grid(basis)
        @test maximum(abs(e(g)-x)) < sqrt(eps(T))
        # - try evaluation using transform+pre/post-normalization
        e = random_expansion(basis)
        x1 = (post2*it*pre2)*coefficients(e)
        x2 = e(grid(basis))
        @test maximum(abs(x1-x2)) < sqrt(eps(T))

        # TODO: check the transposes and inverses here
    end


    ## Test interpolation operator on a suitable interpolation grid
    if is_basis(basis)
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

    ## Test approximation operator
    if supports_approximation(basis)
        A = approximation_operator(basis)
        f = suitable_function(basis)
        e = SetExpansion(basis, A*f)
        x = random_point_in_domain(basis)
        # We choose a fairly large error, because the ndof's can be very small.
        # We don't want to test convergence, only that something terrible did
        # not happen, so an error of 1e-3 will do.
        @test abs(e(x)-f(x...)) < 1e-3
    end
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
    @test bf(x1, x2, x3) ≈ eval_set_element(a, 3, x1) * eval_set_element(b, 4, x2) * eval_set_element(c, 5, x3)

    # Can you iterate over the product set?
    z = zero(T)
    i = 0
    for f in d
        z += f(x1, x2, x3)
        i = i+1
    end
    @test i == length(d)
    @test abs(-0.5 - z) < 0.01

    z = zero(T)
    l = 0
    for i in eachindex(d)
        f = d[i]
        z += f(x1, x2, x3)
        l = l+1
    end
    @test l == length(d)
    @test abs(-0.5 - z) < 0.01

    # Indexing with ranges
    # @test try
    #     d[CartesianRange(CartesianIndex(1,1,1),CartesianIndex(3,4,5))]
    #     true
    # catch
    #     false
    # end
    # Is an error thrown if you index with a range that is out of range?
end
