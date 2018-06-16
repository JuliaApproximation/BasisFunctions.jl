# test_generic_dicts.jl

TEST_CONTINUOUS = true
# We try to test approximation for all function sets, except those that
# are currently known to fail for lack of an implementation.
supports_approximation(s::Dictionary) = true
# Laguerre and Hermite fail due to linear algebra problems in BigFloat
supports_approximation(s::LaguerrePolynomials{BigFloat}) = false
supports_approximation(s::HermitePolynomials{BigFloat}) = false

# It is difficult to do approximation in subsets and operated sets generically
supports_approximation(s::Subdictionary) = false
supports_approximation(s::OperatedDict) = false

supports_approximation(s::TensorProductDict) =
    reduce(&, map(supports_approximation, elements(s)))

supports_interpolation(s::Dictionary) = is_basis(s)

# disable for now
supports_interpolation(s::SingletonSubdict) = false

# Pick a simple function to approximate
suitable_function(s::Dictionary1d) = x->exp(x/supremum(support(s)))

# Make a simple periodic function for Fourier and other periodic sets
suitable_function(set::FourierBasis) =  x -> 1/(10+cos(2*pi*x))
#suitable_function(set::PeriodicSplineBasis) =  x -> 1/(10+cos(2*pi*x))
suitable_function(set::BasisFunctions.PeriodicTranslationDict) =  x -> 1/(10+cos(2*pi*x))
# The function has to be periodic and even symmetric
suitable_function(set::CosineSeries) =  x -> 1/(10+cos(2*pi*x))
# The function has to be periodic and odd symmetric
suitable_function(set::SineSeries) =  x -> x^3*(1-x)^3
# We use a function that is smooth and decays towards infinity
suitable_function(set::LaguerrePolynomials) = x -> 1/(1000+(2x)^2)
suitable_function(set::HermitePolynomials) = x -> 1/(1000+(2x)^2)


suitable_function(set::OperatedDict) = suitable_function(src(set))

function suitable_interpolation_grid(basis::Dictionary)
    if BF.has_grid(basis)
        grid(basis)
    else
        T = domaintype(basis)
        # A midpoint grid avoids duplication of the endpoints for a periodic basis
        MidpointEquispacedGrid(length(basis), point_in_domain(basis, T(0)), point_in_domain(basis, T(1)))
    end
end

suitable_interpolation_grid(basis::TensorProductDict) =
    ProductGrid(map(suitable_interpolation_grid, elements(basis))...)

suitable_interpolation_grid(basis::SineSeries) = MidpointEquispacedGrid(length(basis), 0, 1, domaintype(basis))

suitable_interpolation_grid(basis::WeightedDict) = suitable_interpolation_grid(superdict(basis))

suitable_interpolation_grid(basis::OperatedDict) = suitable_interpolation_grid(src_dictionary(basis))


# Make a tensor product of suitable functions
function suitable_function(s::TensorProductDict)
    if dimension(s) == 2
        f1 = suitable_function(element(s,1))
        f2 = suitable_function(element(s,2))
        (x,y) -> f1(x)*f2(y)
    elseif dimension(s) == 3
        f1 = suitable_function(element(s,1))
        f2 = suitable_function(element(s,2))
        f3 = suitable_function(element(s,3))
        (x,y,z) -> f1(x)*f2(y)*f3(z)
    end
    # We should never get here
end

# Make a suitable function by undoing the map
function suitable_function(s::MappedDict)
    f = suitable_function(superdict(s))
    m = inv(mapping(s))
    x -> f(m*x)
end

function suitable_function(s::WeightedDict1d)
    f = suitable_function(superdict(s))
    g = weightfunction(s)
    x -> g(x) * f(x)
end

function suitable_function(s::WeightedDict2d)
    f = suitable_function(superdict(s))
    g = weightfunction(s)
    (x,y) -> g(x, y) * f(x, y)
end


widen_type(::Type{T}) where {T <: Number} = widen(T)
widen_type(::Type{Tuple{A,B}}) where {A,B} = Tuple{widen(A),widen(B)}
widen_type(::Type{Tuple{A,B,C}}) where {A,B,C} = Tuple{widen(A),widen(B),widen(C)}

test_tolerance(::Type{T}) where {T <: Number} = sqrt(eps(T))
test_tolerance(::Type{Complex{T}}) where {T <: Number} = sqrt(eps(T))
test_tolerance(::Type{T}) where {T} = test_tolerance(float_type(T))

function test_generic_dict_interface(basis, span = Span(basis))
    ELT = coefficient_type(span)
    T = domaintype(basis)
    FT = float_type(T)
    RT = codomaintype(basis)
    SY = codomaintype(span)

    # Does the set of the span agree with the basis?
    @test typeof(dictionary(span)) == typeof(basis)

    # Do the domain types of basis and span agree?
    @test domaintype(basis) == domaintype(span)
    @test typeof(one(coefficient_type(span)) * one(codomaintype(basis))) == codomaintype(span)

    n = length(basis)
    if is_basis(basis)
        @test is_frame(basis)
    end
    if is_orthogonal(basis)
        @test is_biorthogonal(basis)
    end

    # Test type promotion
    @test domaintype(promote_domaintype(basis, T)) == T
    T2 = widen_type(T)
    basis2 = promote_domaintype(basis, T2)
    @test domaintype(basis2) == T2


    ## Test dimensions
    s = size(basis)
    for i = 1:length(s)
        @test size(basis, i) == s[i]
    end

    # Bounds checking
    # disable periodic splines for now, since sometimes left(basis,idx) is not
    # in_support currently...
    if (dimension(basis) == 1) && ~(typeof(basis) <: BasisFunctions.CompactPeriodicTranslationDict)
        s = support(basis)
        l = infimum(s)
        r = supremum(s)
        if ~isinf(l)
            @test in_support(basis, l)
            # This assumes test_tolerance(ELT) is equal to tolerance(dict), should probably use the latter here
            @test in_support(basis, l.-1/10*test_tolerance(ELT))
            @test ~in_support(basis, l.-1)
        end
        if ~isinf(r)
            @test in_support(basis, r)
            @test in_support(basis, r.+1/10*test_tolerance(ELT))
            @test ~in_support(basis, r.+1)
        end
        if ~isinf(l) && ~isinf(r) && ~(typeof(basis) <: BasisFunctions.PiecewiseDict)
            @test in_support(basis, 1, 1/2*(l + r))
        end
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
    # is always the set that was indexed is false, e.g., for multidicts.
    #@test dictionary(bf) == basis

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
    @test bf(x) ≈ eval_element(basis, idx, x)

    if ! (typeof(basis) <: HermitePolynomials)
        x_outside = point_outside_domain(basis)
        @test bf(x_outside) == 0
    end


    # Create a random expansion in the basis to test expansion interface
    e = random_expansion(span)
    coef = coefficients(e)

    ## Does evaluating an expansion equal the sum of coefficients times basis function calls?
    x = fixed_point_in_domain(basis)
    @test e(x) ≈ sum([coef[i] * basis[i](x) for i in eachindex(coef)])

    ## Test evaluation on an array
    x_array = [random_point_in_domain(basis) for i in 1:10]
    z = map(e, x_array)
    @test  z ≈ ELT[ e(x_array[i]) for i in eachindex(x_array) ]

    # Test linearization of coefficients
    linear_coefs = zeros(coeftype(span), length(basis))
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
        E = evaluation_operator(basis, gridbasis(basis))
        z3 = E * coefficients(e)
        @test z1 ≈ z3
    end

    ## Test output type of calling function

    types_correct = true
    # The comma in the line below is important, otherwise the two static vectors
    # are combined into a statix matrix.
    for x in [ fixed_point_in_domain(basis), rationalize(point_in_domain(basis, real(FT)(0.5))) ]
        if length(basis) > 1
            indices = [1 2 n>>1 n-1 n]
        else
            # For a singleton subset we can only use index 1
            indices = 1
        end
        for idx in indices
            z = eval_element(basis, idx, x)
            types_correct = types_correct & (typeof(z) == RT)
        end
    end
    @test types_correct

    if dimension(basis) == 1

        # Test evaluation on a different grid on the support of the basis
        a = infimum(support(basis))
        b = supremum(support(basis))

        if isinf(a)
            a = -T(1)
        end
        if isinf(b)
            b = T(1)
        end

        grid2 = EquispacedGrid(n+3, T(a)+test_tolerance(ELT), T(b)-test_tolerance(ELT))
        z = e(grid2)
        @test z ≈ ELT[ e(grid2[i]) for i in eachindex(grid2) ]

        # Test evaluation on a grid of a single basis function
        idx = random_index(basis)
        z = eval_element(basis, idx, grid2)
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

    # Verify whether evaluation in a larger grid works
    if BF.has_extension(basis) && BF.has_grid(basis)
        basis_ext = extend(basis)
        grid_ext = grid(basis_ext)
        L = evaluation_operator(basis, grid_ext)
        e = random_expansion(basis)
        z = L*e
        L2 = evaluation_operator(basis_ext, grid_ext) * extension_operator(basis, basis_ext)
        z2 = L2*e
        @test maximum(abs.(z-z2)) < 20test_tolerance(ELT)
        # In the future, when we can test for 'fastness' of operators
        # @test is_fast(L2) == is_fast(L)
    end

    ## Test derivatives
    if BF.has_derivative(basis)
        for dim in 1:dimension(basis)
            # TODO: Sort out problem with dim and multidict
            if dimension(basis)>1
                D = differentiation_operator(basis; dim=dim)
            else
                D = differentiation_operator(basis)
            end
            @test basis == src(D)
            diff_dest = dest(D)

            coef1 = random_expansion(basis)
            coef2 = D*coef
            e1 = Expansion(basis, coef)
            e2 = Expansion(diff_dest, coef2)

            x = fixed_point_in_domain(basis)
            delta = test_tolerance(ELT)/10
            N = dimension(basis)
            if N > 1
                unit_vector = zeros(FT, dimension(basis))
                unit_vector[dim] = 1
                x2 = x + SVector{N}(delta*unit_vector)
            else
                x2 = x+delta
            end
            @test abs( (e1(x2)-e1(x))/delta - e2(x) ) / abs(e2(x)) < 2000*test_tolerance(ELT)
        end

        if dimension(basis) == 1
            x = fixed_point_in_domain(basis)
            D = differentiation_operator(basis)
            # Verify derivatives in three basis functions: the first, the last,
            # and the middle one
            i1 = 1
            i2 = length(basis)
            i3 = (i1+i2) >> 1

            c1 = zero(span)
            c1[i1] = 1
            u1 = D*c1
            @test abs(u1(x) - eval_element_derivative(basis, i1, x)) < test_tolerance(ELT)

            c2 = zero(span)
            c2[i2] = 1
            u2 = D*c2
            @test abs(u2(x) - eval_element_derivative(basis, i2, x)) < test_tolerance(ELT)

            c3 = zero(span)
            c3[i3] = 1
            u3 = D*c3
            @test abs(u3(x) - eval_element_derivative(basis, i3, x)) < test_tolerance(ELT)
        end
    end

    ## Test antiderivatives
    if BF.has_antiderivative(basis)
        for dim in 1:dimension(basis)
            D = antidifferentiation_operator(basis; dim=dim)
            @test basis == src(D)
            antidiff_dest = dest(D)

            coef1 = random_expansion(basis)
            coef2 = D*coef
            e1 = Expansion(basis, coef)
            e2 = Expansion(antidiff_dest, coef2)

            x = fixed_point_in_domain(basis)
            delta = test_tolerance(ELT)/10
            N = dimension(basis)
            if N > 1
                unit_vector = zeros(FT, dimension(basis))
                unit_vector[dim] = 1
                x2 = x + SVector{N}(delta*unit_vector)
            else
                x2 = x+delta
            end
            @test abs( (e2(x2)-e2(x))/delta - e1(x) ) / abs(e1(x)) < 2000*test_tolerance(ELT)
        end
    end

    ## Test associated transform
    if BF.has_transform(basis)
        # We have to look into this test
        @test has_transform(basis) == has_transform(basis, gridbasis(basis))
        # Check whether it is unitary
        tbasis = transform_space(basis)
        t = transform_operator(tbasis, basis)
        it = transform_operator(basis, tbasis)
        A = matrix(t)
        if has_unitary_transform(basis)
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
        end

        # Verify the pre and post operators and their inverses
        pre1 = transform_operator_pre(tbasis, basis)
        post1 = transform_operator_post(tbasis, basis)
        pre2 = transform_operator_pre(basis, tbasis)
        post2 = transform_operator_post(basis, tbasis)
        # - try interpolation using transform+pre/post-normalization
        x = rand(Span(tbasis))
        e = Expansion(basis, (post1*t*pre1)*x)
        g = grid(basis)
        @test maximum(abs.(e(g)-x)) < test_tolerance(ELT)
        # - try evaluation using transform+pre/post-normalization
        e = random_expansion(basis)
        x1 = (post2*it*pre2)*coefficients(e)
        x2 = e(grid(basis))
        @test maximum(abs.(x1-x2)) < test_tolerance(ELT)

        # Verify the transposes and inverses
        ## if has_unitary_transform(basis)
        ##     @test maximum(abs.( (t' * t)*x-x)) < test_tolerance(ELT)
        ##     @test maximum(abs.( (it' * it)*x-x)) < test_tolerance(ELT)
        ##     @test maximum(abs.( (inv(t) * t)*x-x)) < test_tolerance(ELT)
        ##     @test maximum(abs.( (inv(it) * it)*x-x)) < test_tolerance(ELT)
        ##     @test maximum(abs.( (it * t)*x-x)) < test_tolerance(ELT)
        ## end
    end


    ## Test interpolation operator on a suitable interpolation grid
    if supports_interpolation(basis)
        g = suitable_interpolation_grid(basis)
        I = interpolation_operator(basis, g)
        x = rand(gridspace(g, coeftype(basis)))
        e = Expansion(basis, I*x)
        @test maximum(abs.(e(g)-x)) < 100test_tolerance(ELT)
    end

    ## Test evaluation operator
    g = suitable_interpolation_grid(basis)
    E = evaluation_operator(basis, g)
    e = random_expansion(basis)
    y = E*e
    @test maximum([abs.(e(g[i])-y[i]) for i in eachindex(g)]) < test_tolerance(ELT)

    ## Test approximation operator
    if supports_approximation(basis)
        A = approximation_operator(basis)
        f = suitable_function(basis)
        e = Expansion(basis, A*f)
        x = random_point_in_domain(basis)

        # We choose a fairly large error, because the ndof's can be very small.
        # We don't want to test convergence, only that something terrible did
        # not happen, so an error of 1e-3 will do.
        @test abs.(e(x)-f(x...)) < 1e-3

        # # continuous operator only supported for 1 D
        # No efficient implementation for BigFloat to construct full gram matrix.
        # if dimension(basis)==1 && is_biorthogonal(basis) && !(   ((typeof(basis) <: OperatedDict) || (typeof(basis)<:BasisFunctions.ConcreteDerivedDict) || typeof(basis)<:WeightedDict) && eltype(basis)==BigFloat)
        if TEST_CONTINUOUS && dimension(basis)==1 && is_biorthogonal(basis) && !((typeof(basis) <: DerivedDict) && real(codomaintype(basis))==BigFloat)
            e = approximate(basis, f; discrete=false, reltol=1e-6, abstol=1e-6)
            @test abs(e(x)-f(x...)) < 1e-3
        end
    end
end
