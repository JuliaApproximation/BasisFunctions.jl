# test_suite.jl
module test_suite

using Base.Test

using BasisFunctions
using FixedSizeArrays
using Debug
BF = BasisFunctions


########
# Auxiliary functions
########

# Keep track of successes, failures and errors
global failures = 0
global successes = 0
global errors = 0

# Custom test handler
custom_handler(r::Test.Success) = begin print_with_color(:green, "#\tSuccess "); println("on $(r.expr)"); global successes+=1;  end
custom_handler(r::Test.Failure) = begin print_with_color(:red, "\"\tFailure "); println("on $(r.expr)\""); global failures+=1; end
custom_handler(r::Test.Error) = begin println("\"\t$(typeof(r.err)) in $(r.expr)\""); global errors+=1; end
#custom_handler(r::Test.Error) = Base.showerror(STDOUT,r);

function message(y)
    println("\"\t",typeof(y),"\"")
    Base.showerror(STDOUT,y)
    global errors+=1
end

function message(y, backtrace)
    rethrow(y)
    println("\tFailure due to ",typeof(y))
    Base.showerror(STDOUT,y)
    Base.show_backtrace(STDOUT,backtrace)
    global errors+=1
end

function delimit(s::AbstractString)
    println("############")
    println("# ",s)
    println("############")
end


##########
# Testing
##########


# Interpolate linearly between the left and right endpoint, using the value 0 <= scalar <= 1
function point_in_domain(basis::FunctionSet{1}, scalar)
    # Try to find an interval within the support of the basis
    a = left(basis)
    b = right(basis)

    T = numtype(basis)
    # Avoid infinities for some bases on the real line
    if isinf(a)
        a = -T(1)
    end
    if isinf(b)
        b = T(1)
    end
    x = scalar * a + (1-scalar) * b
end


# Interpolate linearly between the left and right endpoint, using the value 0 <= scalar <= 1
function point_in_domain{N}(basis::FunctionSet{N}, scalar)
    # Try to find an interval within the support of the basis
    a = left(basis)
    b = right(basis)

    T = numtype(basis)
    # Avoid infinities for some bases on the real line
    a = Vector(a)
    b = Vector(b)
    for i in 1:N
        if isinf(a[i])
            a[i] = -T(1)
        end
        if isinf(b[i])
            b[i] = T(1)
        end
    end
    a = Vec{N,T}(a)
    b = Vec{N,T}(a)
    x = scalar * a + (1-scalar) * b
end


function random_point_in_domain{N,T}(basis::FunctionSet{N,T})
    w = one(T) * rand()
    point_in_domain(basis, w)
end

function fixed_point_in_domain{N,T}(basis::FunctionSet{N,T})
    w = 1/sqrt(T(2))
    point_in_domain(basis, w)
end

random_index(basis::FunctionSet) = 1 + Int(floor(rand()*length(basis)))

Base.rationalize{N}(x::Vec{N,Float64}) = Vec{N,Rational{Int}}([rationalize(x_i) for x_i in x])

Base.rationalize{N}(x::Vec{N,BigFloat}) = Vec{N,Rational{BigInt}}([rationalize(x_i) for x_i in x])


function test_generic_interface(basis, SET)
    delimit("Generic interface for $(name(basis))")

    ELT = eltype(basis)
    T = numtype(basis)
    n = length(basis)

    @test typeof(basis) <: SET

    # Check consistency of traits
    @test isreal(basis) == isreal(SET)()
    @test is_orthogonal(basis) == is_orthogonal(SET)()
    @test is_biorthogonal(basis) == is_biorthogonal(SET)()
    @test is_basis(basis) == is_basis(SET)()
    @test is_frame(basis) == is_frame(SET)()

    if is_basis(SET) == True
        @test is_frame(SET) == True
    end
    if is_orthogonal(SET) == True
        @test is_biorthogonal(SET) == True
    end

    ## Test dimensions
    @test index_dim(basis) == index_dim(SET)
    @test index_dim(basis) == length(size(basis))
    s = size(basis)
    for i = 1:length(s)
        @test size(basis, i) == s[i]
    end
    @test dim(basis) == dim(typeof(basis))
    @test index_dim(basis) <= dim(basis)

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
    @test functionset(bf) == basis

    x = fixed_point_in_domain(basis)
    @test bf(x) ≈ call(basis, idx, x)

    # Create a random expansion in the basis to test expansion interface
    e = random_expansion(basis)
    coef = coefficients(e)

    ## Does evaluating an expansion equal the sum of coefficients times basis function calls?
    x = fixed_point_in_domain(basis)
    @test e(x) ≈ sum([coef[i] * basis(i,x) for i in eachindex(coef)])

    ## Test evaluation on an array
    ARRAY_TYPE = typeof(fixed_point_in_domain(basis))
    x_array = ARRAY_TYPE[random_point_in_domain(basis) for i in 1:10]
    z = e(x_array)
    @test  z ≈ ELT[ e(x_array[i]) for i in eachindex(x_array) ]


    ## Verify evaluation on the associated grid
    if BF.has_grid(basis)

        grid1 = grid(basis)
        @test length(grid1) == length(basis)

        z = e(grid1)
        E = evaluation_operator(set(e),DiscreteGridSpace(grid(set(e))))
        @test z[:] ≈ ELT[ e(grid1[i]) for i in eachindex(grid1) ]
    end

    ## Test output type of calling function

    types_correct = true
    for x in [ fixed_point_in_domain(basis) rationalize(point_in_domain(basis, 0.5)) ]
        for idx in [1 2 n>>1 n-1 n]
            z = call(basis, idx, x)
            types_correct = types_correct & (typeof(z) == ELT)
        end
    end
    @test types_correct

    if dim(basis) == 1

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
        z = basis(idx, grid2)
        @test  z ≈ ELT[ basis(idx, grid2[i]) for i in eachindex(grid2) ]

    end

    ## Test extensions
    if BF.has_extension(basis)
        n2 = extension_size(basis)
        basis2 = similar(basis, n2)
        E = extension_operator(basis, basis2)
        e1 = random_expansion(basis)
        e2 = E * e1        
        x1 = 1/2 * (left(basis) + right(basis))
        @test e1(x1) ≈ e2(x1)
        x2 = 0.3 * left(basis) + 0.7 * right(basis)
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
        delta = sqrt(eps(T))
        @test abs( (e1(x+delta)-e1(x))/delta - e2(x) ) / abs(e2(x)) < 150delta
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
        delta = sqrt(eps(T))
        
        @test abs( (e2(x+delta)-e2(x))/delta - e1(x) ) / abs(e1(x)) < 150delta
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
            println("Skipping conditioning test for BigFloat")
        end
        AI = matrix(it)
        if T == Float64
            @test cond(AI) ≈ 1
        else
            println("Skipping conditioning test for BigFloat")
        end

#       TODO: test accuracy of combination of transform_operator and normalization_operator
#        b = one(ELT) * rand(size(basis))
#        c = t * b
#        d = it * c
#        @test sumabs(d-b) ≈ 0
    end
    # TODO: Test interpolation on associated grid
end


#####
# Tensor sets
#####

function test_tensor_sets(T)
    delimit("Tensor sets")

    a = FourierBasis(12)
    b = FourierBasis(13)
    c = FourierBasis(11)
    d = TensorProductSet(a,b,c)

    bf = d[3,4,5]
    x1 = T(2//10)
    x2 = T(3//10)
    x3 = T(4//10)
    @test bf(x1, x2, x3) ≈ call(a, 3, x1) * call(b, 4, x2) * call(c, 5, x3)

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


#####
# Tensor operators
#####

function test_tensor_operators(T)
    delimit("Tensor operators")

    m1 = 3
    n1 = 4
    m2 = 10
    n2 = 24
    A1 = MatrixOperator(map(T,rand(m1,n1)))
    A2 = MatrixOperator(map(T,rand(m2,n2)))
    A = A1 ⊗ A2

    # Apply the tensor-product operator
    b = map(T, rand(n1, n2))
    c = zeros(T, m1, m2)
    apply!(A, c, b)

    # Apply the operators manually, row by row and then column by column
    intermediate = zeros(T, m1, n2)
    dest = zeros(T, m1)
    for i = 1:n2
        apply!(A1, dest, b[:,i])
        intermediate[:,i] = dest
    end
    c2 = similar(c)
    dest = zeros(T, m2)
    for i = 1:m1
        apply!(A2, dest, intermediate[i,:])
        c2[i,:] = dest
    end
    @test sum(abs(c-c2)) ≈ 0
end

#####
# Fourier series
#####
function test_fourier_series(T)
    delimit("Fourier series")
    
    @test isreal(FourierBasis) == False

    ## Even length
    n = 12
    a = -T(1.2)
    b = T(3.4)
    fb = rescale(FourierBasis(n,T), a, b)
    @test isreal(fb) == False()

    @test left(fb) ≈ a
    @test right(fb) ≈ b

    @test grid(fb) == PeriodicEquispacedGrid(n, a, b)

    # Take a random point in the domain
    x = T(a+rand()*(b-a))
    y = (x-a)/(b-a)

    # Is the 0-index basis function the constant 1?
    freq = 0
    idx = frequency2idx(set(fb), freq)
    @test fb(idx, x) ≈ 1

    # Evaluate in a point in the interior
    freq = 3
    idx = frequency2idx(set(fb), freq)
    @test call(fb, idx, x) ≈ exp(2*T(pi)*1im*freq*y)

    # Evaluate the largest frequency, which is a cosine in this case
    freq = n >> 1
    idx = frequency2idx(set(fb), freq)
    @test call(fb, idx, x) ≈ cos(2*T(pi)*freq*y)

    # Evaluate an expansion
    coef = T[1; 2; 3; 4] * (1+im)
    e = SetExpansion(rescale(FourierBasis(4,T), a, b), coef)
    @test e(x) ≈ coef[1]*T(1) + coef[2]*exp(2*T(pi)*im*y) + coef[3]*cos(4*T(pi)*y) + coef[4]*exp(-2*T(pi)*im*y)

    # Check type promotion: evaluate at an integer and at a rational point
    for i in [1 2]
        @test typeof(call(fb, i, 0)) == Complex{T}
        @test typeof(call(fb, i, 1//2)) == Complex{T}
    end

    # Try an extension
    n = 12
    # This line used to say T, however we don't allow real coefficients for Fourier bases.
    coef = map(complexify(T), rand(n))
    b1 = rescale(FourierBasis(n,T), a, b)
    b2 = rescale(FourierBasis(n+1,T), a, b)
    b3 = rescale(FourierBasis(n+15,T), a, b)
    E2 = extension_operator(b1, b2)
    E3 = extension_operator(b1, b3)
    e1 = SetExpansion(b1, coef)
    e2 = SetExpansion(b2, E2*coef)
    e3 = SetExpansion(b3, E3*coef)
    x = T(2//10)
    @test e1(x) ≈ e2(x)
    @test e1(x) ≈ e3(x)


    # Differentiation test
    coef = map(complexify(T), rand(Float64, size(fb)))
    D = differentiation_operator(fb)
    coef2 = D*coef
    e1 = SetExpansion(fb, coef)
    e2 = SetExpansion(rescale(FourierBasis(length(fb)+1,T),left(fb),right(fb)), coef2)

    
    x = T(2//10)
    delta = sqrt(eps(T))
    @test abs( (e1(x+delta)-e1(x))/delta - e2(x) ) / abs(e2(x)) < 100delta



    ## Odd length
    b = rescale(FourierBasis(13,T), -one(T), one(T))

    @test isreal(b) == False()

    # Is the 0-index basis function the constant 1?
    freq = 0
    idx = frequency2idx(set(b), freq)
    @test call(b, idx, T(2//10)) ≈ 1

    # Evaluate in a point in the interior
    freq = 3
    idx = frequency2idx(set(b), freq)
    x = T(2//10)
    y = (x+1)/2
    @test call(b, idx, x) ≈ exp(2*T(pi)*1im*freq*y)

    # Evaluate an expansion
    coef = [one(T)+im; 2*one(T)-im; 3*one(T)+2im]
    b = rescale(FourierBasis(3,T), -one(T), one(T))
    e = SetExpansion(b, coef)
    x = T(2//10)
    y = (x+1)/2
    @test e(x) ≈ coef[1]*one(T) + coef[2]*exp(2*T(pi)*im*y) + coef[3]*exp(-2*T(pi)*im*y)
    # evaluate on a grid
    g = grid(b)
    result = e(g)
    @test sum([abs(result[i] - e(g[i])) for i in 1:length(g)]) ≈ 0

    # Try an extension
    n = 13
    coef = map(complexify(T), rand(n))
    b1 = FourierBasis(n, T)
    b2 = FourierBasis(n+1, T)
    b3 = FourierBasis(n+15, T)
    E2 = Extension(b1, b2)
    E3 = Extension(b1, b3)
    e1 = SetExpansion(b1, coef)
    e2 = SetExpansion(b2, E2*coef)
    e3 = SetExpansion(b3, E3*coef)
    x = T(2//10)
    @test e1(x) ≈ e2(x)
    @test e1(x) ≈ e3(x)

    # Restriction
    n = 14
    b1 = FourierBasis(n, T)
    b2 = FourierBasis(n-1, T)
    b3 = FourierBasis(n-5, T)
    E1 = Restriction(b1, b2)    # source has even length
    E2 = Restriction(b2, b3)    # source has odd length
    coef1 = map(complexify(T), rand(length(b1)))
    coef2 = E1*coef1
    coef3 = E2*coef2
    @test reduce(&, [ coef2[i+1] == coef1[i+1] for i=0:BasisFunctions.nhalf(b2) ] )
    @test reduce(&, [ coef2[end-i+1] == coef1[end-i+1] for i=1:BasisFunctions.nhalf(b2) ] )
    @test reduce(&, [ coef3[i+1] == coef2[i+1] for i=0:BasisFunctions.nhalf(b3) ] )
    @test reduce(&, [ coef3[end-i+1] == coef2[end-i+1] for i=1:BasisFunctions.nhalf(b3) ] )

    # Differentiation test
    coef = map(complexify(T), rand(Float64, size(b)))
    D = differentiation_operator(b)
    coef2 = D*coef
    e1 = SetExpansion(b, coef)
    e2 = SetExpansion(b, coef2)

    x = T(2//10)
    delta = sqrt(eps(T))
    @test abs( (e1(x+delta)-e1(x))/delta - e2(x) ) / abs(e2(x)) < 150delta

    # Transforms
    b1 = FourierBasis(161, T)
    A = approximation_operator(b1)
    f = x -> 1/(2+cos(pi*x))
    e = approximate(b1, f)
    x0 = T(1//2)
    @test abs(e(T(x0))-f(x0)) < sqrt(eps(T))
end

# Chebyshev polynomials
function test_chebyshev(T)
    delimit("Chebyshev expansions")

    b1 = ChebyshevBasis(160, T)
    A = approximation_operator(b1)
    f = exp
    e = approximate(b1, exp)
    x0 = T(1//2)
    @test abs(e(T(x0))-f(x0)) < sqrt(eps(T))
end

#####
# Grids
#####

function test_grids(T)
    delimit("Grids and tensor grids")

    ## Equispaced grids
    len = 120
    a = -T(1.2)
    b = T(3.5)
    g = EquispacedGrid(len, a, b)

    idx = 5
    @test g[idx] ≈ a + (idx-1) * (b-a)/(len-1)
    @test g[len] ≈ b
    @test_throws BoundsError g[len+1] == b

    # Test iterations
    z = zero(T)
    i = 0
    for x in g
        z = z + x
        i += 1
    end
    @test z ≈ len * (a+b)/2
    @test i == length(g)

    z = zero(T)
    i = 0
    for x in g
        z = z + x
        i += 1
    end
    @test z ≈ len * (a+b)/2
    @test i == length(g)

    z = zero(T)
    i = 0
    for idx in eachindex(g)
        z = z + g[idx]
        i += 1
    end
    @test z ≈ len * (a+b)/2
    @test i == length(g)


    ## Periodic equispaced grids
    len = 120
    a = -T(1.2)
    b = T(3.5)
    g = PeriodicEquispacedGrid(len, a, b)

    idx = 5
    @test g[idx] ≈ a + (idx-1) * (b-a)/len
    @test g[len] ≈ b - stepsize(g)
    @test_throws BoundsError g[len+1] == b

    z = zero(T)
    i = 0
    for x in g
        z = z + x
        i += 1
    end
    @test z ≈ (len-1)*(a+b)/2 + a
    @test i == length(g)

    z = zero(T)
    i = 0
    for x in g
        z = z + x
        i += 1
    end
    @test z ≈ (len-1)*(a+b)/2 + a
    @test i == length(g)

    z = zero(T)
    i = 0
    for idx in eachindex(g)
        z = z + g[idx]
        i += 1
    end
    @test z ≈ (len-1)*(a+b)/2 + a
    @test i == length(g)

    ## Tensor product grids
    len = 120
    g1 = PeriodicEquispacedGrid(len, -one(T), one(T))
    g2 = EquispacedGrid(len, -one(T), one(T))
    g = g1 ⊗ g2
    @test length(g) == length(g1) * length(g2)
    @test size(g) == (length(g1),length(g2))

    idx1 = 5
    idx2 = 9
    x1 = g1[idx1]
    x2 = g2[idx2]
    x = g[idx1,idx2]
    @test x[1] ≈ x1
    @test x[2] ≈ x2

    z = zero(T)
    i = 0
    for x in g
        z = z + sum(x)
        i += 1
    end
    @test z ≈ -len
    @test i == length(g)

    z = zero(T)
    i = 0
    for x in g
        z = z + sum(x)
        i += 1
    end
    @test z ≈ -len
    @test i == length(g)

    z = zero(T)
    i = 0
    for idx in eachindex(g)
        z = z + sum(g[idx])
        i += 1
    end
    @test z ≈ -len
    @test i == length(g)

    # Test a tensor of a tensor
    g3 = g ⊗ g2
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

end

#####
# Orthogonal polynomials
#####

function test_ops(T)
    delimit("Chebyshev polynomials")

    bc = ChebyshevBasis(12, T)

    x1 = T(4//10)
    @test bc(4, x1) ≈ cos(3*acos(x1))


    delimit("Legendre polynomials")

    bl = LegendreBasis{T}(15)

    x1 = T(4//10)
    @test abs(bl(6, x1) - 0.27064) < 1e-5


    delimit("Jacobi polynomials")

    bj = JacobiBasis(15, T(2//3), T(3//4))

    x1 = T(4//10)
    @test abs(bj(6, x1) - 0.335157) < 1e-5


    delimit("Laguerre polynomials")

    bl = LaguerreBasis(15, T(1//3))

    x1 = T(4//10)
    @test abs(bl(6, x1) + 0.08912346) < 1e-5

    delimit("Hermite polynomials")

    bh = HermiteBasis{T}(15)

    x1 = T(4//10)
    @test abs(bh(6, x1) - 38.08768) < 1e-5

end


Test.with_handler(custom_handler) do


    for T in (Float64,BigFloat)

        println()
        println("T is ", T)

        SETS = (FourierBasis, ChebyshevBasis, ChebyshevBasisSecondKind, LegendreBasis, LaguerreBasis, HermiteBasis,
            PeriodicSplineBasis, FullSplineBasis)
        for SET in SETS
            # Choose an odd and even number of degrees of freedom
            for n in (8, 11)
                basis = instantiate(SET, n, T)

                @test length(basis) == n
                @test numtype(basis) == T
                @test promote_type(eltype(basis),numtype(basis)) == eltype(basis)

                test_generic_interface(basis, SET)
            end
        end

        b = FourierBasis(10) ⊗ ChebyshevBasis(12)
        test_generic_interface(b, typeof(b))

        test_tensor_sets(T)

        test_tensor_operators(T)

        test_fourier_series(T)

        test_chebyshev(T)

        test_grids(T)

        test_ops(T)

    end # for T in...

end # Test.with_handler


# Diagnostics
println("Succes rate:\t$successes/$(successes+failures+errors)")
println("Failure rate:\t$failures/$(successes+failures+errors)")
println("Error rate:\t$errors/$(successes+failures+errors)")

end # module

