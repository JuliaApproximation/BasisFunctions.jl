# test_suite.jl

module test_suite

using BasisFunctions
using Base.Test

########
# Auxiliary functions
########

# Keep track of successes, failures and errors
global failures = 0
global successes = 0
global errors = 0

# Custom test handler
custom_handler(r::Test.Success) = begin println("#\tSucces on $(r.expr)"); global successes+=1;  end
custom_handler(r::Test.Failure) = begin println("\"\tFailure on $(r.expr)\""); global failures+=1; end
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

function delimit(s::String)
    println("############")
    println("# ",s)
    println("############")
end

# Approximate equality
approx(a, b, eps) = abs(b-a) < eps

# Strong equality: within 10 times eps
≃(a,b) = ≃(promote(a,b)...)
≃{T <: FloatingPoint}(a::T,b::T) = approx(a, b, 10eps(T))
≃{T <: FloatingPoint}(a::Complex{T}, b::Complex{T}) = approx(a, b, 10eps(T))

# Weaker equality: within 10 times sqrt(eps)
≈(a,b) = ≈(promote(a,b)...)
≈{T <: FloatingPoint}(a::T, b::T) = approx(a, b, 10*sqrt(eps(T)))
≈{T <: FloatingPoint}(a::Complex{T}, b::Complex{T}) = approx(a, b, 10*sqrt(eps(T)))

##########
# Testing
##########


Test.with_handler(custom_handler) do


    for T in (Float64,BigFloat)

        println()
        println("T is ", T)

        delimit("Tensor sets")

        a = FourierBasis(12)
        b = FourierBasis(13)
        c = FourierBasis(11)
        d = TensorProductSet(a,b,c)

        bf = d[3,4,5]
        x1 = T(2//10)
        x2 = T(3//10)
        x3 = T(4//10)
        @test bf(x1, x2, x3) ≃ call(a, 3, x1) * call(b, 4, x2) * call(c, 5, x3)

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

        delimit("Tensor operators")

        m1 = 3
        n1 = 4
        m2 = 10
        n2 = 24
        A1 = MatrixOperator(map(T,rand(m1,n1)))
        A2 = MatrixOperator(map(T,rand(m2,n2)))
#        A = TensorProductOperator(A1, A2)

        # Apply the tensor-product operator
        b = map(T, rand(n1, n2))
        c = zeros(T, m1, m2)
        @test apply!(A, c, b)

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



        delimit("Fourier series")

        ## Even length
        n = 12
        a = -T(1.2)
        b = T(3.4)
        fb = FourierBasis(n, a, b)

        @test left(fb) ≈ a
        @test right(fb) ≈ b

        @test grid(fb) == PeriodicEquispacedGrid(n, a, b)

        # Take a random point in the domain
        x = T(a+rand()*(b-a))
        y = (x-a)/(b-a)

        # Is the 0-index basis function the constant 1?
        freq = 0
        idx = frequency2idx(fb, freq)
        @test fb(idx, x) ≃ 1

        # Evaluate in a point in the interior
        freq = 3
        idx = frequency2idx(fb, freq)
        @test call(fb, idx, x) ≃ exp(2*T(pi)*1im*freq*y)

        # Evaluate the largest frequency, which is a cosine in this case
        freq = n >> 1
        idx = frequency2idx(fb, freq)
        @test call(fb, idx, x) ≈ cos(2*T(pi)*freq*y)

        # Evaluate an expansion
        coef = T[1.0 2.0 3.0 4.0]
        e = SetExpansion(FourierBasis(4, a, b), coef)
        @test e(x) ≈ coef[1]*1.0 + coef[2]*exp(2*T(pi)*im*y) + coef[3]*cos(4*T(pi)*y) + coef[4]*exp(-2*T(pi)*im*y)

        # Try an extension
        n = 12
        coef = map(T, rand(n))
        b1 = FourierBasis(n, a, b)
        b2 = FourierBasis(n+1, a, b)
        b3 = FourierBasis(n+15, a, b)
        E2 = Extension(b1, b2)
        E3 = Extension(b1, b3)
        e1 = SetExpansion(b1, coef)
        e2 = SetExpansion(b2, E2*coef)
        e3 = SetExpansion(b3, E3*coef)
        x = T(2//10)
        @test e1(x) ≃ e2(x)
        @test e1(x) ≃ e3(x)


        # Does indexing work as intended?
        idx = Int(round(rand()*length(fb)))
        bf = fb[idx]
        x = T(2//10)
        @test bf(x) ≃ call(fb, idx, x)

        # Can you iterate over the whole set?
        z = zero(T)
        x = T(2//10)
        i = 0
        for f in fb
            z = z + f(x)
            i = i+1
        end
        @test i == length(fb)

        l = 0
        for i in eachindex(fb)
            f = fb[i]
            z = z + f(x)
            l = l+1
        end
        @test l == length(fb)

        # Differentiation test
        coef = map(T, rand(Float64, size(fb)))
        D = differentiation_operator(fb)
        coef2 = D*coef
        e1 = SetExpansion(fb, coef)
        e2 = SetExpansion(FourierBasis(length(fb)+1,left(fb),right(fb)), coef2)

        x = T(2//10)
        delta = sqrt(eps(T))
        @test abs( (e1(x+delta)-e1(x))/delta - e2(x) ) / abs(e2(x)) < 100delta



        ## Odd length
        b = FourierBasis(13, -one(T), one(T))

        # Is the 0-index basis function the constant 1?
        freq = 0
        idx = frequency2idx(b, freq)
        @test call(b, idx, T(2//10)) ≃ 1

        # Evaluate in a point in the interior
        freq = 3
        idx = frequency2idx(b, freq)
        x = T(2//10)
        y = (x+1)/2
        @test call(b, idx, x) ≃ exp(2*T(pi)*1im*freq*y)

        # Evaluate an expansion
        coef = [1.0 2.0 3.0]
        e = SetExpansion(FourierBasis(3, -one(T), one(T)), coef)
        x = T(2//10)
        y = (x+1)/2
        @test e(x) ≃ coef[1]*one(T) + coef[2]*exp(2*T(pi)*im*y) + coef[3]*exp(-2*T(pi)*im*y)

        # Try an extension
        n = 13
        coef = map(T, rand(n))
        b1 = FourierBasis(n, -one(T), one(T))
        b2 = FourierBasis(n+1, -one(T), one(T))
        b3 = FourierBasis(n+15, -one(T), one(T))
        E2 = Extension(b1, b2)
        E3 = Extension(b1, b3)
        e1 = SetExpansion(b1, coef)
        e2 = SetExpansion(b2, E2*coef)
        e3 = SetExpansion(b3, E3*coef)
        x = T(2//10)
        @test e1(x) ≃ e2(x)
        @test e1(x) ≃ e3(x)

        # Restriction
        n = 14
        b1 = FourierBasis(n, -one(T), one(T))
        b2 = FourierBasis(n-1, -one(T), one(T))
        b3 = FourierBasis(n-5, -one(T), one(T))
        E1 = Restriction(b1, b2)    # source has even length
        E2 = Restriction(b2, b3)    # source has odd length
        coef1 = map(T, rand(length(b1)))
        coef2 = E1*coef1
        coef3 = E2*coef2
        @test reduce(&, [ coef2[i+1] == coef1[i+1] for i=0:BasisFunctions.nhalf(b2) ] )
        @test reduce(&, [ coef2[end-i+1] == coef1[end-i+1] for i=1:BasisFunctions.nhalf(b2) ] )
        @test reduce(&, [ coef3[i+1] == coef2[i+1] for i=0:BasisFunctions.nhalf(b3) ] )
        @test reduce(&, [ coef3[end-i+1] == coef2[end-i+1] for i=1:BasisFunctions.nhalf(b3) ] )

        # Differentiation test
        coef = map(T, rand(Float64, size(b)))
        D = differentiation_operator(b)
        coef2 = D*coef
        e1 = SetExpansion(b, coef)
        e2 = SetExpansion(b, coef2)

        x = T(2//10)
        delta = sqrt(eps(T))
        @test abs( (e1(x+delta)-e1(x))/delta - e2(x) ) / abs(e2(x)) < 150delta


        delimit("Grids and tensor grids")

        ## Equispaced grids
        len = 120
        a = -T(1.2)
        b = T(3.5)
        g = EquispacedGrid(len, a, b)

        idx = 5
        @test g[idx] ≃ a + (idx-1) * (b-a)/(len-1)
        @test g[len] ≃ b
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
        for x in eachelement(g)
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
        @test g[idx] ≃ a + (idx-1) * (b-a)/len
        @test g[len] ≃ b - stepsize(g)
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
        for x in eachelement(g)
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
        @test x[1] ≃ x1
        @test x[2] ≃ x2

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
        for x in eachelement(g)
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
        x = g3[idx1,idx2]
        x1 = g[idx1]
        x2 = g2[idx2]
        @test x[1] ≃ x1[1]
        @test x[2] ≃ x1[2]
        @test x[3] ≃ x2

    end # for T in...

end # Test.with_handler


# Diagnostics
println("Succes rate:\t$successes/$(successes+failures+errors)")
println("Failure rate:\t$failures/$(successes+failures+errors)")
println("Error rate:\t$errors/$(successes+failures+errors)")

end # module

