# test_suite.jl

module test_suite

using BasisFunctions
using Base.Test

########
# Auxiliary functions
########

# Keep track of successes, failures and errors
global failures=0
global successes=0
global errors=0

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
≈(a,b) = ≈(promote(a,b)...)
≈{T <: FloatingPoint}(a::T,b::T) = (abs(a-b) < 10eps(T))
≈{T <: FloatingPoint}(a::Complex{T},b::Complex{T}) = (abs(a-b) < 10eps(T))


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


        delimit("Fourier series")

        ## Even length
        b = FourierBasis(12, -one(T), one(T))

        # Is the 0-index basis function the constant 1?
        freq = 0
        idx = frequency2idx(b, freq)
        @test b(idx, T(2//10)) ≈ 1

        # Evaluate in a point in the interior
        freq = 3
        idx = frequency2idx(b, freq)
        x = T(2//10)
        y = (x+1)/2
        @test call(b, idx, x) ≈ exp(2*T(pi)*1im*freq*y)

        # Evaluate the largest frequency, which is a cosine in this case
        freq = 6
        idx = frequency2idx(b, freq)
        x = T(2//10)
        y = (x+1)/2
        @test call(b, idx, x) ≈ cos(2*T(pi)*freq*y)

        # Evaluate an expansion
        coef = T[1.0 2.0 3.0 4.0]
        e = SetExpansion(FourierBasis(4, -one(T), one(T)), coef)
        x = T(2//10)
        y = (x+1)/2
        @test e(x) ≈ coef[1]*1.0 + coef[2]*exp(2*T(pi)*im*y) + coef[3]*cos(4*T(pi)*y) + coef[4]*exp(-2*T(pi)*im*y)

        # Try an extension
        n = 12
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
        @test e1(x) ≈ e2(x)
        @test e1(x) ≈ e3(x)

        # Does indexing work as intended?
        idx = Int(round(rand()*length(b)))
        bf = b[idx]
        x = T(2//10)
        @test bf(x) ≈ call(b, idx, x)

        # Can you iterate over the whole set?
        z = zero(T)
        x = T(2//10)
        i = 0
        for f in b
            z = z + f(x)
            i = i+1
        end
        @test i == length(b)

        l = 0
        for i in eachindex(b)
            f = b[i]
            z = z + f(x)
            l = l+1
        end
        @test l == length(b)

        # Differentiation test
        coef = map(T, rand(Float64, size(b)))
        D = differentiation_operator(b)
        coef2 = D*coef
        e1 = SetExpansion(b, coef)
        e2 = SetExpansion(FourierBasis(length(b)+1,left(b),right(b)), coef2)

        x = T(2//10)
        delta = sqrt(eps(T))
        @test abs( (e1(x+delta)-e1(x))/delta - e2(x) ) / abs(e2(x)) < 100delta



        ## Odd length
        b = FourierBasis(13, -one(T), one(T))

        # Is the 0-index basis function the constant 1?
        freq = 0
        idx = frequency2idx(b, freq)
        @test call(b, idx, T(2//10)) ≈ 1

        # Evaluate in a point in the interior
        freq = 3
        idx = frequency2idx(b, freq)
        x = T(2//10)
        y = (x+1)/2
        @test call(b, idx, x) ≈ exp(2*T(pi)*1im*freq*y)

        # Evaluate an expansion
        coef = [1.0 2.0 3.0]
        e = SetExpansion(FourierBasis(3, -one(T), one(T)), coef)
        x = T(2//10)
        y = (x+1)/2
        @test e(x) ≈ coef[1]*one(T) + coef[2]*exp(2*T(pi)*im*y) + coef[3]*exp(-2*T(pi)*im*y)

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
        @test e1(x) ≈ e2(x)
        @test e1(x) ≈ e3(x)

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

    end # for T in...

end # Test.with_handler


# Diagnostics
println("Succes rate:\t$successes/$(successes+failures+errors)")
println("Failure rate:\t$failures/$(successes+failures+errors)")
println("Error rate:\t$errors/$(successes+failures+errors)")

end # module

