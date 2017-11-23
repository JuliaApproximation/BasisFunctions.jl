
# Interpolate linearly between the left and right endpoint, using the value 0 <= scalar <= 1
function point_in_domain(basis::FunctionSet, scalar)
    # Try to find an interval within the support of the basis
    a = left(basis)
    b = right(basis)

    if isinf(norm(a)) || isinf(norm(b))
        va = MVector(a)
        vb = MVector(b)
        # Avoid infinities for some bases on the real line
        for i in 1:length(va)
            if isinf(va[i])
                va[i] = -1
            end
            if isinf(vb[i])
                vb[i] = 1
            end
        end
        a = SVector(va)
        b = SVector(vb)
    end
    x = (1-scalar) * a + scalar * b
end
Base.rationalize{N}(x::SVector{N,Float64}) = SVector{N,Rational{Int}}([rationalize(x_i) for x_i in x])
Base.rationalize{N}(x::SVector{N,BigFloat}) = SVector{N,Rational{BigInt}}([rationalize(x_i) for x_i in x])


function delimit(s::AbstractString)
    println("############")
    println("# ",s)
    println("############")
end


# Interpolate linearly between the left and right endpoint, using the value 0 <= scalar <= 1
function point_in_domain(basis::FunctionSet1d, scalar)
    # Try to find an interval within the support of the basis
    a = left(basis)
    b = right(basis)

    T = domaintype(basis)
    # Avoid infinities for some bases on the real line
    if isinf(a)
        a = -T(1)
    end
    if isinf(b)
        b = T(1)
    end
    x = (1-scalar) * a + scalar * b
end




# Abuse point_in_domain with a scalar greater than one in order to get
# a point outside the domain.
point_outside_domain(basis::FunctionSet) = point_in_domain(basis, convert(float_type(domaintype(basis)), 1.1))

point_outside_domain(basis::LaguerrePolynomials) = -one(domaintype(basis))
point_outside_domain(basis::HermitePolynomials) = one(domaintype(basis))+im

function random_point_in_domain(basis::FunctionSet)
    T = float_type(domaintype(basis))
    w = one(T) * rand()
    point_in_domain(basis, w)
end

function fixed_point_in_domain(basis::FunctionSet)
    T = float_type(domaintype(basis))
    w = 1/sqrt(T(2))
    point_in_domain(basis, w)
end

random_index(basis::FunctionSet) = 1 + Int(floor(rand()*length(basis)))
