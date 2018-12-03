#####
# Tensor sets
#####

function test_tensor_sets(T)
    a = FourierBasis(12)
    b = FourierBasis(13)
    c = FourierBasis(11)
    d = TensorProductDict(a,b,c)

    bf = d[3,4,5]
    x1 = T(2//10)
    x2 = T(3//10)
    x3 = T(4//10)
    @test bf(x1, x2, x3) ≈ eval_element(a, 3, x1) * eval_element(b, 4, x2) * eval_element(c, 5, x3)

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
    #     d[CartesianIndices(CartesianIndex(1,1,1),CartesianIndex(3,4,5))]
    #     true
    # catch
    #     false
    # end
    # Is an error thrown if you index with a range that is out of range?
end
