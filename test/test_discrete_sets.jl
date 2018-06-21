# test_discrete_sets.jl

function test_discrete_sets(T)
    n = 10
    s1 = DiscreteVectorDictionary{T}(n)
    @test domaintype(s1) == Int
    @test codomaintype(s1) == T
    @test length(s1) == n
    @test is_discrete(s1)

    s2 = resize(s1, 2*n)
    @test domaintype(s2) == Int
    @test codomaintype(s2) == T
    @test length(s2) == 2*n

    # Support
    @test in_support(s1, 1, 4)
    @test !in_support(s1, 1, 0)
    @test in_support(s1, 1, length(s1))
    @test !in_support(s1, 1, length(s1)+1)
    @test !in_support(s1, 1, 2.5)

    @test support(s1) == ClosedInterval{Int}(1, length(s1))

    # Evaluation
    @test eval_element(s1, 1, 1) == one(T)
    @test eval_element(s1, 1, 2) == zero(T)
    @test eval_element(s1, n, n) == one(T)

    t1 = DiscreteArrayDictionary((10,10), T)
    @test codomaintype(t1) == T
    @test domaintype(t1) == ProductIndex{2}
    @test size(t1) == (10,10)

    @test in_support(t1, (1,1), 5)
    @test !in_support(t1, (1,1), 101)
    @test in_support(t1, (1,1), (3,3))
    @test !in_support(t1, (1,1), (11,3))
    @test in_support(t1, (1,1), CartesianIndex(5,3))
    @test !in_support(t1, (1,1), CartesianIndex(5,11))

    t2 = resize(t1, (20,20))
    @test codomaintype(t2) == T
    @test domaintype(t2) == ProductIndex{2}
    @test size(t2) == (20,20)

    # Evaluation
    @test eval_element(t1, 1, 1) == one(T)
    @test eval_element(t1, 1, (1,1)) == one(T)
    @test eval_element(t1, 1, CartesianIndex(1,1)) == one(T)
    @test eval_element(t1, (1,1), 1) == one(T)
    @test eval_element(t1, CartesianIndex(1,1), 1) == one(T)
    @test eval_element(t1, CartesianIndex(1,1), (1,1)) == one(T)

    @test eval_element(t1, 2, 5) == zero(T)
    @test eval_element(t1, (2,1), 5) == zero(T)
    @test eval_element(t1, (2,1), CartesianIndex(3,3)) == zero(T)

end
