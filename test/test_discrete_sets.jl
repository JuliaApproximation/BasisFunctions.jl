# test_discrete_sets.jl

function test_discrete_sets(T)
    n = 10
    s1 = DiscreteSet{T}(n)

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

    @test domain(s1) == ClosedInterval{Int}(1, length(s1))

    # Evaluation
    @test eval_element(s1, 1, 1) == one(T)
    @test eval_element(s1, 1, 2) == zero(T)
    @test eval_element(s1, n, n) == one(T)
end
