function test_generic_operator_interface(op, T)
    ELT = eltype(op)
    @test promote_type(T,ELT) == ELT
    #@test coefficienttype(dest(op)) == ELT
    # This may no longer always be true:
    # @test eltype(src(op)) == ELT

    m = matrix(op)

    # Test elementwise equality
    for i in 1:size(op,1)
        for j in 1:size(op,2)
            @test m[i,j] == op[i,j]
        end
    end

    # Does the operator agree with its matrix?
    r = zeros(src(op))
    for i in eachindex(r)
        r[i] = convert(ELT, rand())
    end
    v1 = zeros(dest(op))
    apply!(op, v1, r)
    v2 = m*r
    @test maximum(abs.(v1-v2)) < 10*sqrt(eps(T))

    # Verify claim to be in-place
    if is_inplace(op)
        v3 = copy(r)
        apply_inplace!(op, v3)
        @test maximum(abs.(v3-v2)) < 10*sqrt(eps(T))
    end

    # Verify that coef_src is not altered when applying out-of-place
    r2 = copy(r)
    v = zeros(dest(op))
    apply!(op, v, r)
    @test maximum(abs.(r-r2)) < eps(T)

    # Test claim to be diagonal
    if is_diagonal(op)
        for i in 1:size(op,1)
            for j in 1:size(op,2)
                if i != j
                    @test abs(m[i,j]) < 10*sqrt(eps(T))
                end
            end
        end
    end

    # Verify diagonal entries
    d = diagonal(op)
    for i in 1:min(size(op,1),size(op,2))
        d[i] ≈ m[i,i]
        diagonal(op, i) ≈ m[i,i]
    end

    ## # Verify eltype promotion
    ## T2 = widen(eltype(op))
    ## if T2 != eltype(op)
    ##     op2 = promote_eltype(op, T2)
    ##     ELT2 = eltype(op2)
    ##     # ELT2 may not equal T, but it must be wider.
    ##     # For example, when T2 is BigFloat, ELT2 could be Complex{BigFloat}
    ##     @test promote_type(T2, ELT2) == ELT2
    ##     @test coefficienttype(dest(op2)) == ELT2
    ## end

    # Verify inverse
    try
        inv_op = inv(op)
        if ~(typeof(inv_op) <: OperatorInverse)
            @test src(inv_op) == dest(op)
            @test dest(inv_op) == src(op)
            m2 = matrix(inv_op)
            I1 = eye(ELT, length(src(op)))
            I2 = eye(ELT, length(dest(op)))
            @test maximum(abs.(m2*m - I1)) < 10*sqrt(eps(T))
            @test maximum(abs.(m*m2 - I2)) < 10*sqrt(eps(T))
        end
    catch MethodError
        # Inverse was not defined
    end

    # Verify transpose
    try
        ct_op = adjoint(opt)
        if ~(typeof(ct_op) <: OperatorTranspose)
            @test src(ct_op) == dest(op)
            @test dest(ct_op) == src(op)
            m2 = matrix(ct_op)
            @test maximum(abs.(m' - m2)) < 10*sqrt(eps(T))
        end
    catch MethodError
        # transpose was not defined
    end
end
