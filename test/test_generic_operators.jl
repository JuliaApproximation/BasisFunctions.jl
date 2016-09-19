# test_generic_operators

#####
# Tensor operators
#####

function test_tensor_operators(T)
    m1 = 3
    n1 = 4
    m2 = 10
    n2 = 24
    A1 = MatrixOperator(map(T,rand(m1,n1)))
    A2 = MatrixOperator(map(T,rand(m2,n2)))

    A_tp = TensorProductOperator(A1, A2)
    b = map(T, rand(n1, n2))
    c_tp = zeros(T, m1, m2)
    apply!(A_tp, c_tp, b)

    d1 = dimension_operator(src(A1) ⊗ src(A2), dest(A1) ⊗ src(A2), A1, 1)
    d2 = dimension_operator(dest(A1) ⊗ src(A2), dest(A1) ⊗ dest(A2), A2, 2)
    A_dim = compose(d1, d2)
    c_dim = zeros(T, m1, m2)
    apply!(A_dim, c_dim, b)

    # Apply the operators manually, row by row and then column by column
    intermediate = zeros(T, m1, n2)
    result1 = zeros(T, m1)
    for i = 1:n2
        apply!(A1, result1, b[:,i])
        intermediate[:,i] = result1
    end
    c_manual = similar(c_tp)
    result2 = zeros(T, m2)
    for i = 1:m1
        apply!(A2, result2, reshape(intermediate[i,:], n2))
        c_manual[i,:] = result2
    end

    @test sum(abs(c_tp-c_manual)) + 1 ≈ 1
    @test sum(abs(c_dim-c_manual)) + 1 ≈ 1
    @test sum(abs(c_tp-c_dim)) + 1 ≈ 1
end



function test_diagonal_operators(T)
    for SRC in (FourierBasis(10),ChebyshevBasis(11, Complex{T}))
        operators = (CoefficientScalingOperator(SRC,3,2.0+0.3im),CoefficientScalingOperator(SRC,3,2),UnevenSignFlipOperator(SRC),IdentityOperator(SRC),ScalingOperator(SRC,2.0+2.0im),ScalingOperator(SRC,3),DiagonalOperator(rand(eltype(SRC),size(SRC))))
        for Op in operators
            # Test in-place
            coef_src = rand(eltype(src(Op)),size(src(Op)))
            m = matrix(Op)
            coef_dest_m = m * coef_src
            coef_dest = apply!(Op, coef_src)            
            @test sum(abs(coef_dest-coef_dest_m)) + 1 ≈ 1
            # Test out-of-place
            coef_src = rand(eltype(src(Op)),size(src(Op)))
            coef_dest = apply(Op, coef_src)
            coef_dest_m = m * coef_src
            @test sum(abs(coef_dest-coef_dest_m)) + 1 ≈ 1
            # Test inverse
            I = inv(Op)
            @test (sum(abs(I*(Op*coef_src)-coef_src))) + 1 ≈ 1
            @test (sum(abs((I*Op)*coef_src-coef_src))) + 1 ≈ 1
            # Test transposes
            @test (sum(abs(m'*coef_src-Op'*coef_src))) + 1 ≈ 1
            # Test Sum
            @test (sum(abs((Op+Op)*coef_src-2*(Op*coef_src)))) + 1 ≈ 1
            # Test Equivalence to diagonal operator
            @test (sum(abs(Op*coef_src-diagonal(Op).*coef_src))) + 1 ≈ 1
            # Make sure diagonality is retained
            @test is_diagonal(2*Op) && is_inplace(2*Op)
            @test is_diagonal(Op') && is_inplace(Op')
            @test is_diagonal(inv(Op)) && is_inplace(inv(Op))
            @test is_diagonal(Op*Op) && is_inplace(Op*Op)
        end
    end
end




        
