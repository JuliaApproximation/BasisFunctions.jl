

function test_operators(T)
    @testset "$(rpad("test diagonal operators",80))" begin
        test_diagonal_operators(T) end

    @testset "$(rpad("test multidiagonal operators",80))" begin
        test_multidiagonal_operators(T) end

    @testset "$(rpad("test invertible operators",80))" begin
        test_invertible_operators(T) end

    @testset "$(rpad("test noninvertible operators",80))" begin
        test_noninvertible_operators(T) end

    @testset "$(rpad("test tensor operators",80))" begin
        test_tensor_operators(T) end

    @testset "$(rpad("test circulant operator",80))" begin
        test_circulant_operator(T) end
end

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

    @test sum(abs.(c_tp-c_manual)) + 1 ≈ 1
    @test sum(abs.(c_dim-c_manual)) + 1 ≈ 1
    @test sum(abs.(c_tp-c_dim)) + 1 ≈ 1
end

function test_diagonal_operators(T)
    for SRC in (Span(FourierBasis{T}(10)), Span(ChebyshevBasis{T}(11)))
        operators = (CoefficientScalingOperator(SRC, 3, map(coeftype(SRC),rand())),
            UnevenSignFlipOperator(SRC), IdentityOperator(SRC),
            ScalingOperator(SRC, map(coeftype(SRC),rand())), ScalingOperator(SRC,3),
            DiagonalOperator(SRC, map(coeftype(SRC), rand(size(SRC)))))
           # PseudoDiagonalOperator(SRC, map(coeftype(SRC), rand(size(SRC)))))
        for Op in operators
            m = matrix(Op)
            # Test in-place
            if is_inplace(Op)
                coef_src = rand(src(Op))
                coef_dest_m = m * coef_src
                coef_dest = apply!(Op, coef_src)
                @test sum(abs.(coef_dest-coef_dest_m)) + 1 ≈ 1
            end
            # Test out-of-place
            coef_src = rand(src(Op))
            coef_dest = apply(Op, coef_src)
            coef_dest_m = m * coef_src
            @test sum(abs.(coef_dest-coef_dest_m)) + 1 ≈ 1
            # Test inverse
            I = inv(Op)
            @test abs(sum(abs.(I*(Op*coef_src)-coef_src))) + 1 ≈ 1
            @test abs(sum(abs.((I*Op)*coef_src-coef_src))) + 1 ≈ 1
            # Test transposes
            @test abs(sum(abs.(m'*coef_src-Op'*coef_src))) + 1 ≈ 1
            # Test Sum
            @test abs(sum(abs.((Op+Op)*coef_src-2*(Op*coef_src)))) + 1 ≈ 1
            # Test Equivalence to diagonal operator
            @test abs(sum(abs.(Op*coef_src-diagonal(Op).*coef_src))) + 1 ≈ 1
            # Make sure diagonality is retained
            @test is_diagonal(2*Op)
            @test is_diagonal(Op')
            @test is_diagonal(inv(Op))
            @test is_diagonal(Op*Op)
            # Make sure in_place is retained
            if is_inplace(Op)
                @test is_inplace(2*Op)
                @test is_inplace(Op')
                @test is_inplace(inv(Op))
                @test is_inplace(Op*Op)
            end
        end
    end
end

function test_multidiagonal_operators(T)
    MSet = Span(FourierBasis{T}(10)⊕ChebyshevBasis{T}(11))
    operators = (CoefficientScalingOperator(MSet, 3, rand()*one(coeftype(MSet))),
        UnevenSignFlipOperator(MSet), IdentityOperator(MSet),
        ScalingOperator(MSet,2.0+2.0im), ScalingOperator(MSet, 3),
        DiagonalOperator(MSet, map(coeftype(MSet), rand(length(MSet)))))
    for Op in operators
        # Test in-place
        coef_src = zeros(MSet)
        for i in eachindex(coef_src)
            coef_src[i] = rand() * one(eltype(coef_src))
        end
        m = matrix(Op)
        coef_dest_m = m * linearize_coefficients(MSet,coef_src)
        coef_dest = linearize_coefficients(MSet, apply!(Op, coef_src))
        @test sum(abs.(coef_dest-coef_dest_m)) + 1 ≈ 1
        # Test out-of-place
        coef_src = zeros(MSet)
        for i in eachindex(coef_src)
            coef_src[i] = rand() * one(eltype(coef_src))
        end
        coef_dest = linearize_coefficients(MSet, apply(Op, coef_src))
        coef_dest_m = m * linearize_coefficients(MSet, coef_src)
        @test sum(abs.(coef_dest-coef_dest_m)) + 1 ≈ 1

        # Test inverse
        I = inv(Op)
        @test I*(Op*coef_src)≈coef_src
        @test (I*Op)*coef_src≈coef_src
        # Test transposes
        @test m'*linearize_coefficients(MSet,coef_src)≈linearize_coefficients(MSet,Op'*coef_src)
        # Test Sum
        @test (Op+Op)*coef_src≈2*(Op*coef_src)
        # Test Equivalence to diagonal operator
        @test linearize_coefficients(MSet,Op*coef_src)≈diagonal(Op).*linearize_coefficients(MSet,coef_src)
        # Make sure diagonality is retained
        @test is_diagonal(2*Op) && is_inplace(2*Op)
        @test is_diagonal(Op') && is_inplace(Op')
        @test is_diagonal(inv(Op)) && is_inplace(inv(Op))
        @test is_diagonal(Op*Op) && is_inplace(Op*Op)
    end
end

# function test_complexify_operator(T)
#   for SRC in (BSplineTranslatesBasis(11, 2, T), BSplineTranslatesBasis(11, 2, complex(T)))
#     op = ComplexifyOperator(SRC)
#     DEST = dest(op)
#     ELT = eltype(op)
#     @test ELT == eltype(src(op))
#     @test eltype(dest(op)) == complex(T)
#     coef_src = zeros(eltype(SRC),SRC)
#     coef_dest = zeros(eltype(DEST),DEST)
#     for i in eachindex(coef_src)
#         coef_src[i]=map(eltype(src(op)),rand())
#     end
#     apply!(op, coef_dest, coef_src)
#     @test norm(coef_dest-coef_src) == 0
#
#     op = RealifyOperator(SRC)
#     ELT = eltype(op)
#     @test ELT == eltype(dest(op))
#     @test eltype(dest(op)) == T
#     coef_src = zeros(eltype(SRC),SRC)
#     coef_dest = zeros(eltype(DEST),DEST)
#     for i in eachindex(coef_src)
#         coef_src[i]=map(eltype(src(op)),rand())
#     end
#     apply!(op, coef_dest, coef_src)
#     @test norm(coef_dest-coef_src) == 0
#   end
# end

function test_circulant_operator(ELT)
    n = 20
    for T in (ELT, complex(ELT))
        e1 = zeros(T,n); e1[1] = 1
        c = map(T, rand(n))
        C = CirculantOperator(c)
        m = matrix(C)
        for i in 1:n
            @test m[:,i] ≈ circshift(c,i-1)
        end
        invC = inv(C)
        @test BasisFunctions.eigenvalues(invC).*BasisFunctions.eigenvalues(C) ≈ ones(T,n)
        @test m'*e1 ≈ C'*e1

        sumC = invC+C
        @test (typeof(sumC) <:BasisFunctions.CirculantOperator)
        @test sumC*e1 ≈ m[:,1] + matrix(invC)[:,1]

        subC = invC-C
        @test (typeof(subC) <:BasisFunctions.CirculantOperator)
        @test subC*e1 ≈ -m[:,1] + matrix(invC)[:,1]

        mulC = invC*C
        @test (typeof(sumC) <:BasisFunctions.CirculantOperator)
        @test mulC*e1 ≈ (m*matrix(invC))[:,1]

        mulC = 2*C
        @test (typeof(sumC) <:BasisFunctions.CirculantOperator)
        @test mulC*e1 ≈ 2*C*e1

        mulC = C*2
        @test (typeof(sumC) <:BasisFunctions.CirculantOperator)
        @test mulC*e1 ≈ 2*C*e1
    end
end

function test_invertible_operators(T)
    for SRC in (Span(FourierBasis{T}(10)), Span(ChebyshevBasis{T}(11), Complex{T}), Span(ChebyshevBasis{T}(10)))
        operators = (MultiplicationOperator(SRC, SRC, map(coeftype(SRC),rand(length(SRC),length(SRC)))),
            CirculantOperator(map(T,rand(10))),
            CirculantOperator(map(complex(T),rand(10))))
        for Op in operators
            show(Op)
            m = matrix(Op)
            # Test out-of-place
            coef_src = rand(src(Op))
            coef_dest = apply(Op, coef_src)
            coef_dest_m = m * coef_src
            @test sum(abs.(coef_dest-coef_dest_m)) + 1 ≈ 1
            # Test inverse
            I = inv(Op)
            @test abs(sum(abs.(I*(Op*coef_src)-coef_src))) + 1 ≈ 1
            @test abs(sum(abs.((I*Op)*coef_src-coef_src))) + 1 ≈ 1
            # Test transposes
            @test abs(sum(abs.(m'*coef_src-Op'*coef_src))) + 1 ≈ 1
            # Test Sum
            @test abs(sum(abs.((Op+Op)*coef_src-2*(Op*coef_src)))) + 1 ≈ 1
        end
    end
end

# FunctionOperator is currently not tested, since we cannot assume it is a linear operator.
function test_noninvertible_operators(T)
    for SRC in (Span(FourierBasis{T}(10)), Span(ChebyshevBasis{T}(11), Complex{T}))
        operators = (MultiplicationOperator(SRC, resize(SRC,length(SRC)+2), map(coeftype(SRC),rand(length(SRC)+2,length(SRC)))),
            ZeroOperator(SRC))
        for Op in operators
            m = matrix(Op)
            # Test out-of-place
            coef_src = rand(src(Op))
            coef_dest = apply(Op, coef_src)
            coef_dest_m = m * coef_src
            @test sum(abs.(coef_dest-coef_dest_m)) + 1 ≈ 1
            # Test transposes
            @test (sum(abs.(m'-matrix(Op')))) + 1 ≈ 1
            # Test Sum
            @test (sum(abs.((Op+Op)*coef_src-2*(Op*coef_src)))) + 1 ≈ 1
        end
    end
end
