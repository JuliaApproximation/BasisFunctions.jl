using BasisFunctions, BasisFunctions.Test

using Test

types = [Float64,BigFloat,]

function test_operators(T)
    @testset "$(rpad("test identity operator",80))" begin
        test_identity_operator(T) end

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

    @testset "$(rpad("test banded operators",80))" begin
        test_banded_operator(T) end

    @testset "$(rpad("test sparse operator",80))" begin
        test_sparse_operator(T) end
end

function test_generic_operators(T)
    b1 = FourierBasis{T}(3)
    b2 = ChebyshevBasis{T}(4)
    b3 = ChebyshevBasis{T}(3)
    b4 = LegendrePolynomials{T}(3)

    operators = [
        ["Scaling operator", ScalingOperator(b1, b1, T(2))],
        ["Zero operator", BF.ZeroOperator(b1, b2)],
        ["Diagonal operator", DiagonalOperator(b2, b2, rand(T, length(b2)))],
        ["Coefficient scaling operator", BF.CoefficientScalingOperator(b1, 1, T(2))],
        ["Wrapped operator", WrappedOperator(b3, b3, ScalingOperator(b4, b4, T(2))) ],
        ["Index restriction operator", IndexRestrictionOperator(b2, b1, 1:3) ],
        ["Derived operator", ConcreteDerivedOperator(DiagonalOperator(b2, b2, rand(T, length(b2))))],
    ]

    for ops in operators
        @testset "$(rpad("$(ops[1])", 80))" begin
            test_generic_operator_interface(ops[2], T)
        end
    end
end

function test_tensor_operators(T)
    m1 = 3
    n1 = 4
    m2 = 10
    n2 = 24
    A1 = MatrixOperator(rand(T,m1,n1))
    A2 = MatrixOperator(rand(T,m2,n2))

    A_tp = TensorProductOperator(A1, A2)
    b = rand(T, n1, n2)
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
    for SRC in (FourierBasis{T}(10), ChebyshevBasis{T}(11))
        operators = (BF.CoefficientScalingOperator(SRC, 3, rand(coefficienttype(SRC))),
            BF.AlternatingSignOperator(SRC), IdentityOperator(SRC),
            ScalingOperator(SRC, rand(coefficienttype(SRC))), ScalingOperator(SRC,3),
            DiagonalOperator(SRC, rand(coefficienttype(SRC), size(SRC))))
           # PseudoDiagonalOperator(SRC, map(coefficienttype(SRC), rand(size(SRC)))))
        for Op in operators
            m = matrix(Op)
            # Test in-place
            if isinplace(Op)
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
            @test isdiagonal(2*Op)
            @test isdiagonal(Op')
            @test isdiagonal(inv(Op))
            @test isdiagonal(Op*Op)
            # Make sure in_place is retained
            if isinplace(Op)
                @test isinplace(2*Op)
                @test isinplace(Op')
                @test isinplace(inv(Op))
                @test isinplace(Op*Op)
            end
        end
    end
end

function test_multidiagonal_operators(T)
    MSet = FourierBasis{T}(10)⊕ChebyshevBasis{T}(11)
    operators = (BF.CoefficientScalingOperator(MSet, 3, rand(coefficienttype(MSet))),
        BF.AlternatingSignOperator(MSet), IdentityOperator(MSet),
        ScalingOperator(MSet,2.0+2.0im), ScalingOperator(MSet, 3),
        DiagonalOperator(MSet, rand(coefficienttype(MSet),length(MSet))))
    for Op in operators
        # Test in-place
        coef_src = zeros(MSet)
        for i in eachindex(coef_src)
            coef_src[i] = rand(eltype(coef_src))
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
        @test isdiagonal(2*Op) && isinplace(2*Op)
        @test isdiagonal(Op') && isinplace(Op')
        @test isdiagonal(inv(Op)) && isinplace(inv(Op))
        @test isdiagonal(Op*Op) && isinplace(Op*Op)
    end
end

function test_sparse_operator(ELT)
    S = SparseOperator(MatrixOperator(rand(ELT,4,4)))
    test_generic_operator_interface(S, ELT)
end
function test_banded_operator(ELT)
    a = [ELT(1),ELT(2),ELT(3)]
    H = HorizontalBandedOperator(FourierBasis{ELT}(6,0,1), FourierBasis{ELT}(3,0,1),a,3,2)
    test_generic_operator_interface(H, ELT)
    h = matrix(H)
    e = zeros(ELT,6,1)
    for i in 1:3
        e .= 0
        I = (2+(i-1)*3) .+ (1:3)
        for (j,k) in enumerate(mod.(I .- 1,6) .+ 1)
            e[k] = a[j]
        end
        @test e ≈ h[i,:]
    end

    V = VerticalBandedOperator(FourierBasis{ELT}(3,0,1), FourierBasis{ELT}(6,0,1),a,3,2)
    test_generic_operator_interface(V, ELT)
    v = matrix(V)
    e = zeros(ELT,6,1)
    for i in 1:3
        e .= 0
        I = (2+(i-1)*3) .+ (1:3)
        for (j,k) in enumerate(mod.(I .- 1,6) .+ 1)
            e[k] = a[j]
        end
        @test e ≈ v[:,i]
    end
end

function test_circulant_operator(ELT)
    n = 20
    for T in (ELT, complex(ELT))
        e1 = zeros(T,n); e1[1] = 1
        c = rand(T, n)
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

function test_identity_operator(T)
    b = FourierBasis{T}(10)
    I = IdentityOperator(b)
    @test I isa IdentityOperator
    @test eltype(I) == Complex{T}
    @test src(I) == b
    I = IdentityOperator(FourierBasis{T}(10), ChebyshevBasis{T}(10))
    @test src(I) == FourierBasis{T}(10)
    @test dest(I) == ChebyshevBasis{T}(10)
end

function test_invertible_operators(T)
    for SRC in (FourierBasis{T}(10), ChebyshevBasis{T}(10))
        operators = (MultiplicationOperator(SRC, SRC, rand(coefficienttype(SRC),length(SRC),length(SRC))),
            CirculantOperator(rand(T, 10)),
            CirculantOperator(rand(complex(T), 10)))
        for Op in operators
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
    for SRC in (FourierBasis{T}(10), ChebyshevBasis{T}(11))
        operators = (MultiplicationOperator(SRC, resize(SRC,length(SRC)+2), rand(coefficienttype(SRC),length(SRC)+2,length(SRC))),
            BF.ZeroOperator(SRC))
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

for T in types
    delimit(string(T))
    test_operators(T)
    test_generic_operators(T)
end
