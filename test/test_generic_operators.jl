# test_generic_operators

#####
# Tensor operators
#####

function test_generic_operators(T)
    b1 = span(FourierBasis(3, T))
    b2 = span(ChebyshevBasis(4, T))
    b3 = span(ChebyshevBasis(3, T))

    operators = [
        ["Identity operator", IdentityOperator(b1, b1)],
        ["Scaling operator", ScalingOperator(b1, b1, T(2))],
        ["Zero operator", ZeroOperator(b1, b2)],
        ["Diagonal operator", DiagonalOperator(b2, b2, map(T, rand(length(b2))))],
        ["Coefficient scaling operator", CoefficientScalingOperator(b1, b1, 1, T(2))],
        ["Wrapped operator", WrappedOperator(b3, b3, ScalingOperator(b1, b1, T(2))) ],
        ["Index restriction operator", IndexRestrictionOperator(b2, b1, 1:3) ],
        ["Derived operator", ConcreteDerivedOperator(DiagonalOperator(b2, b2, map(T, rand(length(b2)))))],
        ["Pseudo diagonal operator", PseudoDiagonalOperator(b2, map(T, rand(length(b2))))],
        ["Circulant operator", CirculantOperator(b2, map(T, rand(length(b2))))],
    ]

    for ops in operators
        @testset "$(rpad("$(ops[1])", 80))" begin
            test_generic_operator_interface(ops[2], T)
        end
    end
end

function test_generic_operator_interface(op, T)
    ELT = eltype(op)
    @test promote_type(T,ELT) == ELT
    @test coeftype(dest(op)) == ELT
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

    # Verify eltype promotion
    T2 = widen(T)
    if T2 != T
        op2 = promote_eltype(op, T2)
        ELT2 = eltype(op2)
        # ELT2 may not equal T, but it must be wider.
        # For example, when T2 is BigFloat, ELT2 could be Complex{BigFloat}
        @test promote_type(T2, ELT2) == ELT2
        @test coeftype(dest(op2)) == ELT2
    end

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
        ct_op = ctranspose(op)
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
    for SRC in (span(FourierBasis(10, T)), span(ChebyshevBasis(11, T)))
        operators = (CoefficientScalingOperator(SRC, 3, map(coeftype(SRC),rand())),
            UnevenSignFlipOperator(SRC), IdentityOperator(SRC),
            ScalingOperator(SRC, map(coeftype(SRC),rand())), ScalingOperator(SRC,3),
            DiagonalOperator(SRC, map(coeftype(SRC), rand(size(SRC)))),
            PseudoDiagonalOperator(SRC, map(coeftype(SRC), rand(size(SRC)))))
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
    MSet = span(FourierBasis(10, T)⊕ChebyshevBasis(11, T))
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
    for SRC in (span(FourierBasis(10, T)), span(ChebyshevBasis(11, Complex{T})), span(ChebyshevBasis(10,T)))
        operators = (MultiplicationOperator(SRC, SRC, map(coeftype(SRC),rand(length(SRC),length(SRC)))),
            CirculantOperator(map(T,rand(10))),
            CirculantOperator(map(complex(T),rand(10))))
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
    for SRC in (span(FourierBasis(10, T)), span(ChebyshevBasis(11, Complex{T})))
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
