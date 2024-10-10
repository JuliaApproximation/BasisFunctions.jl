
const JACOBI_OR_LEGENDRE = Union{Legendre,AbstractJacobi}

isequaldict(b1::JACOBI_OR_LEGENDRE, b2::JACOBI_OR_LEGENDRE) =
    length(b1)==length(b2) &&
    jacobi_α(b1) == jacobi_α(b2) &&
    jacobi_β(b2) == jacobi_β(b2)

# TODO: implement conversions

