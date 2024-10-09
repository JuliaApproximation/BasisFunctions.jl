
@deprecate sparse_matrix sparse

@deprecate eval_expansion(dict::Dictionary, coefficients, grid::AbstractGrid; options...) eval_expansion.(Ref(dict), Ref(coefficients), grid; options...)

@deprecate roots(dict::Dictionary, coef::AbstractVector) expansion_roots(dict, coef)

@deprecate innerproduct(Φ1::Dictionary, i, Φ2::Dictionary, j; options...) dict_innerproduct(Φ1, i, Φ2, j; options...)
@deprecate innerproduct(Φ1::Dictionary, i, Φ2::Dictionary, j, measure; options...) dict_innerproduct(Φ1, i, Φ2, j, measure; options...)
@deprecate innerproduct_native(Φ1::Dictionary, i, Φ2::Dictionary, j, measure; options...) dict_innerproduct_native(Φ1, i, Φ2, j, measure; options...)
@deprecate innerproduct1(Φ1::Dictionary, i, Φ2::Dictionary, j, measure; options...) dict_innerproduct1(Φ1, i, Φ2, j, measure; options...)
@deprecate innerproduct2(Φ1::Dictionary, i, Φ2::Dictionary, j, measure; options...) dict_innerproduct2(Φ1, i, Φ2, j, measure; options...)

@deprecate apply_map(dict::Dictionary, map) mapped_dict(dict, map)

@deprecate roots(b::OPS) ops_roots(b)

@deprecate dense_evaluation default_evaluation
