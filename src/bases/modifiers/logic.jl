# file for operations that involve multiple Dictionary types

==(dict1::TensorProductDict, dict2::ComplexifiedDict) =
    all(map(==,components(dict1),components(dict2)))
==(dict1::ComplexifiedDict, dict2::TensorProductDict) =
    all(map(==,components(dict1),components(dict2)))


ComplexifiedDict(dict::TensorProductDict) =
    TensorProductDict(map(ComplexifiedDict, components(dict))...)
