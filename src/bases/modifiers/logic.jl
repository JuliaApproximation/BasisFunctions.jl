# file for operations that involve multiple Dictionary types

==(dict1::TensorProductDict, dict2::ComplexifiedDict) =
    all(map(==,elements(dict1),elements(dict2)))
==(dict1::ComplexifiedDict, dict2::TensorProductDict) =
    all(map(==,elements(dict1),elements(dict2)))


ComplexifiedDict(dict::TensorProductDict) =
    TensorProductDict(map(ComplexifiedDict, elements(dict))...)
