# Routines to convert between dictionaries

iscompatible(d1::GridBasis, d2::GridBasis) = length(d1) == length(d2)

conversion_operator(d1::GridBasis, d2::GridBasis; options...) =
    grid_conversion_operator(d1, d2, grid(d1), grid(d2); options...)
