
"""
`MyDictionary` is a custom dictionary type that illustrates the minimal
functionailty required to implement a dictionary.
"""
struct MyDictionary{S,T} <: Dictionary{S,T}
    n       ::  Int
end

# We provide a constructor that will choose a default length and default domain type.
MyDictionary(n = 10) = MyDictionary{Float64}(n)
# For convenience, we automatically choose a codomaintype if only a domaintype is given.
MyDictionary{S}(n) where {S} = MyDictionary{S,S}(n)

# The `length` function is implemented in terms of size.
# The size of a dictionary is always a tuple, just like for arrays.
size(dict::MyDictionary) = (dict.n,)

# By implementing `similar`, we support promotion of the domain type, as well
# as resizing the dictionary.
# If resizing is not possible, one can add a statement such as
# "@assert n == length(dict)", which will cause an error upon any user attempt to resize.
# Same for promotion.
similar(dict::MyDictionary, ::Type{S}, n::Int) where {S} = MyDictionary{S}(n)

support(dict::MyDictionary) = UnitInterval{domaintype(dict)}()

unsafe_eval_element(dict::MyDictionary, idx, x) = x^(idx-1)
