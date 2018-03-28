# Methods that override the standard show(io::IO,op::AbstractOperator), to be better understandable.

# Delegate to show_operator
show(io::IO, op::GenericOperator) = show_operator(io, op)

# Default is the operator string
show_operator(io::IO,op::GenericOperator) = println(string(op))

# Default string is the string of the type
string(op::GenericOperator) = match(r"(?<=\.)(.*?)(?=\{)",string(typeof(op))).match

# Complex expressions substitute strings for symbols.
# Default symbol is first letter of the string
symbol(op::GenericOperator) = string(op)[1]

# Common symbols (to be moved to respective files)
symbol(E::IndexRestrictionOperator) = "R"
symbol(E::IndexExtensionOperator) = "E"

string(op::MultiplicationOperator) = string(op,op.object)
string(op::MultiplicationOperator,object) = "Multiplication by "*string(typeof(op.object))

symbol(op::MultiplicationOperator) = symbol(op,op.object)
symbol(op::MultiplicationOperator,object) = "M"

symbol(op::MultiplicationOperator,object::Base.DFT.FFTW.cFFTWPlan{T,K}) where {T,K} = K<0 ? "FFT" : "iFFT" 
symbol(op::MultiplicationOperator,object::Base.DFT.FFTW.DCTPlan{T,K}) where {T,K} = K==Base.DFT.FFTW.REDFT10 ? "DCT" : "iDCT" 

function string(op::MultiplicationOperator, object::Base.DFT.FFTW.cFFTWPlan) 
    io = IOBuffer()
    print(io,op.object)
    match(r"(.*?)(?=\n)",String(take!(io))).match
end

function string(op::MultiplicationOperator, object::Base.DFT.FFTW.DCTPlan) 
    io = IOBuffer()
    print(io,op.object)
    String(take!(io))
end


# Different operators with the same symbol get added subscripts
subscript(i::Integer) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))


    # Include parentheses based on precedence rules
    parentheses(AbstractOperator)=false
# By default, don't add parentheses
parentheses(t::AbstractOperator,a::AbstractOperator) = false
# Sums inside everything need parentheses
parentheses(t::CompositeOperator,a::OperatorSum) = true
parentheses(t::TensorProductOperator,a::OperatorSum) = true
# Mixing and matching products need parentheses
parentheses(t::CompositeOperator,a::TensorProductOperator) = true
parentheses(t::TensorProductOperator,a::CompositeOperator) = true
    
        
