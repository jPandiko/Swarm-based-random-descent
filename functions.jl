@doc"""
This file provides the tested functions. Some functions require to an additional information of the dimension which they are used in.
@author Jan Pandikow
"""

@doc"""
This function calculates the value of the Rosenbrock-function.
minima expected at: x = (1,...,1).
@param x : x-value
@returns y : f(x)
"""
function f_rosenbrock(x)
    d = size(x)[1];
    y = 0;
    for i in 1:(d-1)
        y += 100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2;
    end
    return y;
end



@doc"""
This function calculates the value of the Rastrigin-function.
minima expected at: x = (0).
@param x : x-value
@returns y : f(x)
"""
function f_rastrigin(x)
    d = size(x)[1];
    y = 10 * d;
    for i in 1:d
        y += x[i]^2 - 10 * cos(2 * pi * x[i]);
    end
    return y;
end


@doc"""
This function calculates the value of the drop_with_valley landscape.
@param x : x-value
@returns y : f(x)
"""
function f_flat_with_valley(x)
    if norm(x) < 1
        return sum( x_val -> x_val^2, x) - 1;
    else
        return 0;3
    end
end


@doc"""
This function calculates the value of the himmelblau-function.
!!! This function is currently only working with d = 2.
@param x : x-value
@returns y : f(x)
"""
function f_himmelblau(x)
    return (x[1]^2+x[2]-11)^2+(x[1]+x[2]^2-7)^2;
end


@doc"""
This function calculates the value of the drop_wave_function.
!!! This function is currently only working with d = 2.
@param x : x-value
@returns y : f(x)
"""
function f_drop_wave_function(x)
    sum_of_x = sum(a -> a^2, x);
    return -(1 + cos(12*sqrt(sum_of_x))) / (0.5 * (sum_of_x) + 2);
end



@doc"""
This function calculates the value of the styblinski_tang_landscape.
@param x : x-value
@returns y : f(x)
"""
function f_styblinski_tang_landscape(x)
    d = size(x)[1];
    sum = 0;
    for i in 1:d
        sum += x[i]^4-16*x[i]^2+5*x[i];
    end
    sum = 0.5 * sum;
    return sum;
end


function f_ackley_landscape(x, a::Float64 = 20.0, b::Float64 = 0.2, c::Float64 = 2 * pi)
    d = length(x)
    term1 = -a * exp(-b * sqrt(sum(x .^ 2) / d))
    term2 = -exp(sum(cos.(c .* x)) / d)
    return term1 + term2 + a + exp(1.0)
end
