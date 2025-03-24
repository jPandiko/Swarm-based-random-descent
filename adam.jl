#@author Jan Pandikow

# ----------------------------------------------------------------------------------------------

using Zygote
using DataFrames

@doc """
 This functions executes the ADAM method on from a set starting point. Works only for d=1.
 @param alpha : maximum step size
 @param beta_1 : drecrease rate of the first moment
 @param beta_2 : drecrease rate of the second moment
 @param eps : tolerance to prevent zero division error
 @param theta : starting point for adam method
 @param tol : lower bound until a number is treated as thero for convergance
 @returns location_store : store of every 100th point that is visited
 """
function adam_sgl(alpha, beta_1, beta_2, eps, f, theta, tol)
    m = 0 # first moment
    v = 0 # second moment
    t = 0 # time step
    current_location = theta # setup the current best location 
    last_location = theta + 1
    # only for presentative matters
    dataStore = DataFrame(location=Float64[],
        first_moment_biased=Float64[],
        first_moment_unbiased=Float64[],
        second_moment_biased=Float64[],
        second_moment_unbiased=Float64[],
        stepsize=Float64[])
    location_cnt = 0
    # until θ convergences
    while abs(current_location - last_location) > tol
        t = t + 1 # update time step
        # calculate the gradient on current location
        grad_t = gradient(x -> f(x), current_location)[1]
        # update biased first moment estimate
        m = beta_1 * m + (1 - beta_1) * grad_t
        # update biased second raw moment estimate
        v = beta_2 * v + (1 - beta_2) * grad_t^2
        # compute bias-corrected first moment estimate
        m_unbiased = m / (1 - beta_1^t)
        # compute bias_corrected second raw moment estimate
        v_unbiased = v / (1 - beta_2^t)
        # safe the current guess for convergence condition check
        last_location = current_location
        # update the current location guess
        update = alpha * m_unbiased / (sqrt(v_unbiased) + eps)
        println(update)
        current_location = current_location - update
        location_cnt += 1
        if location_cnt == 200
            push!(dataStore, (current_location, m, m_unbiased, v, v_unbiased, update))
            location_cnt = 0
        end
    end
    # add last location
    push!(dataStore, (current_location, 0, 0, 0, 0, 0))
    return dataStore
end


#--------------------------------------------------------------------------------------


# d = 2
# !!! note that for functions with higher dimension, the argument should be an array. There shouldn't be any additional parameters,
#     but the coordinates. The values can be unpacked in the function itself.
function f3(coods)
    x = coods[1]
    y = coods[2]
    return (x - 0.5)^2 + (y - 0.5)^2
end

function f5(coods)
    x = coods[1]
    y = coods[2]
    return sin(x) * cos(y)
end

function f6(coods)
    x = coods[1]
    y = coods[2]
    return x^3 - x + y^3 - y
end

# d = 3
function f4(coods)
    x = coods[1]
    y = coods[2]
    z = coods[3]
    return (x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2
end


#------------------------------------------------------------------------------

using Zygote
using LinearAlgebra

@doc """
 This functions executes the ADAM method on from a set starting point.
 @param alpha : maximum step size
 @param beta_1 : drecrease rate of the first moment
 @param beta_2 : drecrease rate of the second moment
 @param eps : tolerance to prevent zero division error
 @param theta : starting point for adam method
 @param tol : lower bound until a number is treated as thero for convergance
 @returns location_store : store of every 100th point that is visited
 """
function adam(alpha, beta_1, beta_2, eps, f, theta, tol)
    m = 0 # first moment
    v = 0 # second moment
    t = 0 # time step
    current_location = theta # setup the current best location 
    last_location = theta .+ 1

    #only for demonstration purpose
    location_store = []
    push!(location_store, theta)
    location_cnt = 0

    # max iteration counter
    counter = 0

    # until θ convergences
    # using the norm of the difference of both points
    while norm(current_location - last_location) > tol && counter < 10^5
        t = t + 1 # update time step
        # calculate the gradient on current location
        grad_t = gradient(x -> f(x), current_location)[1]
        # handle case when there is no gradient
        if isnothing(grad_t)
            # set gradient to zeros
            grad_t = zeros(size(theta)[1])
        end
        # update biased first moment estimate
        m = beta_1 .* m .+ (1 - beta_1) .* grad_t
        # update biased second raw moment estimate
        v = beta_2 .* v .+ (1 - beta_2) .* grad_t .^ 2
        # compute bias-corrected first moment estimate
        m_unbiased = m ./ (1 - beta_1^t)
        # compute bias_corrected second raw moment estimate
        v_unbiased = v ./ (1 - beta_2^t)
        # safe the current guess for convergence condition check
        last_location = current_location
        # update the current location guess
        update = alpha .* (m_unbiased ./ (sqrt.(v_unbiased) .+ eps))
        if counter % 50 == 0
            #println(update);
        end

        # only for graphic needed
        current_location = current_location - update
        if location_cnt == 100
            push!(location_store, current_location)
            location_cnt = 0
        end
        location_cnt += 1
        counter += 1
    end
    counter += 1
    # add last location to path
    push!(location_store, current_location)
    return location_store
end





@doc """
 This functions executes the ADAM method on from a set starting point. It doesnt track any data of 
 the path that ADAM takes. It's designed to work as fast as possible.
 @param alpha : maximum step size
 @param beta_1 : drecrease rate of the first moment
 @param beta_2 : drecrease rate of the second moment
 @param eps : tolerance to prevent zero division error
 @param theta : starting point for adam method
 @param tol : lower bound until a number is treated as thero for convergance
 """
function adam_test_version(alpha, beta_1, beta_2, eps, f, theta, tol)#

    m = 0 # first moment
    v = 0 # second moment
    t = 0 # time step
    current_location = theta # setup the current best location 
    last_location = theta .+ 1

    # max iteration counter
    counter = 0

    # until θ convergences
    # using the norm of the difference of both points
    while norm(current_location - last_location) > tol && counter < 10^5
        t = t + 1 # update time step
        # calculate the gradient on current location
        grad_t = gradient(x -> f(x), current_location)[1]
        # handle case when there is no gradient
        if isnothing(grad_t)
            # set gradient to zeros
            grad_t = zeros(size(theta)[1])
        end
        # update biased first moment estimate
        m = beta_1 .* m .+ (1 - beta_1) .* grad_t
        # update biased second raw moment estimate
        v = beta_2 .* v .+ (1 - beta_2) .* grad_t .^ 2
        # safe last location for convergence check
        last_location = current_location
        # perfom step
        alpha_t = alpha * sqrt(1 - beta_2^t) / (1 - beta_1^t)
        update = alpha_t .* m ./ (sqrt.(v) .+ eps)
        current_location = current_location - update
        counter += 1
    end
    return current_location
end


using LinearAlgebra
include("functions.jl");
# delete later
function find_parting_point()
    println("[+] started");
    range = 4
    dimension = 7
    optimum = [1,1,1,1,1,1,1]
    for j in 1:1000
        # create start point
        start = [rand() * 2 * range + optimum[i] - range for i in 1:dimension]
        path = adam(0.001, 0.9, 0.999, 10^(-8), f_rosenbrock, start, 10^-7)
        result = adam_test_version(0.001, 0.9, 0.999, 10^(-8), f_rosenbrock, start, 10^-7)
        println(norm(result-path[end]));
        if norm(result - path[end]) > 0.1
            println(start)
            break
        end
    end
    println();
end