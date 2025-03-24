#@author Jan Pandikow

using Zygote
using LinearAlgebra
using Distributions

# functions


@doc"""
f1(x) calculates the value of f(x,y) = x^2 + y^2.
@param x::[val1, val2] - x Value
@returns f(x)
"""
function f1(x)
    x1 = x[1];
    x2 = x[2];
    return x1^2 + x2^2;
end

@doc"""
f2(x) calculates the value f(x,y) = (x-1)^2 + (y-1)^2.
@param x::[val1, val2] - x_value
@returns f(x)
"""
function f2(x)
    x1 = x[1];
    x2 = x[2];
    return (x1-1)^2 + (x2-1)^2;
end

# helper functions.

@doc"""
This functions calculates a random descending direction based on the current location and the 
relative mass of an agent.
@paraf: current function
@param location: current location of the agent at time n 
@param mass: current mass of the agent at time n+1
@param tol: if value drops below tol -> it's handled like 0
@returns direction : descent direction !! not normed !!
@returns true_grad : gradient !! not normed !!
"""
function calc_direction(f, location, mass, tol)
    
    # calc the gradient
    true_grad = gradient(x -> f(x), location)[1];
    # if the random gradient is nothing -> set value to [0,...,0]
    if isnothing(true_grad)
        true_grad = zeros(size(location)[1]);
    end
    # norm the gradient
    grad = true_grad ./ norm(true_grad);
    # select random r with 0.5(1+mass) < r < 1 with uniform distribution
    # if 0.5*(1+mass) == 1 -> Uniform Distribution fails
    tmp = 0.5 * (1+mass);
    r = 1;
    if (tmp < 1)
        uniform_distribution = Uniform( 0.5 * (1+mass), 1);
        r = rand(uniform_distribution,1)[1];
    end

    # create d-1 array of normal distributed values
    d = length(location);
    normal_distribution = Normal(0,1);
    y = rand(normal_distribution, d-1);
    # norm y, as it fits in the "unit" sphere
    y = y ./ norm(y);

    # create x values:
    x = [sqrt(1-r^2)*y_i  for y_i in y];
    push!(x, r);
  
    # Housholder projection
    z = vcat(zeros(d-1), [1])
    # if 1- grad(d) == 0
    omega = 0;
    if 1-grad[end] > tol
        v = grad - z;
        omega = x - (2 * dot(v, x) / norm(v)^2) * v;
    else
        omega = x;
    end

    # setup setup the direction 
    direction = norm(true_grad) * omega;

    return direction, true_grad;
end


@doc"""
This function calculates the angle between to given vectors.
Only used for test cases.
@param v1,v2 : vectors 
@returns theta : angle between the vectors.
"""
function angle_between_vectors(v1, v2)
    # Compute dot product
    dot_product = dot(v1, v2)
    
    # Compute norms of the vectors
    norm_v1 = norm(v1)
    norm_v2 = norm(v2)
    
    # Compute cosine of the angle
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Compute angle in radians
    theta = acos(clamp(cos_theta, -1.0, 1.0))  # Clamp to avoid numerical issues
    
    return theta
end


@doc"""
This function calculates the step length in a given direction. Start with stepsize = 1. Decreases by
factor gamma each iteration, until demand on stepsize is satisfied.
@param gamma : shrinking factor of stepsize
@param mass : relative mass of the given agent at n+1
@param current_gradient: gradient at current location
@param f : current function
@param current_location: current location of the agent
@param lambda : sets influence of the mass on stepsize
@returns stepsize
"""
function backtracking(gamma, mass, current_gradient, f, stepsize_start, step_direction, current_location, lambda)
    # initialize step size
    stepsize = stepsize_start;
    grad_abs_sqrd = norm(current_gradient)^2; # TODO why is this squared ???
    # using wolf condition
    while (f(current_location - stepsize .* step_direction) > f(current_location) - 0.5*lambda*mass*stepsize*grad_abs_sqrd)
        # fit stepsize
        stepsize = stepsize * gamma;
    end
    return stepsize;
end


@doc"""
This function selects the best agent by comparing the f(x) values. Since f(x) is calculated, we
also return the highest and lowest values of f(x).
@param positions : positions of the agents at time n 
@param f : current function
@returns index: index of the agent with lowest f(x)
@returns f_min, f_max: lowest and highest value of f(x).
"""
function get_optimal_agent(positions, f)
    index = 1;
    size = length(positions);
    # if array is empty
    if size < 1
        return -1,0,0;
    # if only one agent is contained
    elseif size == 1
        f_val = f(positions[1]);
        return 1,f_val,f_val;
    else
        # create all f(x) values
    	f_values = map(f,positions);
        # get f_max
         f_max = maximum(f_values);
        # get index
        index = argmin(f_values);
        f_min = f(positions[index]);
        return index, f_min, f_max;
    end
end
            

@doc"""
The function merges the agents, when distance is below tolerance. If two agents are merged
the mass of the second agent is added to the first. The mass of the second agent is set on 0.
@param positions: positions of the agents at time n 
@param mass: mass of the agents at time n 
@param tolmerge: tolerance of the minimum distance until agents get merged
"""
function merge_agents!(positions, mass, tolmerge)
    N = length(positions)
    # for each agent, check the agents that come after iteration
    for i in 1:N-1
        # check if the iteration is needed
        if mass[i] != 0
            for j in i+1:N
                # check for tolerance
                if norm(positions[i]-positions[j]) < tolmerge && mass[j] != 0
                    mass[i] += mass[j];
                    mass[j] = 0;
                end
            end
        end
    end
end


@doc"""
The functions removes all agents, where the current mass is below the tolerance.
@param positions : current positions of the agents at time n 
@param mass : current mass of the agents at time n 
@param tolmass: minimal mass an agent can have until its to be removed
@best_index : index of the best positioned agent
@ killed : is true, when an agent was removed, else its false
"""
function kill_agents!(positions, mass, tolmass, best_index)
    N = length(mass);
    index = 1;
    killed = false;
    # until the end of the agents
    while index <= length(mass)
        # not the best agent
        if index != best_index
            if mass[index] < 1/N * tolmass
                # delete position and mass
                deleteat!(mass, index);
                deleteat!(positions, index);
                N = N-1;
                killed = true;
            else
                # update current index
                index += 1;
            end
        # if the element is not worked on
        else
            # move to next one
            index += 1;
        end
    end
    return killed;
end


@doc"""
This function returns the biggest mass value of the current masses of all agents
@param mass : mass of all agents at time n+1
@return max : maximum of all masses
"""
function get_max_mass(mass)
    N = length(mass);
    if N < 1
        return -1;
    elseif N == 1;
        return mass[1];
    else
        max = mass[1];
        for i in 2:N
            if max < mass[i];
                max = mass[i];
            end
        end
        return max;
    end
end



@doc"""
This method performs the swarm based random descent method on a given function.
@param tolmass: lower bound for mass of an agent before beeing deleted
@param tolmerge: lower bound for distance of to agents before beeing merged
@param tolres : lower bound of movement of the two best agents at time n and n+1 before method is stoped
@param nmax : upper bound for number of iterations
@param N : initial number of agents
@param q : Parameter for mass distribution and move of mass between agents
@param f : function sbrd is applied on
@param d : dimension of the problem
@param a,b : bounderies for random placement of the agents
@param gamma : decrease factor for backtracking method
@param step_size: initial step size for backtracking method
@param lambda : factor of the influence of the relative mass on the length taken by backtracking method
"""
function sbrd_graphics(tolmass, tolmerge, tolres, nmax, N, q, f, d, a,b, gamma, stepsize_start, lambda)

    println("[+] swarm based started");
    
    # setup the Agents
    random_uniform_initial = Uniform(a,b);
    positions = [rand(random_uniform_initial, d) for i in 1:N];
    mass = [1/N for i in 1:N];

    println("[+] agents placed");

    # list to tracke the position changes that where made
    movement_tracker = [];

    # setup counter for max iteration limit
    counter = 0;

    # help variable to track convergence 
    last_best_position = 0;
    first_iteration_flag = true;

    while counter < nmax


        
        # merge the agents with postions to close to each other
        merge_agents!(positions, mass, tolmerge);

        # set the index for the optimal agents, F_min, F_max
        index_optima, F_min, F_max = get_optimal_agent(positions, f);

        # delete agents under the tolarence
        agents_killed_flag = kill_agents!(positions, mass, tolmass, index_optima);
        N = length(positions);


        # update the indizes, if agents were agents_killed
        if agents_killed_flag
            println("[+] agent removed in iteration: ", counter);
            index_optima, F_min, F_max = get_optimal_agent(positions, f);
        end

        # calculate new mass
        diff = F_max - F_min;
        for i in 1:N
            if i != index_optima
                # check if diff > 0 : if not -> dont perform the mass transfer
                if diff > 0
                    factor = (f(positions[i]) - F_min) / diff;
                    factor = factor^q;
                    # return get mass to best agent
                    mass[index_optima] = mass[index_optima] + factor * mass[i]
                    # reduce mass for current agent

                    mass[i] = mass[i] - factor * mass[i];
                else
                    # stop the iteration since there is no mass transfer
                    break;
                end
            end
        end
        
        # find max mass to calc relative mass
        max_mass = get_max_mass(mass);
        

        # for each agent
        for i in 1:N
            # calculate the relative mass
            relativ_mass = mass[i]/max_mass;
            # compute a random step direction
            step_direction, current_gradient = calc_direction(f, positions[i], relativ_mass, tolres);
            # compute the stepsize
            current_stepsize = backtracking(gamma, relativ_mass, current_gradient, f, stepsize_start, step_direction, positions[i], lambda)
            if N == 1
                #println("[+] location: ", positions[i]);
                println("[+] descent direction length factor: ", current_stepsize);
                # check for need of breakpoint, if point get
            end
            # compute the new position
            new_position = positions[i] - current_stepsize * step_direction;
            # store the movement
            push!(movement_tracker, [positions[i], new_position]);
            # safe the taken step
            positions[i] = new_position;
        end


        # check whether best agent is not moving
        if first_iteration_flag
            println("[+] first iteration flag deactivated ");
            first_iteration_flag = false
            last_best_position = positions[index_optima];
        else
            # check if agent of last iteration has moved
            println("[+] diff between steps: ", last_best_position);
            if norm(last_best_position .- positions[index_optima]) < tolres
                break
            else
                # safe current best agent for next iteration
                last_best_position = positions[index_optima];
            end
        end 
        
        counter += 1;
    end

    # return the best position
    best_p = positions[1];
    for pos in positions
        if f(pos) < f(best_p)
            best_p = pos
        end
    end
    
    return best_p, movement_tracker;
end


@doc"""
Same functionality as sbrd_graphics. Does work without status updates and does not track the movements
of the agents.
"""
function sbrd(tolmass, tolmerge, tolres, nmax, N, q, f, d, a,b, gamma, stepsize_start, lambda)

    # setup the Agents
    random_uniform_initial = Uniform(a,b);
    positions = [rand(random_uniform_initial, d) for i in 1:N];
    mass = [1/N for i in 1:N];

    # setup counter for max iteration limit
    counter = 0;

    # help variable to track convergence 
    last_best_position = 0;
    first_iteration_flag = true;

    while counter < nmax
        
        # merge the agents with postions to close to each other
        merge_agents!(positions, mass, tolmerge);

        # set the index for the optimal agents, F_min, F_max
        index_optima, F_min, F_max = get_optimal_agent(positions, f);

        # delete agents under the tolarence
        agents_killed_flag = kill_agents!(positions, mass, tolmass, index_optima);
        N = length(positions);


        # update the indizes, if agents were agents_killed
        if agents_killed_flag
            index_optima, F_min, F_max = get_optimal_agent(positions, f);
        end

        # calculate new mass
        diff = F_max - F_min;
        for i in 1:N
            if i != index_optima
                # check if diff > 0 : if not -> dont perform the mass transfer
                if diff > 0
                    factor = (f(positions[i]) - F_min) / diff;
                    factor = factor^q;
                    # return get mass to best agent
                    mass[index_optima] = mass[index_optima] + factor * mass[i]
                    # reduce mass for current agent

                    mass[i] = mass[i] - factor * mass[i];
                else
                    # stop the iteration since there is no mass transfer
                    break;
                end
            end
        end
        # find max mass to calc relative mass
        max_mass = get_max_mass(mass);
        

        # for each agent
        for i in 1:N
            # calculate the relative mass
            relativ_mass = mass[i]/max_mass;
            # compute a random step direction
            step_direction, current_gradient = calc_direction(f, positions[i], relativ_mass, tolres);
            # compute the stepsize
            current_stepsize = backtracking(gamma, relativ_mass, current_gradient, f, stepsize_start, step_direction, positions[i], lambda);
            # compute the new position
            new_position = positions[i] - current_stepsize * step_direction;
            # safe the taken step
            positions[i] = new_position;
        end

        # check whether best agent is not moving
        if first_iteration_flag
            first_iteration_flag = false
            last_best_position = positions[index_optima];
        else
            # check if agent of last iteration has moved
            if norm(last_best_position .- positions[index_optima]) < tolres
                break
            else
                # safe current best agent for next iteration
                last_best_position = positions[index_optima];
            end
        end
        
        counter += 1;
    end

    # return the best position
    best_p = positions[1];
    for pos in positions
        if f(pos) < f(best_p)
            best_p = pos
        end
    end
    
    return best_p;
end


@doc"""
Same functionality as sbrd. The method stops after one agent is left. There is no convergence. The possibility to break the movement early,
if the best agent doesnt converge remains.
"""
function sbrd_without_convergence(tolmass, tolmerge, tolres, nmax, N, q, f, d, a,b, gamma, stepsize_start, lambda)
    # setup the Agents
    random_uniform_initial = Uniform(a,b);
    positions = [rand(random_uniform_initial, d) for i in 1:N];
    mass = [1/N for i in 1:N];

    # setup counter for max iteration limit
    counter = 0;

    # help variable to track convergence 
    last_best_position = 0;
    first_iteration_flag = true;

    while counter < nmax
        
        # merge the agents with postions to close to each other
        merge_agents!(positions, mass, tolmerge);

        # set the index for the optimal agents, F_min, F_max
        index_optima, F_min, F_max = get_optimal_agent(positions, f);

        # delete agents under the tolarence
        agents_killed_flag = kill_agents!(positions, mass, tolmass, index_optima);
        N = length(positions);


        # update the indizes, if agents were agents_killed
        if agents_killed_flag
            index_optima, F_min, F_max = get_optimal_agent(positions, f);
        end

        # calculate new mass
        diff = F_max - F_min;
        for i in 1:N
            if i != index_optima
                # check if diff > 0 : if not -> dont perform the mass transfer
                if diff > 0
                    factor = (f(positions[i]) - F_min) / diff;
                    factor = factor^q;
                    # return get mass to best agent
                    mass[index_optima] = mass[index_optima] + factor * mass[i]
                    # reduce mass for current agent

                    mass[i] = mass[i] - factor * mass[i];
                else
                    # stop the iteration since there is no mass transfer
                    break;
                end
            end
        end
        # find max mass to calc relative mass
        max_mass = get_max_mass(mass);
        

        # for each agent
        for i in 1:N
            # calculate the relative mass
            relativ_mass = mass[i]/max_mass;
            # compute a random step direction
            step_direction, current_gradient = calc_direction(f, positions[i], relativ_mass, tolres);
            # compute the stepsize
            current_stepsize = backtracking(gamma, relativ_mass, current_gradient, f, stepsize_start, step_direction, positions[i], lambda);
            # compute the new position
            new_position = positions[i] - current_stepsize * step_direction;
            # safe the taken step
            positions[i] = new_position;
        end

        # check wether the number of agents is 1
        if N == 1
            break
        end
        counter += 1;

    end


    # return the best position
    best_p = positions[1];
    for pos in positions
        if f(pos) < f(best_p)
            best_p = pos
        end
    end
    
    return best_p;
end




# delete later 
function test(f, dimension, range)
    # parameter used for the algorithm
    tolmass = 10e-10;
    tolmerge = 10e-10;
    tolres = 10e-10;
    nmax = 100;
    N = 10;
    q = 2;
    d = dimension;
    a = -range;
    b = range;
    gamma = 0.9
    stepsize_start = 1;
    lambda = 1;
    result = sbrd(tolmass, tolmerge, tolres, nmax, N, q, f, d, a,b, gamma, stepsize_start, lambda);
    return result;
end