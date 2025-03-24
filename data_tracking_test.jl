# @author Jan Pandikow

using TimerOutputs;
using DataFrames;
using LinearAlgebra;
using Printf;
using Dates;
using JSON; # use for dataFrame as it provides complex structures
using JSON3; # for dictionary support

include("functions.jl"); # use functions
include("sbrd.jl"); # use swarm based method
include("adam.jl"); # use adam method
 

# setup timer object
const timer = TimerOutput();



@doc"""
This functions calculates the errors. We track the error between the optimal x-value and the optimal solution.
Also the f(x) value and the the error between the y-value of the given solution and the y-value of the given optimum.
The results are stored as a line in a DataFrame.
@param f : function to be calculated
@param x_val : calculated solution
@param optimum : optimal solution (if multiple solutions are possible, the nearest one will be selected)
@param dataStore : location, where data is to be stored
@param name : title of the iteration -> to track it to timing data
"""
function calculate_errors!(name::String, f, x_val, optimum, dataStore::DataFrame)
    x_error = norm(x_val - optimum);
    f_val = f(x_val);
    f_error = norm(f_val - f(optimum));
    push!(dataStore, (name, x_val, x_error, f_val, f_error));
end



@doc"""
This functions takes the data in form of a dictionary and stores it into a textfile.
There are 3 files created. First for regular information (date of creation, used algorithm, number of iterations).
The second one stores the data of the timer. The third stores the data of the x values.
@param used_algorithm : name of the used algorithm
@param paras_algorithm : stores the parameter used for algorithm.
@param number_of_iterations : iteration counter used in test
@param precision : precision used in test
@param range : range used in test
@param optimum : optimum used in test
@param function_name : name of the function used in test
@param data_timing : contains the timing data - often as Dict
@param data_function : contains the data of the solution and the made error
"""
function safe_data_in_file(used_algorithm::String, paras_algorithm, number_of_iterations, precision, range, 
    optimum, function_name, data_timing, data_function::DataFrame)
    # safe date for file_name
    today_date = string(today());
    # create new folder to store data
    location_folder = "data";
    base_folder = joinpath(location_folder, used_algorithm * "_" * today_date);
    println(base_folder);
    if !isdir(base_folder)
        mkdir(base_folder)
        println("Base folder '$base_folder' created.");
    else
        # if a folder already exists create a new name with ...(1), ...(2) etc.
        currentNumber = 1;
        current_name = used_algorithm * "_" * today_date * "(" * string(currentNumber) * ")";
        base_folder = joinpath(location_folder, current_name)
        while isdir(base_folder)
            currentNumber += 1;
            current_name = used_algorithm * "_" * today_date * "(" * string(currentNumber) * ")";
            base_folder = joinpath(location_folder, current_name)
        end
        mkdir(base_folder);
    end

    # create the general information file
    general_information_file_path = joinpath(base_folder, "general_information.txt")
    # write date, algorithm, iterations:
    open(general_information_file_path, "w") do file
        # Write each piece of information with explicit newline characters
        write(file, "algorithm : " * used_algorithm * " : ");        # Algorithm used in the test
        # write the parameter from the dictionary
        for key in keys(paras_algorithm)
            write(file, "[" * string(key) * "-" * string(paras_algorithm[key]) * "]");
        end
        write(file, "\n");
        write(file, "date: " * today_date * "\n");                       # Date
        write(file, "number of iterations: " * string(number_of_iterations) * "\n");  # Number of iterations
        write(file, "precision: " * string(precision) * "\n");           # Precision
        write(file, "range: " * string(range) * "\n");                   # Range
        write(file, "optimum: " * string(optimum) * "\n");               # Optimum
        write(file, "function name: " * string(function_name) * "\n");   # Function name
    end

    # write the data of the time tracking dictionary into json file
    time_data_file_path = joinpath(base_folder, "timing_data.txt");
    data_as_string = JSON3.write(data_timing);
    open(time_data_file_path, "w") do file
        write(file, data_as_string);
    end


    # write data of the dataframe into JDF-file
    function_error_data_file_path = joinpath(base_folder, "function_error_data.txt");
    function_error_data_as_string = JSON.json(data_function);
    open(function_error_data_file_path, "w") do file
        write(file, function_error_data_as_string);
    end
end



@doc"""
This function loads the data from a given folder. It return a dictionary with the general information. A dictionary with
with the information of the timer and a dataframe with the function error information. If the original project structure
is used, mark the flag. Then only the test name is needed. 
@param location : name of the test that is to be loaded : if original structure remains, only the test name is needed
@param original_structure : marks wether the original structure still can be used
@returns function_error_data : DataFrame with error and solution information from "function_error_data.txt"
@returns timing_data_as_dict : Dict with timing information from "timing_data.txt"
@return general_information_data_as_vector : Dict with general information from "general_information.txt"
"""
function load_test_from_file(location, original_structure::Bool)

    # build path to base folder
    base_folder = "";
    if original_structure 
        # if original structure is still in use build path
        base_folder = joinpath(@__DIR__, "data", location);
    else
        # else: use given path
        base_folder = location;
    end

    
    # load the function error data
    function_error_file_path = joinpath(base_folder, "function_error_data.txt");
    println(function_error_file_path);
    function_error_file = open(function_error_file_path, "r");
    function_error_data_as_string = read(function_error_file, String);
    close(function_error_file);
    # use JSON to parse into DataFrame
    function_error_data = DataFrame(JSON.parse(function_error_data_as_string));


    # load the timing data
    timing_data_file_path = joinpath(base_folder, "timing_data.txt");
    println(timing_data_file_path);
    timing_file = open(timing_data_file_path, "r");
    timing_data_as_json = JSON3.read(timing_file);
    timing_data_as_dict = Dict(timing_data_as_json);
    close(timing_file);


    # load the test infos into a dictionary
    test_information_file_path = joinpath(base_folder, "general_information.txt")
    lines = readlines(test_information_file_path);
    general_information_data = Dict{String, String}();
    for line in lines
        parts = split(line, ":", limit=2);
        key = strip(parts[1]); 
        value = strip(parts[2]);
        general_information_data[key] = value;
    end

    


    return function_error_data, timing_data_as_dict, lines, general_information_data;  
end




@doc"""
This function runs a test for the adam algorithm. It tracks the storage used. The time needed.
The quality of the solution and the parameters used for the test.
@param iterations : times the experiment is to be reapeted
@param precision : precision which is required to end the algorithm
@param range : distance from the optimum where the algorithm starts
@param optimum : center, where the starting points are placed around
@param dimension : dimension where the problem is executed
@param f : the function, which ADAM is tested on
"""
function test_adam(iterations, precision, range, optimum, dimension, f)
    
    # paramter for adam
    alpha = 0.1;
    beta_1 = 0.9;
    beta_2 = 0.999;
    eps = 10e-8;
    
    # DataFrame to store the results of the methods
    # DataFrame needs to be serialized later 
    # -> solution vector needs to be stored element by element
    data_function_error = DataFrame(name = String[],
                          solution=Vector{Vector{Float64}}(), 
                          x_error = Float64[],
                          f_value = Float64[],
                          f_error = Float64[],
    );


    for iteration_counter in 1:iterations

        # find random starting point 
        start = [rand() * 2 * range + optimum[i] - range for i in 1:dimension]
        
        # time the function and safe result
        title_of_iteration = "adam : " * string(iteration_counter);
        @timeit timer title_of_iteration begin
            result = adam_test_version(alpha, beta_1, beta_2, eps, f, start, precision);
        end
        calculate_errors!(title_of_iteration, f, result, optimum, data_function_error);
    end

    # write the data into an external file
    # collcect the data from timer
    data_timing = TimerOutputs.todict(timer)["inner_timers"];

    # store the parameter used in the test in dictionary
    paras_algorithm = Dict("alpha" => alpha, "beta_1" => beta_1, "beta_2" => beta_2, "eps" => eps);
    
    # safe the data in local folder
    safe_data_in_file("adam", paras_algorithm, iterations, precision, range, optimum, f, data_timing, data_function_error);
end



@doc"""
This function runs a test for the sbrd method. It tracks the storage used. The time needed.
The quality of the solution and the parameters used for the test. Since the sbrd-method places
it starting locations for itself, the optimum and range define the starting point and the bounderies
used for this random placement.
@param iterations : times the experiment is to be reapeted
@param range : distance from the optimum where the algorithm starts
@param optimum : center, where the starting points are placed around
@param dimension : dimension where the problem is executed
@param f : the function, which SBRD is tested on
"""
function test_sbrd(iterations, range, optimum, dimension, agents , f)

    # parameter used for the algorithm
    tolmass = 10e-4;
    tolmerge = 10e-3;
    tolres = 10e-4;
    nmax = 200;
    N = agents;
    q = 2;
    d = dimension;
    # use range around the optimum -> move optimum in the center of start search area
    a = optimum[1] - range;
    b = optimum[1] + range;
    gamma = 0.9
    stepsize_start = 1;
    lambda = 0.2;

    # DataFrame to store the results of the methods
    # DataFrame needs to be serialized later 
    # -> solution vector needs to be stored element by element
    data_function_error = DataFrame(name = String[],
                          solution=Vector{Vector{Float64}}(), 
                          x_error = Float64[],
                          f_value = Float64[],
                          f_error = Float64[],
    );


    for iteration_counter in 1:iterations
        # time the function and safe result
        title_of_iteration = "sbrd : " * string(iteration_counter);
        @timeit timer title_of_iteration begin
            result = sbrd(tolmass, tolmerge, tolres, nmax, N, q, f, d, a,b, gamma, stepsize_start, lambda);
        end
        calculate_errors!(title_of_iteration, f, result, optimum, data_function_error);
    end


    # write the data into an external file
    # collcect the data from timer
    data_timing = TimerOutputs.todict(timer)["inner_timers"];

    # store the parameter used in the test in dictionary
    paras_algorithm = Dict("tolmerge"=>tolmerge, "tolmass"=> tolmass, "tolres"=>tolres, "nmax"=>nmax, "N"=>N, "q"=>q, "d"=>d, "a"=>a, "b"=>b, "gamma"=>gamma, "stepsize_start"=>stepsize_start, "lambda"=>lambda);
    
    # safe the data in local folder
    precision = "xxx";
    safe_data_in_file("sbrd", paras_algorithm, iterations, precision, range, optimum, f, data_timing, data_function_error);
end




@doc"""
This function runs a test for the sbrd_without_convergence method. It tracks the storage used. The time needed.
The quality of the solution and the parameters used for the test. Since the sbrd-method places
it starting locations for itself, the optimum and range define the starting point and the bounderies
used for this random placement.
@param iterations : times the experiment is to be reapeted
@param range : distance from the optimum where the algorithm starts
@param optimum : center, where the starting points are placed around
@param dimension : dimension where the problem is executed
@param f : the function, which SBRD is tested on
"""
function test_sbrd_without_convergence(iterations, range, optimum, dimension, agents , f)

    # parameter used for the algorithm
    tolmass = 10e-4;
    tolmerge = 10e-3;
    tolres = 10e-4;
    nmax = 200;
    N = agents;
    q = 2;
    d = dimension;
    # use range around the optimum -> move optimum in the center of start search area
    a = optimum[1] - range;
    b = optimum[1] + range;
    gamma = 0.9
    stepsize_start = 1;
    lambda = 0.2;

    # DataFrame to store the results of the methods
    # DataFrame needs to be serialized later 
    # -> solution vector needs to be stored element by element
    data_function_error = DataFrame(name = String[],
                          solution=Vector{Vector{Float64}}(), 
                          x_error = Float64[],
                          f_value = Float64[],
                          f_error = Float64[],
    );


    for iteration_counter in 1:iterations
        # time the function and safe result
        title_of_iteration = "sbrd_without_convergence : " * string(iteration_counter);
        @timeit timer title_of_iteration begin
            result = sbrd(tolmass, tolmerge, tolres, nmax, N, q, f, d, a,b, gamma, stepsize_start, lambda);
        end
        calculate_errors!(title_of_iteration, f, result, optimum, data_function_error);
    end


    # write the data into an external file
    # collcect the data from timer
    data_timing = TimerOutputs.todict(timer)["inner_timers"];

    # store the parameter used in the test in dictionary
    paras_algorithm = Dict("tolmerge"=>tolmerge, "tolmass"=> tolmass, "tolres"=>tolres, "nmax"=>nmax, "N"=>N, "q"=>q, "d"=>d, "a"=>a, "b"=>b, "gamma"=>gamma, "stepsize_start"=>stepsize_start, "lambda"=>lambda);
    
    # safe the data in local folder
    precision = "xxx";
    safe_data_in_file("sbrd_without_convergence", paras_algorithm, iterations, precision, range, optimum, f, data_timing, data_function_error);
end


@doc"""
This functions runs all tests for a given target function using test_adam and test_sbrd for the dimension d=2,...,6. 
The srbd tests are performed for N=10,25,50,100.
@param func : the target functions which the tests are performed on
"""
function run_all_tests_for_function(func)
    iterations = 1000;
    range = 3;
    optimum = [-2.903534,-2.903534,-2.903534,-2.903534,-2.903534,-2.903534];
    precision = 10e-5; # only for adam
    
    for dimension in 2:6
        current_opt = optimum[1:dimension];
        println(current_opt);
        # perform  adam
        test_adam(iterations, precision, range, current_opt, dimension, func);
        for number_of_agents in [10,25,50,100]
            # perfom sbrd
            test_sbrd(iterations, range, current_opt, dimension, number_of_agents, func);
        end
        println("[+] ", dimension, " done");
    end
end


@doc"""
This functions runs all test for a given target function using test_sbrd_without_convergence for the dimension
d=2,...,6. The sbrd_without_convergence tests are performed for N = 10,25,50,100.
@param func : the target function which the tsts are perfomed on.
"""
function run_all_tests_for_sbrd_without_convergence(func)
    iterations = 1000;
    range = 3;
    optimum = [0,0,0,0,0,0];
    
    for dimension in 2:6
        current_opt = optimum[1:dimension];
        println(current_opt);
        for number_of_agents in [10,25,50,100]
            # perfom sbrd
            test_sbrd_without_convergence(iterations, range, current_opt, dimension, number_of_agents, func);
        end
        println("[+] ", dimension, " done");
    end
end
