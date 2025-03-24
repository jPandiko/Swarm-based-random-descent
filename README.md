This project contains the base files needed for my bachelor thesis " A comperization of the ADAM-method with Swarm-based-random-descent". It includes the files necessary to run both methods.
Also it contains the methods used for testing, storing the results locally and to reload the results from the format.

The code needed for the ADAM-algorithm is stored in the file "adam.jl". It contains 2 main methods ("adam()" and "adam_test_version()"). The first is used to further analyse the path ADAM takes.
It stores every 200th step and returns the path at the end. The second method is used for the testing. It's the numerical optimized version and only returns the computed solution. 

The code needed for Swarm-based-random-descent is stored in the file "sbrd.jl". It contains 2 main methods, which can perform the algorithm. The first is "sbrd_graphics()". This methods performes 
SBRD and tracks every step the swarm has taken. This method is used to display the behavior of the swarm. The second method is "sbrd()". This version only stores the necessary values and only returns
the computed solution. It's also used in the testing files. This file also contains SBRD with a different ending-criteria. 

The functions, on which the minimization problems are performed on, are stored in the file "functions.jl".

The file "data_tracking_test.jl" contains all methods for testing. Every test stores its data in a folder in "\data". A test tracks the time used, the solution and the parameter of the test.
The file also contains the method for reloading the data from the files.

!!! To use these files you need to install Zygote and TimerOutputs first !!!
