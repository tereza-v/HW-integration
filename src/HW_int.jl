
module HW_int

	using Plots
	using FastGaussQuadrature
	using Roots
	using Sobol
	using Distributions
	using SymPy

	q(p) = 2 .* p.^-.5

	function question_1b(n)

		# Gauss-Legendre transformation. Note that (b-a)/2 = (4-1)/2 = 3/2
		transformation(p) = 3 .* (3/2 .* p .+ 3/2).^-.5

		nodes, weights = gausslegendre(n)
    result = dot(weights, transformation(nodes))

		# Percent error
		percent_error = (result-4)/4 * 100

		return result, percent_error, weights, nodes

	end

	function plot_1b(n)
		transformation(p) = 3 .* (3/2 .* p .+ 3/2).^-.5
		nodes, weights = gausslegendre(n)

		# plot the integration nodes together with the original function
		# define the transformed q(p) function to have it on the [-1,1] interval
		qt(p) = 2 * (1.5*p + 1.5)^-.5
		plot(qt,-1,1,label="True function")
		plot!(xlabel = "integration nodes", ylabel = "corresponding function values")
		scatter!(nodes, transformation(nodes),marker=(2,:circle),color=:black,label="Gauss-Legendre nodes")
		plot!(xlims = (-1, 1), ylims = (0, 16), xticks = -1:.5:1, yticks = 0:4:16)
	end

	function question_1c(n)

		# We draw nodes from the interval [1,4]
		d = Uniform(1,4)
		nodes = rand(d,n)

		# The weights are simply 1/n. We also multiply by the length of the
		# inerval over which we're integrating which is 4 - 1 = 3
		result = sum(q(nodes) .* (3/n))

		# error
		percent_error = (result-4)/4 * 100

		return result, percent_error, nodes

	end

	function plot_1c(n)
		d = Uniform(1,4)
		nodes = rand(d,n)
		nodes = sort(nodes)

		# plot integration nodes and the original function
		plot(q,0,5,label="True function")
		plot!(xlabel = "integration nodes", ylabel = "corresponding function values")
		scatter!(nodes, q(nodes),marker=(2,:circle),color=:black,label="Monte Carlo nodes")
		plot!(xlims = (0, 5), ylims = (1, 2), xticks = 0:1:5, yticks = 1:.25:2)
	end

	function question_1d(n)

		s = SobolSeq(1, 1, 4)
		nodes = hcat([next(s) for i=1:n]...)
		result = sum(q(nodes) .* (3/n))

		percent_error = (result - 4)/4

		return result, percent_error, nodes

	end

	function plot_1d(n)
		s = SobolSeq(1, 1, 4)
		nodes = hcat([next(s) for i=1:n]...)
		nodes = sortrows(nodes)
		plot(q,0,5,leg=false)
		plot!(xlabel = "integration nodes", ylabel = "corresponding function values")
		scatter!(nodes, q(nodes),marker=(2,:circle),color=:black)
		plot!(xlims = (0, 5), ylims = (1, 2), xticks = 0:.5:5, yticks = 1:.25:2)
	end


	function question_2a(n)

		# First, I solve for the equilibrium price as a function of the
		# two thetas using the SymPy package
		p = Sym("p")
		theta_1 = Sym("theta_1")
		theta_2 = Sym("theta_2")

		# Market equilibrium condition
		eqm(p, theta_1, theta_2) = theta_1 * p^-1 + theta_2 * p^-.5 - 2

		# Price function is obtained by finding the roots of the mkt eqm condition
		price = solve(eqm(p, theta_1, theta_2), p)

		# Using the root found by the solve() function, I define the price function
		# I use only the root with the "plus" sign in front of the square root
		# because it generates prices close to equilibrium when thetas are positive

		PriceFct(A::Array{Float64,1}) = 0.5*A[1] + 0.125*A[2]^2 + 0.125*A[2]*sqrt(8.0*A[1] + A[2]^2)

		# Defining nodes and weights keeping in mind that we have 2-dimensional shock
		nodes_h, weights_h = gausshermite(n)
		nodes = Any[]
		push!(nodes,repeat(nodes_h,inner=[1],outer=[10])) # dim1
		push!(nodes,repeat(nodes_h,inner=[10],outer=[1])) # dim2
		weights = kron(weights_h,weights_h)

		# Variance-covariance matrix and its Cholesky decomposition
		sigma = reshape([0.02, 0.01, 0.01, 0.01], 2, 2)
		omega = ctranspose(chol(sigma))

		# We need to perform a change of variable
		v = [nodes[1] nodes[2]] # creating a 2x100 array
		change_of_var = exp(v * omega) # change of variable
		# take exponential of the nodes because theta is distributed as lognormal

		##### EXPECTED PRICE #####

		G = Any[]    # G is a vector of expected prices for each pair of nodes
		for i in 1:n^2
    	push!(G,PriceFct(change_of_var'[:,i]))
		end
		expected_price = dot(weights, G) / pi
		# divide by pi because after the change of variable, pi^(-N/2) remains in front of the sum

		#### VARIANCE ####

		# var calculated by the formula sum((X-E[x])^2 * Pr(X))

		# G contains the variance formula (X-E[x])^2 for each possible state of
		# the world (by state of the world I mean Gauss-Hermite pair of nodes)
		G = Any[]
		for i in 1:n^2
    	push!(G,(PriceFct(change_of_var'[:,i]) -expected_price )^2)
		end
		var = dot(weights, G) / pi
		return expected_price, var, weights, v, PriceFct

	end


	function question_2b(n)

    # Defining the price function (obtained by finding the 2 roots of the equilibrium condition)
    PriceFct(A::Array{Float64,1}) = 0.5*A[1] + 0.125*A[2]^2 + 0.125*A[2]*sqrt(8.0*A[1] + A[2]^2)

    # Sigma is the variance-covariance matrix of the shocks theta
    sigma = reshape([0.02, 0.01, 0.01, 0.01], 2, 2)

    # For Monte Carlo integration, we should choose the nodes randomly
		#from the multivariate lognormal distribution of thetas
		# (assuming they're not independently distributed)
    dist = MvLogNormal(sigma)
    nodes_mc = rand(dist,n^2) #2x100 #we choose 100 pairs of thetas

    weights = (1/n^2) .* ones(n^2) #100-element array #weight for each node will correspond to 1/n^2

		#### EXPECTED PRICE ####

    G = Any[]    # G contains values of the price function obtained by plugging in nodes_mc for values of theta
    for i in 1:n^2
        push!(G,PriceFct(nodes_mc[:,i]))
    end
    # expected price is just the dot product of values of price function and their associated weights
    expected_price = dot(weights, G)

		#### EX-ANTE VARIANCE ####

    # The variance is calculated by the formula 1/n * sum((X - E(X))^2)
    G = Any[]
    for i in 1:n^2
        push!(G,(PriceFct(nodes_mc[:,i]) -expected_price )^2)
    end
    var =  dot(weights, G)

		return expected_price, var, weights, nodes_mc, PriceFct

	end

	# function to run all questions
	function runall(n)
		println("Running all questions of HW-integration:")
		println("Results of question 1:")
		q1b = question_1b(n)
		res = q1b[1]
		per = q1b[2]
		println("The estimated change in consumer's surplus using the Gauss-Legendre method and $n nodes is $res. The error is $per%.")
		q1c = question_1c(n)
		res2 = q1c[1]
		per2 = q1c[2]
		println("The estimated change in consumer's surplus using the Monte Carlo method and $n nodes is $res2. The error is $per2%.")
		q1d = question_1d(n)
		res = q1d[1]
		per = q1d[2]
		println("The estimated change in consumer's surplus using the Quasi Monte Carlo method and $n nodes is $res. The error is $per%.")
		println("results of question 2:")
		q2a = question_2a(n)
		exp = q2a[1]
		var = q2a[2]
		println("The expected price using the Gauss-Hermite integration and $n nodes is $exp.")
		println("The ex-ante variance of price using the Gauss-Hermite integration and $n nodes is $var.")
		q2b = question_2b(n)
		exp = q2b[1]
		var = q2b[2]
		println("The expected price using Monte Carlo method and $n nodes is $exp.")
    println("The ex-ante variance using Monte Carlo method and $n nodes is $var.")
		println("End of HW-integration")
		plot_1b(n)
		plot_1c(n)
		plot_1d(n)
	end


end
