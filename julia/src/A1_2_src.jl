using LinearAlgebra
using StaticArrays

function A1_data(path="julia/data/cancer_data.csv")
	if !isfile(path)
		download_data(path)
	end

	# load the data from the file
	data = readdlm(path, ';')

	# return the data
	data[:, 2:end], data[:, 1]
end


# train a model based on n dimensions of the data
function learn_nparams_bad(X, y, n_params::Int=0; steps=5000)
	n_params = n_params == 0 ? size(X, 2) : n_params
	println("Training a model based on ", n_params, " dimensions of the data...")
	
	function logistic_regression(X, y, α, num_iters)
		sigmoid(x) = 1 / (1 + exp(-x))
		m, n = size(X)
		θ = zeros(n)
		for i in 1:num_iters
			h = sigmoid.(X * θ)
			gradient = (1/m) .* X' * (((h .- y)))
			θ = θ - α * gradient
		end
		return θ
	end
	
	
	X = X[:, 1:n_params]

	# add a column of ones to the feature matrix for the bias term
	X = hcat(ones(size(X, 1)), X)

	# Set the learning rate and the number of iterations
	α = 0.01

	# Train the model
	θ = logistic_regression(X, y, α, steps)

	return θ

	# accuracy is probably terrible, because the loss function is not suitable for yes/no data.
	# it is better suited for continuous data.
end


# train a model based on n dimensions of the data
function learn_nparams(X, y, n_params::Int=0; steps=5000, α=0.01)
	
	function logistic_regression(X, y, α, num_iters)
		# sigmoid(x) = 1 / (1 + exp(-x))
		gradient(X, y, θ) = (-2/size(X, 1)) * X' * ((X * θ .> 0.5) .- y)
		θ = zeros(size(X, 2))
		for _ in 1:num_iters
			θ = θ + α * gradient(X, y, θ)
		end
		return θ
	end
	
	
	
	n_params = n_params == 0 ? size(X, 2) : n_params
	# println("Training a model based on ", n_params, " dimensions of the data...")
	X = X[:, 1:n_params]

	# add a column of ones to the feature matrix for the bias term
	X = hcat(ones(size(X, 1)), X)

	# Train the model
	θ = logistic_regression(X, y, α, steps)

	return θ

	# accuracy is probably terrible, because the loss function is not suitable for yes/no data.
	# it is better suited for continuous data.
end

