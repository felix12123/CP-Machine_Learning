using Plots
using DelimitedFiles

include("../src/data_utils.jl")
include("../src/A1_1_src.jl")
include("../src/A1_2_src.jl")


function plot_results(X::Matrix, y::Vector, p::Vector{<:Real})::Plots.Plot
	if size(X, 2) != length(p)
		X = X[:, 1:length(p)-1]
	end
	class_rule(x::Vector, p) = sum(vcat([1], x) .* p) .> 0.5

	x1 = [X[i, 1] for i in eachindex(y) if y[i] == 1]
	y1 = [X[i, 2] for i in eachindex(y) if y[i] == 1]
	x0 = [X[i, 1] for i in eachindex(y) if y[i] == 0]
	y0 = [X[i, 2] for i in eachindex(y) if y[i] == 0]
		
	xs = minimum(X[:, 1]):0.05:maximum(X[:, 1])
	ys = minimum(X[:, 2]):0.05:maximum(X[:, 2])
	
	hm = heatmap(xs, ys, (x, y) -> class_rule([x, y], p), color=:grays, alpha=0.5, legend=false, dpi=300)
	scatter!(x1, y1, color="red", label="")
	scatter!(x0, y0, color="blue", label="")
	
	return hm
end


function A1_1()
	println("=============== A1_1 ===============")
	# import_data()
	X, y = A1_data()
	X = X[:, 1:2]

	class_rule(x::Matrix, p) = x[:, 1] .* p[2] .+ x[:, 2] .* p[3] .+ p[1] .> 0.5
	class_rule(x::Vector, p) = x[1] .* p[2] .+ x[2] .* p[3] .+ p[1] .> 0.5

	# define the parameters for a first guess
	p0 = [65, -3, -1]


	# calculate the accuracy
	accuracy(p::Vector{<:Real}) = sum(y .== class_rule(X, p)) / length(y)

	# print the accuracy
	println("Accuracy for the first guess $p0: ", accuracy(p0), "\n")

	# improve the guess
	println("we can now improve the guess by small variations of the initial guess")
	p = improve_guess(X, y, p0, (-2, 2), (-1.0, 0.5), (-0.5, 0.5), 20)
	println("improved guess: ", p, " with accuracy: ", accuracy(p), "\n")
	
	println("multiple repeated improvements:")
	# further improvement of the guess is not possible
	for s in [0.5, 0.2, 0.1, 0.05, 0.01, 0.005]
		p = improve_guess(X, y, p, (-s, s), (-s, s), (-s, s))
	end
	println("further improved guess: ", p, " with accuracy: ", accuracy(p))
	

	# plot the data	
	plt = plot_results(X, y, p)
	savefig(plt, "julia/media/A1_1.png")
end


function A1_2()
	println("\n=============== A1_2 ===============")
	# get data from csv
	X, y = A1_data()

	
	function accuracy(p, X=X, y=y)
		function predict(X, θ)
			if size(X, 2) != length(θ)
				X = hcat(ones(size(X, 1)), X)
			end
			probability = X * θ
			return [p >= 0.5 ? 1 : 0 for p in probability]
		end
		if size(X, 2) != length(p)
			X = X[:, 1:length(p)-1]
		end
		correct = sum(y .== predict(X, p))
		return correct / length(y)
	end
	
	

	println("squared diff Ansatz:")
	p_good = learn_nparams(X, y, 2, steps=100_000, α=0.01)
	acc = accuracy(p_good, X)
	println("Parameters returned by gradient descent: ", p_good)
	println("Accuracy: ", acc)

	plt = plot_results(X, y, p_good)
	savefig(plt, "julia/media/A1_2_n2_good.png")

	ns = 2:30
	accs = similar(ns, Float64)

	Threads.@threads for i in eachindex(ns)
		p = learn_nparams(X, y, ns[i], steps=10_000, α=0.01)
		acc = accuracy(p, X)
		accs[i] = acc
	end
	plt = scatter(ns, accs, label="Accuracy", xlabel="Number of dimensions", ylabel="Accuracy", dpi=300)
	savefig(plt, "julia/media/A1_2_n_acc.png")
	println("best accuracy: ", maximum(accs), " for ", ns[argmax(accs)], " dimensions")
end


function A1()
	A1_1()
	A1_2()
end

