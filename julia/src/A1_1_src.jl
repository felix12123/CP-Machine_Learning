# using MLJ


function A1_data(path="julia/data/cancer_data.csv")
	if !isfile(path)
		download_data(path)
	end

	# load the data from the file
	data = readdlm(path, ';')

	# return the data
	data[:, 2:end], data[:, 1]
end

function improve_guess(X::Matrix, y::Vector, p0::Vector{<:Real}, as::NTuple{2, <:Real}, bs::NTuple{2, <:Real}, cs::NTuple{2, <:Real}, steps::Int=21)::Vector{Float64}
	# class_rule(x::Matrix, p) = x[:, 1] .* p[1] .+ x[:, 2] .* p[2] .+ p[3] .> 0.5
	# class_rule(x::Vector, p) = x[1] .* p[1] .+ x[2] .* p[2] .+ p[3] .> 0.5
	class_rule(x::Matrix, p) = hcat(ones(size(x, 1)), x) * p .> 0.5

	# create slightly different values for the parameters
	p1s = p0[1] .+ collect(range(as..., step=round((as[2] - as[1]) / (steps-1), sigdigits=3)))
	p2s = p0[2] .+ collect(range(bs..., step=round((bs[2] - bs[1]) / (steps-1), sigdigits=3)))
	p3s = p0[3] .+ collect(range(cs..., step=round((cs[2] - cs[1]) / (steps-1), sigdigits=3)))

	# ensure initial parameters are part of the search space to not miss the best parameters
	pushfirst!(p1s, p0[1])
	pushfirst!(p2s, p0[2])
	pushfirst!(p3s, p0[3])
	unique!(p1s)
	unique!(p2s)
	unique!(p3s)

	ps = Iterators.product(p1s, p2s, p3s) |> collect .|> collect

	# calculate the accuracy for each parameter combination
	accs = zeros(length(ps))
	for i in eachindex(ps)
		p0 = ps[i]
		# println(p)
		y_pred = class_rule(X, p0)
		accs[i] = sum(y .== y_pred) / length(y)
	end

	# return the maximal accuracy and the corresponding parameters
	best = argmax(accs) # minimal accuracy at "best" index
	best_res = filter(x -> x[2] == accs[best], enumerate(accs) |> collect) #get all indices with the same accuracy
	sort!(best_res, by=x->sum(length.(string.(ps[x[1]])))) #sort by "complexity" of the parameters
	best_res = best_res[1][1] # pick lesat complex parameter to reduce unneccecary amount of precision
	return ps[best_res] |> collect
end
