# Not implemented yet
function load_and_visualize(path="julia/data/cancer_data.csv")
	if !isfile(path)
		download_data(path)
	end

	# load the data from the file
	data = readdlm(path, ';')

	# return the data
	data[:, 2:end], data[:, 1]
end

function A2()
	
end