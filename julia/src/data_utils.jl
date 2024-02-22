using PyCall

# import python module sklearn
@pyimport sklearn.datasets as datasets


# we use a dataset from the python module sklearn
function download_data(path="julia/data/cancer_data.csv")
	# get the data from the python module
	data, target = datasets.load_breast_cancer(return_X_y=true)

	# save the data to a file
	writedlm(path, hcat(target, data), ';')
end