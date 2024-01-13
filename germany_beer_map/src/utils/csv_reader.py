import csv

def read_csv_file(csvfile):
	# Open the csv file
	with open(csvfile, 'r') as file:
		# Create a csv reader
		reader = csv.DictReader(file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
		# Convert the csv reader to a list and return it
		data = list(reader)
	return data