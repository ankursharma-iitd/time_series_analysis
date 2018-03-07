import csv

def findcenters(csvfile):
    with open(csvfile) as file:
        reader = csv.reader(file)
        for line in reader:
            if line[-1] == '40':
                print(line)

file = 'data/original/mandis.csv'

findcenters(file)