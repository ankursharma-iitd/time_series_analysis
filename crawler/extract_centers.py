import csv

def findcenters(csvfile):
    temp = []
    with open(csvfile) as file:
        reader = csv.reader(file)
        for line in reader:
            if line[-1] == '40':
                temp.append(line[1])
    return temp

file = 'data/original/mandis.csv'

mandis = findcenters(file)
print(mandis)