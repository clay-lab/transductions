from sys import argv
import csv
import pandas as pd


def read_file():
    filename = argv[1]
    with open(filename, newline='') as rfile:
        reader = csv.reader(rfile, delimiter='\t', lineterminator='\n')
        newreader = [line for line in reader]
        nreader = []
        for line in newreader:
            newline = []
            for sentence in line:
                splitsent = sentence.split(" ")
                for word in range(len(splitsent)):
                    if sentence == 'source target prediction':
                        splitsent = sentence      
                    elif splitsent[word] == '<eos>':
                        splitsent = splitsent[:word]
                        break
                newline.append(splitsent)
            nreader.append(newline)
        return nreader



def write_file():
    with open("newresults.txt", mode='w') as new_file:
        out_writer = csv.writer(new_file, delimiter='\t', lineterminator='\n', quotechar='/')
        newreader = read_file()
        for newline in newreader:
            if len(newline) < 3:
                out_writer.writerow(["target", "prediction"])
            else:
                out_writer.writerow([' '.join(newline[1]), ' '.join(newline[2])])

    with open("newresults.txt", 'r') as results_file, open("lengthresults.txt", 'w') as length_file, open("accuracyresults.txt", 'w') as acc_file:
        results_reader = csv.reader(results_file, delimiter='\t')
        len_writer = csv.writer(length_file, delimiter='\t', lineterminator='\n', quotechar='/')
        acc_writer = csv.writer(acc_file, delimiter='\t', lineterminator='\n', quotechar='/')
        len_writer.writerow(["target" , "prediction"])
        for line in results_reader:
            if len(line[0].split(" ")) != len(line[1].split(" ")):
                len_writer.writerow(line)
            if line[0] != line[1]:
                acc_writer.writerow(line)
                 
write_file()