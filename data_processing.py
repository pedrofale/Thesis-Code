# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:16:02 2017

This script transforms the data from the TCGA-BRCA project into a
usable pattern matrix and a corresponding label vector

@author: pedro
"""

import sys
import csv
import numpy as np
import os
import gzip

if len(sys.argv) != 4:
    print "Invalid arguments."
    sys.exit()

####################################################################
# DATA INFO                                                        
num_genes = 60483
metadata_filename = sys.argv[1] #'/home/pedro/IST/IIEEC/TCGA/File_metadata.csv'
data_path = sys.argv[2] #'/home/pedro/IST/IIEEC/TCGA/brca_data/'
matrices_directory = sys.argv[3]
####################################################################

"Function to search for a file in directory and subdirectories"
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

# Convert the .txt to .csv to make the process easier
in_txt = csv.reader(open(metadata_filename, "rb"), delimiter = '\t')
metadata_filename_csv = r"metadata_csv.csv"
metadata_csv = csv.writer(open(metadata_filename_csv, 'wb'))
metadata_csv.writerows(in_txt)

with open(metadata_filename, 'r') as fp:
    reader = csv.reader(fp, delimiter='\t', quotechar='"')
    next(reader, None)  # skip the headers
    metadata = [row for row in reader]

# Transform this into an array
metadata_array = np.asarray(metadata)
num_samples = metadata_array.shape[0]

####################################################################
# LABELS VECTOR                                                    #
####################################################################
# We only want the sample types
labels_str = metadata_array[:, 2]
labels = np.zeros((num_samples, 1))
# Use integers as class labels
for i in range(labels_str.size):
    if labels_str[i] == "Solid Tissue Normal":
        labels[i] = 0
    elif labels_str[i] == "Primary Tumor":
        labels[i] = 1
    elif labels_str[i] == "Metastatic":
        labels[i] = 2

####################################################################
# PATTERNS MATRIX                                                  #
####################################################################

# Get only the filenames
filenames = metadata_array[:, 5]

# Initialize the patterns matrix
patterns = np.zeros((num_samples, num_genes))

i = 0
for filename in filenames:
    # Open the file <filename>
    print "Searching for file ", filename, "..."
    curr_file = find(filename, data_path)
    print "found."
    with gzip.open(curr_file, 'rb') as f:
        file_content = f.read()
        
    # Transform the data into a vector
    file_lines = file_content.splitlines(True)
    
    print "Inserting sample in pattern matrix..."
    j = 0
    # Put the vector in the patterns matrix
    for line in file_lines:
        line_vector = line.strip().split('\t')
        patterns[i, j] = line_vector[1]
        j = j + 1        
    print "inserted."        
    
    i = i + 1
    
# Remove genes which have 0 expression value for all samples
zero_genes = np.where((~patterns.any(axis=0)))[0]
print "Total of ", patterns.shape[1], "\tgenes"
print "Removing ", zero_genes.size, "\tgenes without expression for any sample..."
patterns = np.delete(patterns, zero_genes, axis = 1)
print "removed."

####################################################################
# DESCRIBE THE DATA                                                #
####################################################################
unique, counts = np.unique(labels, return_counts=True)

print "Total: ", labels.size, "\tsamples"
print "Sample types:"
print "\t", counts[0], "\tnormal tissue samples"
print "\t", counts[1], "\tprimary tumor samples"
print "\t", counts[2], "\tmetastatic samples"
    
####################################################################
# SAVE THE DATA                                                    #
####################################################################

print "Saving data arrays into " + matrices_directory + "..."
np.save(matrices_directory + 'brca-labels.npy', labels)    
np.save(matrices_directory + 'brca-patterns.npy', patterns)
print "Succesfully saved the numpy arrays brca-labels.npy and brca-labels.npy."   

# To read later:
# labels = np.load(matrices_directory + 'brca-labels.npy')    
# patterns = np.load(matrices_directory + 'brca-patterns.npy')    