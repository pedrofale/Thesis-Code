# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:16:02 2017

This script transforms the data from the TCGA-BRCA project into a
usable pattern matrix and a corresponding label vector

@author: pedro
"""

import csv
import numpy as np
import os
import gzip

"Function to search for a file in directory and subdirectories"
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

with open('/home/pedro/IST/IIEEC/TCGA/File_metadata.csv', 'r') as fp:
    reader = csv.reader(fp, delimiter=',', quotechar='"')
    next(reader, None)  # skip the headers
    metadata = [row for row in reader]

# Transform this into an array
metadata_array = np.asarray(metadata)
num_samples = metadata_array.shape[0]

####################################################################
# LABELS VECTOR                                                    #
####################################################################
# We only want the sample types
labels = metadata_array[:, 2]

# Use integers as class labels
for i in range(labels.size):
    if labels[i] == "Solid Tissue Normal":
        labels[i] = 0
    elif labels[i] == "Primary Tumor":
        labels[i] = 1
    elif labels[i] == "Metastatic":
        labels[i] = 2
        
# Describe the data
unique, counts = np.unique(labels, return_counts=True)

print "Total: ", labels.size, "\tsamples"
print "Sample types:"
print "\t", counts[0], "\tnormal tissue samples"
print "\t", counts[1], "\tprimary tumor samples"
print "\t", counts[2], "\tmetastatic samples"

####################################################################
# PATTERNS MATRIX                                                  #
####################################################################

# Get only the filenames
filenames = metadata_array[:, 5]

# Initialize the patterns matrix
patterns = np.zeros((num_samples, 60483))

i = 0
for filename in filenames:
    # Open the file <filename>
    print "Searching for file ", filename, "..."
    curr_file = find(filename, '/home/pedro/IST/IIEEC/TCGA/brca_data/')
    print "done."
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
    print "done."        
    
    i = i + 1
    