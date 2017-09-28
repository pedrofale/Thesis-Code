# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:58:40 2017

@author: pedro
"""

import sys
import csv
import numpy as np

if len(sys.argv) != 3:
    print "Invalid arguments."
    sys.exit()

manifest_dir = sys.argv[1]
manifest_path = manifest_dir + sys.argv[2]

with open(manifest_path, 'r') as fp:
    reader = csv.reader(fp, delimiter='\t', quotechar='"')
    next(reader, None)  # skip the headers
    metadata = [row for row in reader]

# Transform this into an array
metadata_array = np.asarray(metadata)

num_files = metadata_array.shape[0]

# The UUID column is the first one
uuid = metadata_array[:, 0]
uuid = np.asarray(['"' + s + '"' for s in uuid])

uuid_str = ','.join(x for x in uuid)

# Build the GDC query
part1 = '{"filters":{"op":"in","content":{"field":"files.file_id","value":[ '
part2 = '] }},"format":"TSV","fields":"file_id,file_name,cases.submitter_id,cases.case_id,data_category,data_type,cases.samples.tumor_descriptor,cases.samples.tissue_type,cases.samples.sample_type,cases.samples.submitter_id,cases.samples.sample_id,cases.samples.portions.analytes.aliquots.aliquot_id,cases.samples.portions.analytes.aliquots.submitter_id","size":"' + str(num_files) + '"} '
query = part1 + uuid_str + part2 

payload_file = open(manifest_dir + "payload.txt", 'w')
payload_file.write(query)
payload_file.close()