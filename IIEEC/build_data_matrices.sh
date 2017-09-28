# Use this script to transform the BRCA files downloaded from the GDC database into ready-to-use numpy arrays
# This assumes the data was downloaded using a manifest file which you saved in $manifest_directory as $manifest_filename.
# The saved numpy arrays are two:
# -- brca-labels.npy, which contains the labels for each sample (normal tissue:0, primary tumo:1, metastatic:2)
# -- brca-patterns.npy, which is a NUM_SAMPLES-by-NUM_GENES matrix
# Each line in brca-labels corresponds to a sample which is on the same line in the brca-patterns matrix.

manifest_directory='/home/pedro/IST/IIEEC/TCGA/'
manifest_filename='gdc_manifest_brca_htseqfpkm.tsv'

metadata_directory='/home/pedro/IST/IIEEC/TCGA/'
datafiles_directory='/home/pedro/IST/IIEEC/TCGA/brca_data/'

datamatrices_directory='/home/pedro/IST/IIEEC/TCGA/'

# Create a payload.txt file containing a query for GDC, according to https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/#request-parameters
python uuid_to_barcode.py $manifest_directory $manifest_filename

# Query GDC to get a file file_metadata.txt containing the information about each sample 
curl --request POST --header "Content-Type: application/json" --data @$manifest_directory/payload.txt "https://gdc-api.nci.nih.gov/files" > $metadata_directory/file_metadata.txt

# Create the labels and patterns matrices
python data_processing.py $metadata_directory/file_metadata.txt $datafiles_directory $datamatrices_directory
