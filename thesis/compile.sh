#!/bin/sh

# This script will compile the complete thesis and will remove all the uncessesary
# files after the compilation process is done. It will place the resulting pdf
# in the "output" directory. Which will never be included in the Git repository.

# Remove the old output directory and pdf.
rm -r output
rm master_thesis_joeri_hermans.pdf
# Create a new output file.
mkdir output
# Compile the thesis and write the output files.
pdflatex thesis
makeindex thesis.nlo -s nomencl.ist -o thesis.nls
bibtex thesis
pdflatex thesis
pdflatex thesis
# Move all files to the output directory.
mv thesis.aux output
mv thesis.log output
mv thesis.nlo output
mv thesis.pdf master_thesis_joeri_hermans.pdf
mv thesis.out output
mv thesis.toc output
mv thesis.blg output
mv thesis.bbl output
mv thesis.bcf output
mv thesis-blx.bib output
mv thesis.dvi output
mv thesis.run.xml output
mv thesis.fdb_latexmk output
mv thesis.idx output
mv thesis.ilg output
mv thesis.ind output
mv thesis.fls output
mv thesis.nls output
mv bibliography.log output
