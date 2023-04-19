#!/bin/bash

echo "Run handin 3 Liz van der Kamp (s2135752)"

echo "Download files for in report..."
if [ ! -e satgals_m11.txt ]; then
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m11.txt
fi
echo "Download files for in report..."
if [ ! -e satgals_m12.txt ]; then
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m12.txt
fi
echo "Download files for in report..."
if [ ! -e satgals_m13.txt ]; then
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m13.txt
fi
echo "Download files for in report..."
if [ ! -e satgals_m14.txt ]; then
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m14.txt
fi
echo "Download files for in report..."
if [ ! -e satgals_m15.txt ]; then
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m15.txt
fi

# Script that returns a txt file
echo "Run the first script ..."
python3 NURHW3LizQ1.py 


echo "Generating the pdf"

pdflatex SolutionsHW3Liz.tex
bibtex SolutionsHW3Liz.aux
pdflatex SolutionsHW3Liz.tex
pdflatex SolutionsHW3Liz.tex
