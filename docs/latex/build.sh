#!/bin/bash
# Build the OpenSG tutorial PDF.  Run from this folder:  bash build.sh
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error opensg_tutorial.tex > build1.log
pdflatex -interaction=nonstopmode -halt-on-error opensg_tutorial.tex > build2.log
rm -f build1.log build2.log opensg_tutorial.aux opensg_tutorial.toc \
      opensg_tutorial.out opensg_tutorial.lof opensg_tutorial.lot
echo "wrote opensg_tutorial.pdf"
