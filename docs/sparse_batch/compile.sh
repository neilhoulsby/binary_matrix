#!/bin/bash

pdflatex binMatFacVB.tex
bibtex binMatFacVB
pdflatex binMatFacVB.tex
bibtex binMatFacVB
pdflatex binMatFacVB.tex
evince binMatFacVB.pdf
