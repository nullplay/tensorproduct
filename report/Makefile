PAPER=report
BIB = refs.bib confs.bib
# Halt on errors instead of going to the shell
FLAGS = -halt-on-error -interaction=batchmode
TEX = $(wildcard *.tex)
PLOTS = $(shell grep -o "plots/.*\.pdf" floats.tex)
REVNUM = $(shell git log --pretty=oneline | wc -l | sed 's/ //g')

# dsm: On macOS, use GNU sed instead of BSD sed (why can't systems converge on such a stupid thing...)
# hrlee: I do not understand, but sed to remove \OK{...} does not work with the -r command on Linux??
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	SED := gsed
	READFILE_OPT := -r
else
	SED := sed
	READFILE_OPT :=
endif

default: paper 

paper: $(PAPER).pdf

TEXFILES=$(wildcard ?_*.tex) floats.tex

# explicit rule for the paper alone
$(PAPER).pdf: $(TEX) IEEEtran.cls
	pdflatex $(FLAGS) $(PAPER) || pdflatex $(PAPER)
	#bibtex -min-crossrefs=30000 $(PAPER)
	pdflatex $(FLAGS) $(PAPER)
	pdflatex $(FLAGS) $(PAPER)
	pdflatex $(FLAGS) $(PAPER)
	ps2pdf -dPDFSETTINGS=/prepress -dCompatibilityLevel=1.4 -dEmbedAllFonts=true -dSubsetFonts=true -r600 $(PAPER).pdf $(PAPER)_opt.pdf
	# Preserve metadata
	#pdftk $(PAPER).pdf dump_data output metadata.txt
	#pdftk $(PAPER)_opt.pdf update_info metadata.txt output $(PAPER).pdf
	#cp $(PAPER).pdf $(PAPER)_opt.pdf
	#qpdf --linearize $(PAPER)_opt.pdf $(PAPER).pdf

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPER).pdf $(DIFF_NAME).pdf

.PHONY: paper clean
