PAPER = paper

# Make pdflatex/bibtex find uai2026.cls (and any future template assets)
# without copying them out of the official template directory.
export TEXINPUTS := .:./uai2026-template/:$(TEXINPUTS)
export BIBINPUTS := .:./uai2026-template/:$(BIBINPUTS)

.PHONY: all clean distclean

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	pdflatex -interaction=nonstopmode $(PAPER)
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER)

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).bbl $(PAPER).blg $(PAPER).toc $(PAPER).synctex.gz

distclean: clean
	rm -f $(PAPER).pdf
