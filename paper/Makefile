# Build the paper. Works with either engine:
#
#   make            latexmk if a full TeX distribution is installed
#   make tectonic   single-binary Tectonic, which fetches only what it needs
#   make clean
#
# Two passes plus bibtex are needed for \cite and \ref to resolve; latexmk and
# tectonic both handle that internally, so neither target loops manually.

PAPER = main

.PHONY: all tectonic clean watch

all:
	latexmk -pdf -halt-on-error $(PAPER).tex

tectonic:
	tectonic -X compile $(PAPER).tex

watch:
	latexmk -pdf -pvc -halt-on-error $(PAPER).tex

clean:
	latexmk -C 2>/dev/null || true
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log \
	      $(PAPER).out $(PAPER).fls $(PAPER).fdb_latexmk $(PAPER).synctex.gz
