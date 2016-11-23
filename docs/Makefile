# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: help clean tutorials html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest gettext

default: dirhtml

content: *.ipynb user/*.ipynb
	ls *.ipynb | xargs jupyter nbconvert --to rst
	ls user/*.ipynb | xargs jupyter nbconvert --to rst

tutorials: _static/notebooks/*.ipynb
	cd tutorials; ls ../_static/notebooks/*.ipynb | xargs jupyter nbconvert --to rst

clean:
	-rm -rf $(BUILDDIR)/*

dirhtml: tutorials content
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."
