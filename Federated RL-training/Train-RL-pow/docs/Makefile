# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
PICTURES      = $(wildcard _static/*.svg)
PNGS          = $(subst svg,png,$(PICTURES))


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

_static/%.png: _static/%.svg
	inkscape $< -o $@


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf latexpdfja text man texinfo info gettext changes xml pseudoxml linkcheck doctest coverage clean: Makefile $(PNGS)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
