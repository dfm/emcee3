LATEX       = pdflatex
BASH        = bash -c
ECHO        = echo
RM          = rm -rf
TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out
CHECK_RERUN =

NAME = ms

all: ${NAME}.pdf

${NAME}.pdf: ${NAME}.tex *.bib
	${LATEX} ${NAME}
	bibtex ${NAME}
	${LATEX} ${NAME}
	( grep Rerun ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."
	( grep Rerun ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."

clean:
	${RM} $(foreach suff, ${TMP_SUFFS}, ${NAME}.${suff})
