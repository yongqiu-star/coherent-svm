LATEXMK ?= latexmk
TARGET = svm_gpu_technical_report

.PHONY: all clean distclean

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex sections/*.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(TARGET).tex

clean:
	$(LATEXMK) -c $(TARGET).tex

distclean:
	$(LATEXMK) -C $(TARGET).tex
