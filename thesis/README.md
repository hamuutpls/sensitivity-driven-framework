# Thesis LaTeX sources (literature review update)

Files to copy into the thesis LaTeX project (MastersDoctoralThesis template):

- `Chapters/Chapter3v2.tex`: the literature review, which is the survey paper
  "Quantization and Compression Techniques for Low-Latency LLM Deployment" word for word,
  with its 7 tables and Fig. 1.
- `references.bib`: the 45 existing entries plus the survey's references [1], [2], [27], [28], [29] (50 in total).
- `Figures/lr-fig1-pipeline.png`: Fig. 1 of the survey.
- `Figures/adu-logo.jpg`: university logo from the report template, used on the title page.
- `main.tex`: the existing main file with the university logo added to the title page.

Build: pdflatex, biber, pdflatex, pdflatex.
