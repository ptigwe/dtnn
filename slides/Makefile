build: slides.html

slides.html: slides.md
	pandoc --standalone -s --mathjax -i -t revealjs -o output/$@ slides.md

slides.pdf: slides.md
	pandoc --standalone -t beamer -o output/$@ $<

