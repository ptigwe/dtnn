build: slides.html

slides.html: slides.md
	pandoc --standalone -s --mathjax -t revealjs -o output/$@ slides.md
	cp output/slides.html output/reveal.js/.

slides.pdf: slides.md
	pandoc --standalone -t beamer -o output/$@ $<

