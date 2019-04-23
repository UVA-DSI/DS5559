#! /bin/bash

# Transitions: none/fade/slide/convex/concave/zoom

echo "$1.md -> $1.html"
#pandoc -t revealjs -s -o $1.html $1.md -V revealjs-url=../../lib/reveal
pandoc --mathjax -t revealjs -s -o $1.html $1.md \
    -V revealjs-url=https://revealjs.com \
    -V theme=simple \
    -V transition=none \
    -V controls=true \
    -V progress=true \
    -V slideNumber=true \
    --css default-pandoc-slides.css \
    --slide-level=1    
