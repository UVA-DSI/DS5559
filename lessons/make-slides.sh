#! /bin/bash

echo "$1.md -> $1.html"
#pandoc -t revealjs -s -o $1.html $1.md -V revealjs-url=../../lib/reveal
pandoc -t revealjs -s -o $1.html $1.md -V revealjs-url=https://revealjs.com