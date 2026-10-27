#!/bin/bash

for f  in *.json; do
	dir="${f%.json}"
	mkdir -p "$dir"
	mv "$f" "$dir"
done
