#!/bin/bash
# For each file type prefix A, B, C, D, E in the current directory
for prefix in A B C D E; do
    echo "Processing type: $prefix"
    # List matching files, sort lexically, and store them in an array
    files=( $(ls ${prefix}_*.png 2>/dev/null | sort) )
    
    # Only proceed if there is more than one file
    if [ ${#files[@]} -gt 1 ]; then
        # All files except the last (lexicographically greatest)
        to_remove=( "${files[@]:0:${#files[@]}-1}" )
        echo "Removing: ${to_remove[@]}"
        rm "${to_remove[@]}"
    else
        echo "Not enough files to remove for type $prefix."
    fi
done
