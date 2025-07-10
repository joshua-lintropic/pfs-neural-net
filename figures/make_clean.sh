#!/bin/bash
for prefix in A B C D E L; do
    for ext in png txt; do
        echo "Processing type: ${prefix} with extension: $ext"
        files=( $(ls ${prefix}_*.${ext} 2>/dev/null | sort) )
        
        if [ ${#files[@]} -gt 1 ]; then
            to_remove=( "${files[@]:0:${#files[@]}-1}" )
            echo "Removing: ${to_remove[@]}"
            rm "${to_remove[@]}"
        else
            echo "Not enough ${ext} files to remove for type $prefix."
        fi
    done
done