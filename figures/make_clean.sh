#!/bin/bash

for prefix in A B C D E L; do
    rm prefix.*
done

for prefix in A B C D E L; do
    for ext in png txt; do
        echo "Processing type: ${prefix} with extension: ${ext}"
        files=( $(ls ${prefix}_*.${ext} 2>/dev/null | sort) )
        
        if [ ${#files[@]} -gt 1 ]; then
            # remove all but the last one
            to_remove=( "${files[@]:0:${#files[@]}-1}" )
            echo "Removing: ${to_remove[@]}"
            rm -- "${to_remove[@]}"
        else
            echo "Not enough ${ext} files to remove for type ${prefix}."
        fi

        # ==== Rename step ====
        remaining=( $(ls ${prefix}_*.${ext} 2>/dev/null | sort) )
        if [ ${#remaining[@]} -eq 1 ]; then
            src="${remaining[0]}"
            dst="${prefix}.${ext}"
            echo "Renaming ${src} → ${dst}"
            mv -- "${src}" "${dst}"
        else
            echo "Skipping rename for ${prefix}.${ext}: found ${#remaining[@]} files: ${remaining[*]}"
        fi

    done
done

