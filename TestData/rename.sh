dir="."
counter=1
for file in "$dir"/*.jpg; do
    base_name=$(basename "$file")
    new_name="A${counter}.jpg"
    mv "$file" "$dir/$new_name"
    ((counter++))
done

