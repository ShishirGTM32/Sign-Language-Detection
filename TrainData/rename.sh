dir="/home/decoy/Downloads/data/train/24"
counter=1
for file in "$dir"/*.png; do
    base_name=$(basename "$file")
    new_name="Y${counter}.jpg"
    mv "$file" "$dir/$new_name"
    ((counter++))
done
