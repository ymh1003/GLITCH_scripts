#!/bin/bash  --login
FILE_1=""
FILE_2=""

# Make sure that GNU version of sed is used

# export PATH="/opt/homebrew/opt/gnu-sed/libexec/gnubin:$PATH"
# export PATH=$PATH:/Users/yumenghe/Library/Python/3.10/bin
# export PATH=$PATH:/Applications/UltiMaker\ Cura.app/Contents/MacOS

output=$(stlinfo "$1")
height=$(echo $output | awk -F'Z: ' '{print $2}' | awk '{print $1}')
center=$(echo "$height" | awk '{printf "%.1f", $1 / 2}')

pj3d test1 create "Voron 0" /path/to/voron_print_settings.txt
pj3d test1 add "$1"
pj3d test1 pack
pj3d test1 print

pj3d test2 create "Voron 0" /path/to/voron_print_settings.txt
pj3d test2 add "$1"
pj3d test2 pack
pj3d test2 printpart --rotxyz "$2","$3","$4" --all

for file in "test1.job"/*.gcode; do
    if [ -f "$file" ]; then
        FILE_1="$file"
        break
    fi
done

for file in "test2.job"/*.gcode; do
    if [ -f "$file" ]; then
        FILE_2="$file"
        break
    fi
done

scp "$FILE_1" "$FILE_2" yhe54@capella:Gcode-Checking-Project/gcode/
rm -r test1.job
rm -r test2.job

ssh -t capella "conda activate Gcode-comparer; cd Gcode-Checking-Project; python gcode_comp_Z.py 'gcode/$(basename $FILE_1)' 'gcode/$(basename $FILE_2)' '$(echo $2 | bc -l)' '$(echo $3 | bc -l)' '$(echo $4 | bc -l)' '$center' -g '$(echo $5 | bc -l)' -c '$(echo $6 | bc -l)' '$(echo $7 | bc -l)' --cubesize '$(echo ${8} | bc -l)' '$(echo ${9} | bc -l)' '$(echo ${10} | bc -l)' -p '$(echo ${11} | bc -l)'"
scp yhe54@capella:Gcode-Checking-Project/heatmap.pcd /path/to/store/heatmap/