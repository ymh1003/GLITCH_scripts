# GLITCH_scripts

To run comp_script.sh, which takes a STL file as input and visualizes the generated heatmap on local machine:

    chmod +x comp_script.sh
    ./comp_script.sh [/path/to/stlfile] [rot_x] [rot_y] [rot_z] [sampling-gap] [center_x] [center_y] [boxsize_x] [boxsize_y] [boxsize_z] [percentile]

To run stlscale, which scales a STL file to fit the given build volume:
    
    python stlscale [/path/to/stlfile] --volume [vol_x,vol_y,vol_z]