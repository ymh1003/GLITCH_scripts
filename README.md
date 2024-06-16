# GLITCH_scripts

To run comp_script.sh, which takes a STL file as input and visualizes the generated heatmap on local machine:

    chmod +x comp_script.sh
    ./comp_script.sh [stlfilepath] [rotx] [roty] [rotz] [sampling-gap] [center-x] [center-y] [boxsize-x] [boxsize-y] [boxsize-z] [percentile]

To run stlscale, which scales a STL file to fit the given build volume:
    
    python stlscale [stlfilepath] --volume [vol_x,vol_y,vol_z]