# GLITCH_scripts

To run comp_script.sh, which takes a STL file as input and visualizes the generated heatmap on local machine:

    chmod +x comp_script.sh
    ./comp_script.sh [stlfile] [rotx] [roty] [rotz] [sampling-gap] [center-x] [center-y] [boxsize-x] [boxsize-y] [boxsize-z] [percentile]

To run stlscale, which scales a STL file to fit the given build volume:
    
    python stlscale [stlfilepath]

Note: `stlscale` utilizes the `fits` method from `stlinfo`. This `fits` method will also return true if there is a possible rotation of the model that allows it to fit on the build plate. Instead, `fits` in stlscale should only return true if the model can fit without rotation.