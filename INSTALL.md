Install CURA, you should be able to locate a CuraEngine binary.

If you haven't configured a printer, use Cura to add a printer.
Usually this is the Voron Trident 300.

Create a virtual environment: python39 39venv

Activate the venv: source 39venv/bin/activate

Install stlinfo
  cd stlinfo
  python3 setup.py develop

Install pj3d
  pip install -r requirements.txt
  python3 setup.py develop

  Configure pj3d to set the path of CuraEngine using a config file (see the README for details)

  Also generate a settings file for pj3d and store it somewhere. This
  is required to run pj3d but not essential right now. A sample
  settings file is provided (vt300_settings.txt)

Install glitch requirements (this assumes requirements from pj3d are installed)
  cd Gcode-Checking-Project
  pip install -r requirements.txt

Install glitch scripts requirements [fast parse branch]
  pip install numpy-stl
  Setup a glitch_runner configure: ./glitch_runner.py test.yaml config
