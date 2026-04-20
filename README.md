# AstrometricSignalProcessing

To use ASP, create a seperate folder for your project and copy `dependencies.py` and `config.json` into your folder and do the following things:
- set the path to ASP and gaiamock in your version of `config.json`
- or, uncomment and manually set the home directory for ASP in your version of `dependencies.py`

Once `dependencies.py` is imported with `from dependencies import *`, you can use `asp_import("name", alias="alias")` to load in my additional scripts. For example, to use Gaiamock wrapper, do `asp_import("GaiamockWrapper", alias="gw")`. The script automatically recursively checks all folders in ASP so you just need to provide the name of the python script
