from cx_Freeze import setup, Executable

additional_mods = ['numpy.core._methods', 'numpy.lib.format']

setup(name="wizard",
      version="0.1",
      description="",
      options={'build_exe': {'includes': additional_mods}},
      executables=[Executable("wizard.py")])
