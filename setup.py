from cx_Freeze import setup, Executable


setup(
        name = " ",
        version = "0.1",
        description = " ",
        options = {"build_exe": {"packages": ["numpy.lib.format"]}},
        executables = [Executable("./src/cxpokus.py")]
        #executables = [Executable("./src/organ_segmentation.py")]
        )
        #executables = [Executable("./src/organ_segmentation.py")]
