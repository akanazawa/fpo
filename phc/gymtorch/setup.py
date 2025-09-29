from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension

ver = torch.__version__.split(sep=".")
ver_major = int(ver[0])
ver_minor = int(ver[1])

cflags = [
    "-DTORCH_MAJOR=%d" % ver_major,
    "-DTORCH_MINOR=%d" % ver_minor,
    "-Wl,-rpath," + torch.__path__[0],
]

setup(
    name="gymtorch",
    packages=["gymtorch"],
    version="0.1.0",
    description="Made Isaacgym's gymtorch a separate compiled module for easier debugging",
    install_requires=["torch", "setuptools"],
    ext_modules=[
        CppExtension(
            name="gymtorch._C",  # Note the ._C suffix - this is a common pattern
            sources=["gymtorch/gymtorch.cpp"],
            extra_compile_args=cflags,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
