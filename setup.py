import os
from io import open
from os import path
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from sys import platform

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


COMPILE_ARGS = []

if os.name != "nt":
    COMPILE_ARGS.extend([
        "-std=c++11",
        "-funsigned-char",
        "-Wno-register",
        "-Wno-unused-function",
        "-Wno-unused-local-typedefs",
    ])
    if not platform.startswith("darwin"):
        COMPILE_ARGS.append("-fopenmp")
        COMPILE_ARGS.append("-march=native")

if platform.startswith("darwin"):
    COMPILE_ARGS.append("-stdlib=libc++")
    COMPILE_ARGS.append("-mmacosx-version-min=10.7")

cylimiter = Extension(
    name="cylimiter",
    language="c++",
    extra_compile_args=COMPILE_ARGS,
    include_dirs=[os.path.dirname(os.path.abspath(__file__)) + "/extensions"],
    sources=["extensions/cylimiter.pyx", "extensions/limiter.cpp", "extensions/reverb_rir.cpp"],
)
extensions = [cylimiter]

CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None

if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

__version__ = "0.4.2"

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf8") as source:
    long_description = source.read()


def main() -> None:
    setup(
        name="cylimiter",
        version=__version__,
        author="Piotr Żelasko",
        description="Audio limiter in Cython.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        keywords=[
            "signal processing",
            "audio",
        ],
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: Scientific/Engineering :: Mathematics",
        ],
        license="Apache 2.0",
        install_requires=['numpy'],
        extras_require={
            'dev': ['pytest', 'Cython']
        },
        ext_modules=extensions,
        packages=find_packages(exclude=["tests", "benchmark"]),
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
