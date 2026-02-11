from setuptools import setup
import sys
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
   Pybind11Extension("cpp_backend_impl", ["binding.cc"], extra_compile_args=['-O2', '-std=c++17', '-Wall', '-Wextra', '-Werror'],)
]

## uncomment for gdb debug
# ext_modules = [
#    Pybind11Extension("cpp_backend_impl", ["binding.cc"],
#                      extra_compile_args=['-g', '-O0', '-std=c++17', '-Wall', '-Wextra', '-Werror'],
#                      extra_link_args=['-g']),
# ]

class clean_and_build_ext(build_ext):
    def run(self):
        import shutil
        import os
        build_dir = os.path.join(os.getcwd(), 'build')
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        super().run()

if len(sys.argv) == 1:
    print("No command specified, defaulting to 'build_ext --inplace'")
    sys.argv.extend(['build_ext', '--inplace'])

setup(
    name="cpp_backend_impl",
    ext_modules=ext_modules,
    cmdclass={"build_ext": clean_and_build_ext},
    zip_safe=False,
)