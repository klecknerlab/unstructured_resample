from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from juliacall import Main as jl

REQUIRED_JULIA_PKG = ["StaticArrays"]

def install_jl_modules(parent):
    with open('test.txt', 'wt') as f:
        for pkg in REQUIRED_JULIA_PKG:
            try: 
                jl.seval(f"using {pkg}")
                parent.announce(f'Julia package {pkg} already installed.')
            except:
                parent.announce(f'Installing Julia Package "{pkg}"')
                jl.seval(f'import Pkg; Pkg.add("{pkg}")')

class InstallWithPost(install):
    def run(self):
        install.run(self)
        install_jl_modules(self)

class DevelopWithPost(develop):
    def run(self):
        develop.run(self)
        install_jl_modules(self)
 
setup(
    name='unstructured_resample',
    version='0.1',
    description='Code for resampling non-uniform meshes from simulation data.  Developed by the Kleckner lab at UC Merced.',
    url='https://github.com/klecknerlab/unstructured_resample',
    author='Dustin Kleckner',
    author_email='dkleckner@ucmerced.edu',
    license='Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)',
    packages=['unstructured_resample'],
    install_requires=[ #Many of the packages are not in PyPi, so assume the user knows how to isntall them!
        # 'numpy',
    ],
    cmdclass=dict(
        install = InstallWithPost,
        develop = DevelopWithPost
    ),
    zip_safe=False
)
