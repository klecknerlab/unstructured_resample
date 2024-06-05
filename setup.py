from setuptools import setup

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
    zip_safe=False
)
