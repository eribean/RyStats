from setuptools import setup, convert_path

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if __name__ == '__main__':
    setup(
        name="RyStats", 
        packages=['RyStats.factoranalysis', 'RyStats.dimensionality',
                  'RyStats.common'],
        package_dir={'RyStats.common':convert_path('./common'),
                     'RyStats.factoranalysis':convert_path('./factoranalysis'),
                     'RyStats.dimensionality':convert_path('./dimensionality')
                     },
        version="0.1.0",
        license="MIT",
        description="Psychology Related Statistics in Python!",
        long_description=long_description.replace('<ins>','').replace('</ins>',''),
        long_description_content_type='text/markdown',
        author='Ryan C. Sanchez',
        author_email='ryan.sanchez@gofactr.com',
        url = 'https://github.com/eribean/RyStats',
        keywords = ['Psychology', 'Psychometrics', 'FactorAnalysis'],
        install_requires = ['numpy', 'scipy'],
        classifiers = [
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering', 
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',            
        ]
    )
