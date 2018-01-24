from distutils.core import setup
import versioneer
from setuptools import find_packages

setup(name='sklearntoolsbase',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Tools for working with scikit-learn.',
      author='Jason Rudy',
      author_email='jcrudy@gmail.com',
      packages=find_packages(),
      install_requires = ['scikit-learn', 'sklearn2code']
     )