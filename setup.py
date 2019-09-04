from setuptools import setup, find_packages


setup(name='pyrol',
      version='0.0.1',
      description='Pytorch framework for Robotic systems, Learning, and classical control.',
      url='',
      author='Gabriel Fernandez and Colin Togashi',
      author_email='gabriel808[at]g[dot]ucla[dot]edu',
      license='',
      packages=[package for package in find_packages() if package.startswith('pyrol')],
      zip_safe=False,
      install_requires=['numpy >= 1.17.0',
                        'torch >= 1.1.0',
                        'matplotlib >= 3.0.2',
                        'scipy >= 1.3.0'],
      # extras_require='',
      package_data={},
      tests_require=[],
      python_requires='>=3.6',
      classifiers=['Programming Language :: Python :: 3.6'],
      )

