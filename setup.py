from setuptools import setup, find_packages


setup(name='pyrol',
      version='0.0.1',
      description='Pytorch framework for Robotic systems, Learning, and classical control.',
      url='',
      author='Gabriel Fernandez',
      author_email='gabriel808[at]g[dot]ucla[dot]edu',
      license='',
      packages=[package for package in find_packages() if package.startswith('pyrol')],
      zip_safe=False,
      install_requires=[],
      # extras_require='',
      package_data={},
      tests_require=[],
      python_requires='>=3.6',
      classifiers=['Programming Language :: Python :: 3.6'],
      )

