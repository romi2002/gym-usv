import setuptools

setuptools.setup(name='gym_usv',
      version='0.0.2',
      install_requires=['gym>=0.2.3'],
      package_dir={"": "gym_usv"},
      packages=setuptools.find_packages(where="."),
)
