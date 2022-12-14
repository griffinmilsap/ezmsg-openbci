import setuptools

setuptools.setup(
    name='ezmsg-openbci',
    packages=setuptools.find_namespace_packages(include=['ezmsg.*']),
    zip_safe=False
)
