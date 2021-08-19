from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='opteeq',
    version='1',
    packages=['tools', 'tools.aws', 'tools.via', 'tools.via.structure'],
    url='https://github.com/assansanogo/Opteeq',
    license='',
    author='',
    author_email='',
    description='library to read digitalised paper receipts',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['google-cloud-vision==2.4.2',
                      'boto3==1.18.20',
                      'botocore==1.21.20',
                      'tqdm==4.62.0',
                      'python-dateutil~=2.8.1',
                      'Pillow~=8.3.1',
                      ],
    extras_require={
        "dev": ["boto-stubs", "mypy_boto3_builder", "moto"],
        "dev_docs": ["Sphinx", "sphinx_rtd_theme", "myst_parser"]
    },
    python_requires='>=3.9',
)
