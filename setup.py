#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Luis Antonio VASQUEZ",
    author_email='luis-antonio.vasquez-reina@inrae.fr',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Text classification for the projects of the Bibliome group at MaIAGE, INRAE",
    entry_points={
        'console_scripts': [
            'text_classification_bibliome=text_classification_bibliome.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='text_classification_bibliome',
    name='text_classification_bibliome',
    packages=find_packages(include=['text_classification_bibliome', 'text_classification_bibliome.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/luis-antonio.vasquez-reina/text_classification_bibliome',
    version='0.1.0',
    zip_safe=False,
)
