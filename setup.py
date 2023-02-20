from setuptools import setup, find_packages


setup(
    name='plotly-upset',
    version='0.0.1',
    license='MIT',
    url="https://github.com/hshhrr/plotly-upset",
    author="Hasan Shahrier",
    author_email="hasan.shahrier.27@gmail.com",
    packages=find_packages(),
    include_package_data=False,
    install_requires=[
        'plotly>=5.5.0',
        'numpy>=1.21.6',
        'pandas>=1.3.5',
    ]
)
