from setuptools import setup, find_packages

setup(
    name="csp-workflow-mp",
    version="0.1.0",
    description="Formula-to-Structure Generation via Space-Group-Guided Template Retrieval (Materials Project edition)",
    author="Yen-Ju Wu",
    author_email="wu.yenju@nims.go.jp",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "pymatgen",
        "ase",
        "mp-api",
        "matplotlib",
        "seaborn",
    ],
    extras_require={
        "relaxation": ["torch", "mattersim"],
        "dev": ["jupyter", "pytest"],
    },
)
