from setuptools import setup, find_packages

setup(
    name="raiselab",
    version="0.0.1",
    description="Modular VQE Framework (Monolithic + Distributed)",
    author="Milad Hasanzadeh",
    author_email="e.mhasanzadeh1377@yahoo.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy", "scipy", "qiskit==0.39.0", "qiskit-aer==0.11.0",
        "qiskit-terra==0.22.0", "qiskit-optimization==0.4.0",
        "diskit", "matplotlib", "joblib"
    ],
    python_requires=">=3.8",
)