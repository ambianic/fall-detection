from setuptools import setup, find_packages

setup(
    name="fall-detection",
    version="1.0.0",
    author = "Bhavika Panara",
    author_email = "panara.bhavika@gmail.com",
    description = "Standalone Python ML library for people fall detection based on Tensorflow and PoseNet 2.0.",
    url="https://ambianic.ai",
    license="Apache Software License 2.0",
    classifiers=[
        "Development Status :: Beta",
        "Programming Language :: Python :: 3",
        "OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "TOPIC :: HOME AUTOMATION",
        "TOPIC :: SOFTWARE DEVELOPMENT :: EMBEDDED SYSTEMS",
        "TOPIC :: SCIENTIFIC/ENGINEERING :: ARTIFICIAL INTELLIGENCE",
        "Intended Audience :: Developers",
    ],  
    include_package_data=True,     
    install_requires=[
        "numpy>=1.16.2", 
        "Pillow>=5.4.1",
        "PyYAML>=5.1.2"
        ],        
    packages= find_packages(),
)
