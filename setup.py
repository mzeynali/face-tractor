from setuptools import setup

setup(
    name='face_tractor',
    version='0.0.1',
    keywords='face_alignment, mediapipe, facemesh',
    url='',
    description='face alignment using mediapipe facemesh.',
    packages=['face_tractor'],
    install_requires=[
        "numpy==1.22.0",
        'mediapipe>=0.8.9.1',
        'opencv_contrib_python>=4.5.5.62',

    ],
    include_package_data=True,
    package_data={'': ['*.yaml']}
)
