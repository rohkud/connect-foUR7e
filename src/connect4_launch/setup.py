from setuptools import find_packages, setup

package_name = 'connect4_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/connect4_launch']),
        ('share/connect4_launch', ['package.xml']),
        ('share/connect4_launch/launch', ['launch/perception.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ee106a-aab',
    maintainer_email='rohankudchadker@berkeley.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
