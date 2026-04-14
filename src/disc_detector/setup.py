from setuptools import setup

package_name = 'disc_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ee106a',
    maintainer_email='ee106a@example.com',
    description='Detect red Connect 4 discs using USB camera',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'disc_node = disc_detector.disc_node:main',
        ],
    },
)