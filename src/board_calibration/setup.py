from setuptools import setup

package_name = 'board_calibration'

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
    description='Board corner calibration tool for Connect Four',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'board_corners = board_calibration.board_corners:main',
            'disc_colors = board_calibration.disc_colors:main',
        ],
    },
)