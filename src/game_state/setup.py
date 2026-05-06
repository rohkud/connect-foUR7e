from setuptools import setup

package_name = 'game_state'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ee106a',
    maintainer_email='ee106a@example.com',
    description='Builds a Connect 4 board state from board and disc detector results',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'board_state = {package_name}.board_state:main',
            f'solver = {package_name}.solver:main'
        ],
    },
)
