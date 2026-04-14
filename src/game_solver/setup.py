from setuptools import setup

package_name = 'game_solver'

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
    maintainer='todo',
    maintainer_email='todo@todo.com',
    description='Connect Four game solver',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'game_solver_node = game_solver.game_solver_node:main',
        ],
    },
)