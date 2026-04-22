from setuptools import setup

package_name = 'game_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/game_planner.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ee106a',
    maintainer_email='ee106a@example.com',
    description='Game planner - reads board state and requests moves from solver',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'game_planner = game_planner.game_planner_node:main',
        ],
    },
)