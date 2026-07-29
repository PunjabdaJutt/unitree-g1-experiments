from setuptools import find_packages, setup

package_name = 'g1_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/bridge.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='unitree-g1-experiments',
    maintainer_email='unitree@unitree.com',
    description='Bridges the Unitree G1 raw DDS /lowstate into sensor_msgs/JointState',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'joint_state_bridge = g1_bridge.joint_state_bridge:main',
        ],
    },
)
