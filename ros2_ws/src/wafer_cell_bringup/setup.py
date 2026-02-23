from setuptools import setup

package_name = 'wafer_cell_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/wafer_cell_sim.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aipowerisraelense',
    maintainer_email='n8naigabriel@gmail.com',
    description='Wafer cell bringup package for 3-arm mechatronic cell simulation',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wafer_cell_sim = wafer_cell_bringup.wafer_cell_sim_node:main',
        ],
    },
)
