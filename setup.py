from setuptools import setup, find_packages

setup(
    name="carpaint",
    version="0.1",
    packages=find_packages(where="."),  # 明确指定搜索路径
    package_dir={"": "."},
    include_package_data=True,
)