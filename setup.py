# mypy: disable-error-code="import-untyped, import-not-found"
#!/usr/bin/env python
"""Setup script for the project."""

import re

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()

with open("kst/requirements.txt", "r", encoding="utf-8") as f:
    requirements: list[str] = f.read().splitlines()
    
with open("kst/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in kst/__init__.py"
version: str = version_re.group(1)


setup(
    name="kst",
    version=version,
    description="K-Scale's library for speech tokenization",
    author="K-Scale Labs",
    url="https://github.com/kscalelabs/kst",
    license_files=("LICENSE",),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    install_requires=requirements,
)
