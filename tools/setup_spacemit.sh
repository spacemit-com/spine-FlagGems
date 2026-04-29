#!/bin/bash

uv pip install triton --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple

uv pip install -e .[test]