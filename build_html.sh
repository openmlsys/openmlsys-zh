#!/bin/bash

set -e

rm -rf _build/rst _build/html
d2lbook build rst
cp static/frontpage.html _build/rst/
d2lbook build html
cp -r static/image/* _build/html/_images/