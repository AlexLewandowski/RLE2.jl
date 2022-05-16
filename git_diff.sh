#!/usr/bin/env bash

git ls-files src/ --others --exclude-standard | xargs -n 1 git --no-pager diff /dev/null
echo ""
