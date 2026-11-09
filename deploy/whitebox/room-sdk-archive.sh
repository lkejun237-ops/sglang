#!/bin/sh
# Export the pinned Room SDK inputs from Git, including existing build fixtures.
set -eu
exec git -C "$1" archive --format=tar "$2" -- . \
  ':(exclude)asset-library/subjects' \
  ':(exclude)asset-library/tests'
