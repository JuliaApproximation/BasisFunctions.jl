#!/bin/bash
set -ev

{
  julia -e "Pkg.checkout(\"Domains\",\"$TRAVIS_BRANCH\")"
} || { # catch
  julia -e "Pkg.checkout(\"Domains\",\"development\")"
}

julia -e "Pkg.checkout(\"WaveletsCopy\",\"development\")"
