#! /usr/bin/env bash

# Not yet determined
auda pipe run DS-GDP-PW TF-TRIMMING MD-IF ST-BASIC MD-SVR PL-SVR SHOW \
    '--inputs=upper_trimming_percentage=0.2;degree=5'

# Or this
auda pipe run DS-GDP-URBAN-POP-PW ST-BASIC MD-PR-2D PL-PR-2D SHOW \
    --inputs=degree=1
