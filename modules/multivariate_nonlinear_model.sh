# Or this
auda pipe run DS-GDP-URBAN-POP-PW ST-BASIC MD-PR-2D PL-PR-2D SHOW \
    --inputs=degree=1

auda pipe run \
    DS-GPD-URBN-POP-PW \
    AD-IF:on=dataset \
    TF-Z-NORM:on=normalized_dataset \
    MD-LR:on=inlier_dataset \
    PL-LR-3D:on=dataset \
    PL-SHOW
