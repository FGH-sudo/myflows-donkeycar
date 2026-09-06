# Stage-one derived evidence

All values derive from named run artifacts.

## Performance: operator-call and transfer-inclusive (ms)

| Case | Backend | Phase | Mean | Median | P95 | N | NumPy/CUDA | CuPy/CUDA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | numpy | forward | 0.063575 | 0.059300 | 0.073925 | 150 |  |  |
| P0 | numpy | backward | 0.103873 | 0.102600 | 0.115960 | 150 |  |  |
| P0 | numpy | combined | 0.167857 | 0.165950 | 0.176965 | 150 |  |  |
| P0 | numpy | transfer_inclusive | 0.301682 | 0.281150 | 0.452615 | 150 |  |  |
| P0 | cupy | forward | 0.379686 | 0.264192 | 1.229363 | 150 |  |  |
| P0 | cupy | backward | 0.583081 | 0.430528 | 1.601058 | 150 |  |  |
| P0 | cupy | combined | 0.985715 | 0.730784 | 1.988173 | 150 |  |  |
| P0 | cupy | transfer_inclusive | 2.051780 | 1.539400 | 4.251330 | 150 |  |  |
| P0 | cuda_c | forward | 0.342768 | 0.288416 | 1.096683 | 150 | 0.185 | 1.108 |
| P0 | cuda_c | backward | 1.025895 | 0.897056 | 1.864962 | 150 | 0.101 | 0.568 |
| P0 | cuda_c | combined | 1.313838 | 1.202240 | 2.048907 | 150 | 0.128 | 0.750 |
| P0 | cuda_c | transfer_inclusive | 2.085266 | 1.786600 | 4.330090 | 150 | 0.145 | 0.984 |
| P1 | numpy | forward | 4.425111 | 4.094500 | 5.677860 | 150 |  |  |
| P1 | numpy | backward | 7.794496 | 7.695600 | 8.435015 | 150 |  |  |
| P1 | numpy | combined | 12.362013 | 12.161800 | 14.258920 | 150 |  |  |
| P1 | numpy | transfer_inclusive | 14.417168 | 14.167200 | 15.757750 | 150 |  |  |
| P1 | cupy | forward | 0.335627 | 0.249920 | 1.044307 | 150 |  |  |
| P1 | cupy | backward | 0.557382 | 0.393728 | 1.765142 | 150 |  |  |
| P1 | cupy | combined | 0.928843 | 0.678944 | 2.024634 | 150 |  |  |
| P1 | cupy | transfer_inclusive | 2.363167 | 2.038450 | 4.370675 | 150 |  |  |
| P1 | cuda_c | forward | 0.426253 | 0.390480 | 1.325210 | 150 | 10.381 | 0.787 |
| P1 | cuda_c | backward | 1.050087 | 0.937424 | 1.981725 | 150 | 7.423 | 0.531 |
| P1 | cuda_c | combined | 1.412187 | 1.219056 | 2.343014 | 150 | 8.754 | 0.658 |
| P1 | cuda_c | transfer_inclusive | 2.676726 | 2.483650 | 4.794115 | 150 | 5.386 | 0.883 |
| P2 | numpy | forward | 3.101420 | 3.033400 | 3.582620 | 150 |  |  |
| P2 | numpy | backward | 9.320329 | 9.213850 | 10.279665 | 150 |  |  |
| P2 | numpy | combined | 12.434339 | 12.335150 | 13.126595 | 150 |  |  |
| P2 | numpy | transfer_inclusive | 16.642105 | 16.209750 | 17.452760 | 150 |  |  |
| P2 | cupy | forward | 0.352402 | 0.277040 | 1.275802 | 150 |  |  |
| P2 | cupy | backward | 0.627785 | 0.499264 | 1.799885 | 150 |  |  |
| P2 | cupy | combined | 0.955794 | 0.820624 | 1.909160 | 150 |  |  |
| P2 | cupy | transfer_inclusive | 2.568421 | 2.194550 | 5.233320 | 150 |  |  |
| P2 | cuda_c | forward | 0.394153 | 0.335216 | 1.182560 | 150 | 7.869 | 0.894 |
| P2 | cuda_c | backward | 1.284262 | 1.138176 | 2.242904 | 150 | 7.257 | 0.489 |
| P2 | cuda_c | combined | 1.474788 | 1.353744 | 2.506394 | 150 | 8.431 | 0.648 |
| P2 | cuda_c | transfer_inclusive | 2.814039 | 2.571200 | 4.692005 | 150 | 5.914 | 0.913 |
| P3-0-0 | numpy | forward | 0.909851 | 0.887650 | 1.039635 | 150 |  |  |
| P3-0-0 | numpy | backward | 0.966149 | 0.950850 | 1.090010 | 150 |  |  |
| P3-0-0 | numpy | combined | 1.905511 | 1.874950 | 2.141765 | 150 |  |  |
| P3-0-0 | numpy | transfer_inclusive | 2.220654 | 2.099350 | 2.447175 | 150 |  |  |
| P3-0-0 | cupy | forward | 0.456161 | 0.364368 | 0.719872 | 150 |  |  |
| P3-0-0 | cupy | backward | 0.293962 | 0.214096 | 0.634522 | 150 |  |  |
| P3-0-0 | cupy | combined | 0.914108 | 0.690032 | 2.027008 | 150 |  |  |
| P3-0-0 | cupy | transfer_inclusive | 1.839813 | 1.341700 | 3.400780 | 150 |  |  |
| P3-0-0 | cuda_c | forward | 0.333829 | 0.300416 | 0.524147 | 150 | 2.726 | 1.366 |
| P3-0-0 | cuda_c | backward | 0.398434 | 0.343568 | 1.229362 | 150 | 2.425 | 0.738 |
| P3-0-0 | cuda_c | combined | 0.812448 | 0.681888 | 1.628342 | 150 | 2.345 | 1.125 |
| P3-0-0 | cuda_c | transfer_inclusive | 1.399605 | 1.137800 | 2.518925 | 150 | 1.587 | 1.315 |
| P3-0-1 | numpy | forward | 1.056162 | 1.030900 | 1.210340 | 150 |  |  |
| P3-0-1 | numpy | backward | 1.690401 | 1.615750 | 2.146090 | 150 |  |  |
| P3-0-1 | numpy | combined | 2.721684 | 2.713950 | 2.947970 | 150 |  |  |
| P3-0-1 | numpy | transfer_inclusive | 3.223837 | 3.141750 | 3.553235 | 150 |  |  |
| P3-0-1 | cupy | forward | 0.527859 | 0.405344 | 1.524042 | 150 |  |  |
| P3-0-1 | cupy | backward | 0.332028 | 0.229904 | 1.069926 | 150 |  |  |
| P3-0-1 | cupy | combined | 0.710013 | 0.518448 | 1.704859 | 150 |  |  |
| P3-0-1 | cupy | transfer_inclusive | 1.753153 | 1.268150 | 3.304750 | 150 |  |  |
| P3-0-1 | cuda_c | forward | 0.324335 | 0.283520 | 0.991237 | 150 | 3.256 | 1.628 |
| P3-0-1 | cuda_c | backward | 0.358584 | 0.327792 | 0.863334 | 150 | 4.714 | 0.926 |
| P3-0-1 | cuda_c | combined | 0.716259 | 0.628736 | 1.576680 | 150 | 3.800 | 0.991 |
| P3-0-1 | cuda_c | transfer_inclusive | 1.365107 | 1.039900 | 2.462765 | 150 | 2.362 | 1.284 |
| P3-0-2 | numpy | forward | 6.747280 | 6.691750 | 7.453070 | 150 |  |  |
| P3-0-2 | numpy | backward | 7.341987 | 7.296500 | 7.871865 | 150 |  |  |
| P3-0-2 | numpy | combined | 14.142173 | 14.120800 | 14.892970 | 150 |  |  |
| P3-0-2 | numpy | transfer_inclusive | 15.562515 | 15.378400 | 16.305880 | 150 |  |  |
| P3-0-2 | cupy | forward | 1.614518 | 1.281536 | 3.372475 | 150 |  |  |
| P3-0-2 | cupy | backward | 0.420128 | 0.213344 | 1.287066 | 150 |  |  |
| P3-0-2 | cupy | combined | 1.514792 | 1.191424 | 2.942362 | 150 |  |  |
| P3-0-2 | cupy | transfer_inclusive | 2.891601 | 2.711550 | 5.232360 | 150 |  |  |
| P3-0-2 | cuda_c | forward | 0.330653 | 0.284512 | 1.155629 | 150 | 20.406 | 4.883 |
| P3-0-2 | cuda_c | backward | 0.385626 | 0.333472 | 1.305139 | 150 | 19.039 | 1.089 |
| P3-0-2 | cuda_c | combined | 0.709880 | 0.611568 | 1.550899 | 150 | 19.922 | 2.134 |
| P3-0-2 | cuda_c | transfer_inclusive | 1.545907 | 1.300800 | 2.658985 | 150 | 10.067 | 1.870 |
| P3-1-0 | numpy | forward | 0.399327 | 0.378700 | 0.498610 | 150 |  |  |
| P3-1-0 | numpy | backward | 0.627443 | 0.605700 | 0.737745 | 150 |  |  |
| P3-1-0 | numpy | combined | 1.086960 | 1.062250 | 1.320210 | 150 |  |  |
| P3-1-0 | numpy | transfer_inclusive | 1.844623 | 1.734000 | 2.227150 | 150 |  |  |
| P3-1-0 | cupy | forward | 0.380319 | 0.330240 | 0.673106 | 150 |  |  |
| P3-1-0 | cupy | backward | 0.316527 | 0.215472 | 0.837429 | 150 |  |  |
| P3-1-0 | cupy | combined | 0.633539 | 0.490128 | 1.555507 | 150 |  |  |
| P3-1-0 | cupy | transfer_inclusive | 1.997138 | 1.973200 | 3.524795 | 150 |  |  |
| P3-1-0 | cuda_c | forward | 0.375562 | 0.342848 | 1.102234 | 150 | 1.063 | 1.013 |
| P3-1-0 | cuda_c | backward | 0.417911 | 0.348608 | 1.233605 | 150 | 1.501 | 0.757 |
| P3-1-0 | cuda_c | combined | 0.814420 | 0.748032 | 1.591090 | 150 | 1.335 | 0.778 |
| P3-1-0 | cuda_c | transfer_inclusive | 1.460535 | 1.246750 | 2.473565 | 150 | 1.263 | 1.367 |
| P3-1-1 | numpy | forward | 0.678349 | 0.648550 | 0.825910 | 150 |  |  |
| P3-1-1 | numpy | backward | 1.286225 | 1.263300 | 1.368920 | 150 |  |  |
| P3-1-1 | numpy | combined | 2.078403 | 2.051500 | 2.316415 | 150 |  |  |
| P3-1-1 | numpy | transfer_inclusive | 3.304013 | 3.113650 | 3.695875 | 150 |  |  |
| P3-1-1 | cupy | forward | 0.511435 | 0.419600 | 1.337190 | 150 |  |  |
| P3-1-1 | cupy | backward | 0.321113 | 0.213728 | 1.238595 | 150 |  |  |
| P3-1-1 | cupy | combined | 0.755105 | 0.518992 | 1.907134 | 150 |  |  |
| P3-1-1 | cupy | transfer_inclusive | 1.849937 | 1.312450 | 3.626070 | 150 |  |  |
| P3-1-1 | cuda_c | forward | 0.354748 | 0.288800 | 1.235405 | 150 | 1.912 | 1.442 |
| P3-1-1 | cuda_c | backward | 0.396575 | 0.331408 | 1.329014 | 150 | 3.243 | 0.810 |
| P3-1-1 | cuda_c | combined | 0.732741 | 0.625584 | 1.455813 | 150 | 2.836 | 1.031 |
| P3-1-1 | cuda_c | transfer_inclusive | 1.388317 | 1.208700 | 2.425825 | 150 | 2.380 | 1.333 |
| P3-1-2 | numpy | forward | 5.159683 | 5.159450 | 5.669225 | 150 |  |  |
| P3-1-2 | numpy | backward | 5.531362 | 5.499350 | 5.885575 | 150 |  |  |
| P3-1-2 | numpy | combined | 10.975227 | 10.965850 | 11.664285 | 150 |  |  |
| P3-1-2 | numpy | transfer_inclusive | 13.296374 | 13.009650 | 14.017120 | 150 |  |  |
| P3-1-2 | cupy | forward | 1.654859 | 1.137008 | 3.349914 | 150 |  |  |
| P3-1-2 | cupy | backward | 0.340680 | 0.236544 | 1.219072 | 150 |  |  |
| P3-1-2 | cupy | combined | 1.343939 | 0.973264 | 2.465112 | 150 |  |  |
| P3-1-2 | cupy | transfer_inclusive | 2.518711 | 1.935850 | 4.937665 | 150 |  |  |
| P3-1-2 | cuda_c | forward | 0.313630 | 0.288352 | 0.771840 | 150 | 16.452 | 5.276 |
| P3-1-2 | cuda_c | backward | 0.405269 | 0.349184 | 1.193165 | 150 | 13.649 | 0.841 |
| P3-1-2 | cuda_c | combined | 0.644073 | 0.607792 | 1.229363 | 150 | 17.040 | 2.087 |
| P3-1-2 | cuda_c | transfer_inclusive | 1.437225 | 1.077200 | 2.702255 | 150 | 9.251 | 1.752 |

## Errors against independent FP64 reference

| Run | Case | Backend | Tensor | Max abs | Max rel | Allclose |
| --- | --- | --- | --- | --- | --- | --- |
| correctness-final-002 | T0 | numpy | y | 0 | 0 | True |
| correctness-final-002 | T0 | numpy | dx | 5.6624413e-07 | 1.1702282e-07 | True |
| correctness-final-002 | T0 | numpy | dw | 7.8678131e-06 | 8.4373206e-08 | True |
| correctness-final-002 | T0 | numpy | db | 3.2782555e-07 | 7.7474816e-08 | True |
| correctness-final-002 | T0 | cupy | y | 0 | 0 | True |
| correctness-final-002 | T0 | cupy | dx | 5.6624413e-07 | 1.1702282e-07 | True |
| correctness-final-002 | T0 | cupy | dw | 1.0550022e-05 | 1.2442917e-07 | True |
| correctness-final-002 | T0 | cupy | db | 1.4901161e-07 | 3.5215826e-08 | True |
| correctness-final-002 | T0 | cuda_c | y | 0 | 0 | True |
| correctness-final-002 | T0 | cuda_c | dx | 5.6624413e-07 | 1.1702282e-07 | True |
| correctness-final-002 | T0 | cuda_c | dw | 3.8444996e-06 | 5.3326779e-08 | True |
| correctness-final-002 | T0 | cuda_c | db | 1.4901161e-07 | 3.5215826e-08 | True |
| correctness-final-002 | T1 | numpy | y | 1.4614346e-06 | 2.5298283e-06 | True |
| correctness-final-002 | T1 | numpy | dx | 9.1509453e-07 | 2.5871833e-06 | True |
| correctness-final-002 | T1 | numpy | dw | 1.6429729e-06 | 4.2467727e-06 | True |
| correctness-final-002 | T1 | numpy | db | 4.4517219e-07 | 2.1178575e-07 | True |
| correctness-final-002 | T1 | cupy | y | 1.7831906e-06 | 1.8809323e-06 | True |
| correctness-final-002 | T1 | cupy | dx | 9.1509453e-07 | 2.5871833e-06 | True |
| correctness-final-002 | T1 | cupy | dw | 1.6429729e-06 | 1.6113875e-06 | True |
| correctness-final-002 | T1 | cupy | db | 2.8777868e-07 | 1.2848926e-07 | True |
| correctness-final-002 | T1 | cuda_c | y | 1.4614346e-06 | 2.5298283e-06 | True |
| correctness-final-002 | T1 | cuda_c | dx | 1.354093e-06 | 6.4967838e-06 | True |
| correctness-final-002 | T1 | cuda_c | dw | 1.3534902e-06 | 2.5710528e-06 | True |
| correctness-final-002 | T1 | cuda_c | db | 1.0021031e-06 | 4.5268785e-07 | True |
| correctness-final-002 | T2 | numpy | y | 4.3475765e-07 | 1.7591356e-06 | True |
| correctness-final-002 | T2 | numpy | dx | 2.7194828e-07 | 3.2536745e-07 | True |
| correctness-final-002 | T2 | numpy | dw | 1.8738959e-06 | 2.0190044e-07 | True |
| correctness-final-002 | T2 | numpy | db | 5.0477684e-07 | 6.0216966e-07 | True |
| correctness-final-002 | T2 | cupy | y | 4.3475765e-07 | 2.9419765e-06 | True |
| correctness-final-002 | T2 | cupy | dx | 3.0737631e-07 | 3.1824243e-07 | True |
| correctness-final-002 | T2 | cupy | dw | 8.9099527e-07 | 2.0190044e-07 | True |
| correctness-final-002 | T2 | cupy | db | 1.1050142e-06 | 4.1725041e-07 | True |
| correctness-final-002 | T2 | cuda_c | y | 4.3475765e-07 | 1.7591356e-06 | True |
| correctness-final-002 | T2 | cuda_c | dx | 2.7194828e-07 | 3.2536745e-07 | True |
| correctness-final-002 | T2 | cuda_c | dw | 1.8738959e-06 | 2.0190044e-07 | True |
| correctness-final-002 | T2 | cuda_c | db | 4.4889748e-07 | 5.3550881e-07 | True |
| correctness-final-002 | T3 | numpy | y | 1.0257513e-05 | 4.2876154e-05 | True |
| correctness-final-002 | T3 | numpy | dx | 2.6794183e-06 | 1.7569e-05 | True |
| correctness-final-002 | T3 | numpy | dw | 2.8470075e-06 | 7.8384808e-05 | True |
| correctness-final-002 | T3 | numpy | db | 9.611249e-07 | 2.1786175e-07 | True |
| correctness-final-002 | T3 | cupy | y | 4.2150528e-06 | 1.5373563e-05 | True |
| correctness-final-002 | T3 | cupy | dx | 2.4351234e-06 | 1.2962264e-05 | True |
| correctness-final-002 | T3 | cupy | dw | 3.3211542e-06 | 0.00045231891 | True |
| correctness-final-002 | T3 | cupy | db | 9.4622374e-07 | 9.1266408e-07 | True |
| correctness-final-002 | T3 | cuda_c | y | 1.0257513e-05 | 4.2876154e-05 | True |
| correctness-final-002 | T3 | cuda_c | dx | 5.6555496e-06 | 2.1572301e-05 | True |
| correctness-final-002 | T3 | cuda_c | dw | 4.2691177e-06 | 0.00055456526 | True |
| correctness-final-002 | T3 | cuda_c | db | 9.4622374e-07 | 1.477927e-06 | True |
| correctness-final-002 | T4 | numpy | y | 8.2144483e-07 | 1.6673774e-06 | True |
| correctness-final-002 | T4 | numpy | dx | 4.5256397e-07 | 2.1291466e-06 | True |
| correctness-final-002 | T4 | numpy | dw | 1.9444861e-06 | 7.3920948e-07 | True |
| correctness-final-002 | T4 | numpy | db | 2.6449561e-07 | 1.8046219e-07 | True |
| correctness-final-002 | T4 | cupy | y | 8.2144483e-07 | 4.7993312e-06 | True |
| correctness-final-002 | T4 | cupy | dx | 1.022645e-06 | 2.1291466e-06 | True |
| correctness-final-002 | T4 | cupy | dw | 1.8702111e-06 | 8.2306604e-07 | True |
| correctness-final-002 | T4 | cupy | db | 2.7054921e-07 | 5.6068733e-07 | True |
| correctness-final-002 | T4 | cuda_c | y | 8.2144483e-07 | 1.6673774e-06 | True |
| correctness-final-002 | T4 | cuda_c | dx | 1.022645e-06 | 8.5345851e-07 | True |
| correctness-final-002 | T4 | cuda_c | dw | 1.9444861e-06 | 7.3920948e-07 | True |
| correctness-final-002 | T4 | cuda_c | db | 2.6449561e-07 | 1.8046219e-07 | True |
| correctness-final-002 | T5-0-0 | numpy | y | 0 | 0 | True |
| correctness-final-002 | T5-0-0 | numpy | dx | 0 | 0 | True |
| correctness-final-002 | T5-0-0 | cupy | y | 0 | 0 | True |
| correctness-final-002 | T5-0-0 | cupy | dx | 0 | 0 | True |
| correctness-final-002 | T5-0-0 | cuda_c | y | 0 | 0 | True |
| correctness-final-002 | T5-0-0 | cuda_c | dx | 0 | 0 | True |
| correctness-final-002 | T5-0-1 | numpy | y | 0 | 0 | True |
| correctness-final-002 | T5-0-1 | numpy | dx | 1.1920929e-07 | 5.9054422e-08 | True |
| correctness-final-002 | T5-0-1 | cupy | y | 0 | 0 | True |
| correctness-final-002 | T5-0-1 | cupy | dx | 1.1920929e-07 | 5.9054422e-08 | True |
| correctness-final-002 | T5-0-1 | cuda_c | y | 0 | 0 | True |
| correctness-final-002 | T5-0-1 | cuda_c | dx | 1.1920929e-07 | 5.9054422e-08 | True |
| correctness-final-002 | T5-0-2 | numpy | y | 0 | 0 | True |
| correctness-final-002 | T5-0-2 | numpy | dx | 7.1106479e-07 | 2.4202687e-07 | True |
| correctness-final-002 | T5-0-2 | cupy | y | 0 | 0 | True |
| correctness-final-002 | T5-0-2 | cupy | dx | 2.4260953e-07 | 1.101432e-06 | True |
| correctness-final-002 | T5-0-2 | cuda_c | y | 0 | 0 | True |
| correctness-final-002 | T5-0-2 | cuda_c | dx | 3.9488077e-07 | 1.101432e-06 | True |
| performance-final-004 | P0 | numpy | y | 2.8231106e-06 | 0.00011754722 | True |
| performance-final-004 | P0 | numpy | dx | 2.50502e-06 | 5.2134646e-05 | True |
| performance-final-004 | P0 | numpy | dw | 8.8170747e-06 | 0.00013911512 | True |
| performance-final-004 | P0 | numpy | db | 4.2396132e-06 | 6.8420715e-07 | True |
| performance-final-004 | P0 | cupy | y | 2.1907429e-06 | 8.1236599e-05 | True |
| performance-final-004 | P0 | cupy | dx | 2.5299093e-06 | 0.0001054035 | True |
| performance-final-004 | P0 | cupy | dw | 5.1394829e-06 | 8.9631455e-05 | True |
| performance-final-004 | P0 | cupy | db | 2.0076986e-06 | 1.9128607e-07 | True |
| performance-final-004 | P0 | cuda_c | y | 2.8231106e-06 | 0.00011754722 | True |
| performance-final-004 | P0 | cuda_c | dx | 5.8856129e-06 | 0.00011627527 | True |
| performance-final-004 | P0 | cuda_c | dw | 1.8214497e-05 | 0.00015622178 | True |
| performance-final-004 | P0 | cuda_c | db | 1.325109e-05 | 1.2597085e-06 | True |
| performance-final-004 | P1 | numpy | y | 2.4423392e-05 | 0.0049258506 | True |
| performance-final-004 | P1 | numpy | dx | 1.2995313e-05 | 0.0024522571 | True |
| performance-final-004 | P1 | numpy | dw | 7.8412586e-05 | 0.0012463407 | True |
| performance-final-004 | P1 | numpy | db | 1.3607147e-05 | 7.5783181e-06 | True |
| performance-final-004 | P1 | cupy | y | 2.4423392e-05 | 0.0049258506 | True |
| performance-final-004 | P1 | cupy | dx | 1.3085892e-05 | 0.0032308418 | True |
| performance-final-004 | P1 | cupy | dw | 4.625076e-05 | 0.0012469299 | True |
| performance-final-004 | P1 | cupy | db | 2.3833407e-05 | 3.6883919e-06 | True |
| performance-final-004 | P1 | cuda_c | y | 2.4423392e-05 | 0.0049258506 | True |
| performance-final-004 | P1 | cuda_c | dx | 4.6527973e-05 | 0.0041439015 | True |
| performance-final-004 | P1 | cuda_c | dw | 0.00045387377 | 0.0016702934 | True |
| performance-final-004 | P1 | cuda_c | db | 0.00012531521 | 2.1168298e-05 | True |
| performance-final-004 | P2 | numpy | y | 2.2617624e-05 | 0.001841731 | True |
| performance-final-004 | P2 | numpy | dx | 6.3139409e-06 | 0.0025485979 | True |
| performance-final-004 | P2 | numpy | dw | 8.2184956e-05 | 9.7971512e-05 | True |
| performance-final-004 | P2 | numpy | db | 6.1929459e-06 | 2.0661985e-06 | True |
| performance-final-004 | P2 | cupy | y | 2.2617624e-05 | 0.001841731 | True |
| performance-final-004 | P2 | cupy | dx | 6.3139409e-06 | 0.0025485979 | True |
| performance-final-004 | P2 | cupy | dw | 3.8283857e-05 | 0.00070958777 | True |
| performance-final-004 | P2 | cupy | db | 9.5360156e-06 | 4.0967489e-06 | True |
| performance-final-004 | P2 | cuda_c | y | 2.2617624e-05 | 0.001841731 | True |
| performance-final-004 | P2 | cuda_c | dx | 1.6925744e-05 | 0.0027197267 | True |
| performance-final-004 | P2 | cuda_c | dw | 0.00060350503 | 0.00047512713 | True |
| performance-final-004 | P2 | cuda_c | db | 0.00014223251 | 7.0703987e-05 | True |
| performance-final-004 | P3-0-0 | numpy | y | 0 | 0 | True |
| performance-final-004 | P3-0-0 | numpy | dx | 0 | 0 | True |
| performance-final-004 | P3-0-0 | cupy | y | 0 | 0 | True |
| performance-final-004 | P3-0-0 | cupy | dx | 0 | 0 | True |
| performance-final-004 | P3-0-0 | cuda_c | y | 0 | 0 | True |
| performance-final-004 | P3-0-0 | cuda_c | dx | 0 | 0 | True |
| performance-final-004 | P3-0-1 | numpy | y | 0 | 0 | True |
| performance-final-004 | P3-0-1 | numpy | dx | 4.1723251e-07 | 1.9626893e-06 | True |
| performance-final-004 | P3-0-1 | cupy | y | 0 | 0 | True |
| performance-final-004 | P3-0-1 | cupy | dx | 4.1723251e-07 | 5.8880678e-06 | True |
| performance-final-004 | P3-0-1 | cuda_c | y | 0 | 0 | True |
| performance-final-004 | P3-0-1 | cuda_c | dx | 4.4703484e-07 | 3.9253786e-06 | True |
| performance-final-004 | P3-0-2 | numpy | y | 0 | 0 | True |
| performance-final-004 | P3-0-2 | numpy | dx | 1.4007092e-06 | 0.00022303586 | True |
| performance-final-004 | P3-0-2 | cupy | y | 0 | 0 | True |
| performance-final-004 | P3-0-2 | cupy | dx | 1.0877848e-06 | 0.00024884166 | True |
| performance-final-004 | P3-0-2 | cuda_c | y | 0 | 0 | True |
| performance-final-004 | P3-0-2 | cuda_c | dx | 1.2852252e-06 | 0.00038032836 | True |
| performance-final-004 | P3-1-0 | numpy | y | 0 | 0 | True |
| performance-final-004 | P3-1-0 | numpy | dx | 0 | 0 | True |
| performance-final-004 | P3-1-0 | cupy | y | 0 | 0 | True |
| performance-final-004 | P3-1-0 | cupy | dx | 0 | 0 | True |
| performance-final-004 | P3-1-0 | cuda_c | y | 0 | 0 | True |
| performance-final-004 | P3-1-0 | cuda_c | dx | 0 | 0 | True |
| performance-final-004 | P3-1-1 | numpy | y | 0 | 0 | True |
| performance-final-004 | P3-1-1 | numpy | dx | 4.7683716e-07 | 3.860989e-06 | True |
| performance-final-004 | P3-1-1 | cupy | y | 0 | 0 | True |
| performance-final-004 | P3-1-1 | cupy | dx | 4.7683716e-07 | 3.860989e-06 | True |
| performance-final-004 | P3-1-1 | cuda_c | y | 0 | 0 | True |
| performance-final-004 | P3-1-1 | cuda_c | dx | 4.1723251e-07 | 6.1415204e-06 | True |
| performance-final-004 | P3-1-2 | numpy | y | 0 | 0 | True |
| performance-final-004 | P3-1-2 | numpy | dx | 1.2218952e-06 | 0.00055060665 | True |
| performance-final-004 | P3-1-2 | cupy | y | 0 | 0 | True |
| performance-final-004 | P3-1-2 | cupy | dx | 1.2218952e-06 | 0.00055060665 | True |
| performance-final-004 | P3-1-2 | cuda_c | y | 0 | 0 | True |
| performance-final-004 | P3-1-2 | cuda_c | dx | 1.9073486e-06 | 0.00011295606 | True |

## dX optimization: ordinary operator-call timings

| Case | Phase | Before ms | After ms | Speedup |
| --- | --- | --- | --- | --- |
| P0 | forward | 0.319086 | 0.340605 | 0.937 |
| P0 | backward | 0.969197 | 0.941138 | 1.030 |
| P0 | combined | 1.269449 | 1.255871 | 1.011 |
| P1 | forward | 0.404562 | 0.439600 | 0.920 |
| P1 | backward | 1.326880 | 0.959035 | 1.384 |
| P1 | combined | 1.719162 | 1.187929 | 1.447 |
| P2 | forward | 0.358741 | 0.357007 | 1.005 |
| P2 | backward | 1.315105 | 0.909053 | 1.447 |
| P2 | combined | 1.628944 | 1.192942 | 1.365 |

## Systems: three forward/backward pairs per capture

| Run | Kernel calls | Kernel total us | NVTX total us |
| --- | --- | --- | --- |
| nsys-P0-cupy-002 | 72 | 131.616 | 3260.347 |
| nsys-P0-dx-window-003 | 21 | 108.608 | 4161.105 |
| nsys-P1-cupy-002 | 69 | 695.676 | 3312.918 |
| nsys-P1-dx-window-003 | 21 | 1704.754 | 4522.580 |

## Compute: conv2d_backward_input

| Metric | Unit | P0 | P1 |
| --- | --- | --- | --- |
| gpu__time_duration.sum | us | 10.432000 | 86.240000 |
| sm__throughput.avg.pct_of_peak_sustained_elapsed | % | 1.942608 | 62.023919 |
| sm__warps_active.avg.pct_of_peak_sustained_active | % | 14.502875 | 82.788499 |
| gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed | % | 3.024038 | 9.295482 |
| l1tex__throughput.avg.pct_of_peak_sustained_active | % | 13.228132 | 64.674472 |
| launch__registers_per_thread | register/thread | 40 | 40 |
| launch__block_size |  | 256 | 256 |
| launch__occupancy_limit_registers | block | 6.000000 | 6.000000 |
| launch__waves_per_multiprocessor |  | 0.02 | 1.78 |

## PS equivalence and one-run wall time

| Run | Shards | Max abs | PS wall s | Single wall s | Cleanup s |
| --- | --- | --- | --- | --- | --- |
| restored-ps-1-001 | [32] | 0 | 0.465770 | 0.015413 | 0.000086 |
| restored-ps-2-001 | [16, 16] | 5.5511151e-17 | 0.508892 | 0.015211 | 0.000129 |
| restored-ps-4-001 | [8, 8, 8, 8] | 2.7755576e-17 | 0.568102 | 0.015770 | 0.000096 |
| restored-ps-uneven-001 | [20, 12] | 5.5511151e-17 | 0.486503 | 0.016179 | 0.000087 |

## PS phase timings (includes first-step startup wait)

| Run | Phase | N | Mean ms | P95 ms |
| --- | --- | --- | --- | --- |
| restored-ps-1-001 | compute_s | 20 | 0.169095 | 0.211855 |
| restored-ps-1-001 | upload_submit_s | 20 | 0.020320 | 0.023760 |
| restored-ps-1-001 | upload_ack_wait_s | 20 | 0.374370 | 0.502845 |
| restored-ps-1-001 | parameter_wait_receive_s | 21 | 0.214171 | 0.470200 |
| restored-ps-1-001 | collect_wait_s | 20 | 1.413130 | 1.741205 |
| restored-ps-1-001 | aggregate_s | 20 | 0.022145 | 0.030820 |
| restored-ps-1-001 | update_s | 20 | 0.024685 | 0.036510 |
| restored-ps-1-001 | broadcast_submit_s | 20 | 0.080045 | 0.120400 |
| restored-ps-1-001 | step_s | 20 | 1.543695 | 1.874095 |
| restored-ps-2-001 | compute_s | 40 | 0.189968 | 0.247635 |
| restored-ps-2-001 | upload_submit_s | 40 | 0.025955 | 0.054360 |
| restored-ps-2-001 | upload_ack_wait_s | 40 | 0.384210 | 0.616395 |
| restored-ps-2-001 | parameter_wait_receive_s | 42 | 0.601867 | 0.753885 |
| restored-ps-2-001 | collect_wait_s | 20 | 2.133860 | 2.411235 |
| restored-ps-2-001 | aggregate_s | 20 | 0.028740 | 0.046550 |
| restored-ps-2-001 | update_s | 20 | 0.028140 | 0.043725 |
| restored-ps-2-001 | broadcast_submit_s | 20 | 0.119245 | 0.144645 |
| restored-ps-2-001 | step_s | 20 | 2.314220 | 2.603830 |
| restored-ps-4-001 | compute_s | 80 | 0.181694 | 0.277905 |
| restored-ps-4-001 | upload_submit_s | 80 | 0.034678 | 0.084055 |
| restored-ps-4-001 | upload_ack_wait_s | 80 | 0.380659 | 0.659530 |
| restored-ps-4-001 | parameter_wait_receive_s | 84 | 1.105631 | 0.967060 |
| restored-ps-4-001 | collect_wait_s | 20 | 3.456525 | 3.591935 |
| restored-ps-4-001 | aggregate_s | 20 | 0.036070 | 0.052925 |
| restored-ps-4-001 | update_s | 20 | 0.024435 | 0.036545 |
| restored-ps-4-001 | broadcast_submit_s | 20 | 0.175005 | 0.219035 |
| restored-ps-4-001 | step_s | 20 | 3.696900 | 3.887160 |
| restored-ps-uneven-001 | compute_s | 40 | 0.176520 | 0.276695 |
| restored-ps-uneven-001 | upload_submit_s | 40 | 0.026582 | 0.057320 |
| restored-ps-uneven-001 | upload_ack_wait_s | 40 | 0.353773 | 0.549055 |
| restored-ps-uneven-001 | parameter_wait_receive_s | 42 | 0.431738 | 0.509795 |
| restored-ps-uneven-001 | collect_wait_s | 20 | 1.758535 | 1.929470 |
| restored-ps-uneven-001 | aggregate_s | 20 | 0.026010 | 0.033800 |
| restored-ps-uneven-001 | update_s | 20 | 0.024760 | 0.035270 |
| restored-ps-uneven-001 | broadcast_submit_s | 20 | 0.117575 | 0.149200 |
| restored-ps-uneven-001 | step_s | 20 | 1.931005 | 2.129990 |
