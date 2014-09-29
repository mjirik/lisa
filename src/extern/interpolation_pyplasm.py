#! /usr/bin/python
# -*- coding: utf-8 -*-

from larcc import *
from lar2psm import *
from mapper import *
from splines import *


def TRIANGULAR_COONS_PATCH(params):
    ab0Fn = params[0]
    bc1Fn = params[1] 
    ca0Fn = params[2]

    def TRIANGULAR_COONS_PATCH0(point):
        u = point[0]
        v = point[1]
        w = point[2]


        if(hasattr(ab0Fn, '__call__')):
            Sab0 = ab0Fn(point)
        else:
            Sab0 = ab0Fn
        if(hasattr(bc1Fn, '__call__')):
            Sbc1 = bc1Fn(point)
        else:
            Sbc1 = bc1Fn
        if(hasattr(ca0Fn, '__call__')):
            Sca0 = ca0Fn(point)
        else:
            Sca0 = ca0Fn

        rn = len(Sab0)
        mapped = [0] * rn

        if (u == 1):
            ur = 0
        else:
            ur = ((u * v) / double(v + w))
        if (v == 1):
            vs = 0
        else:
            vs = ((v * w) / double(w + u))
        if (w == 1):
            wt = 0
        else:
            wt = ((w * u) / double(v + u))

        for i in range(0, rn):
            mapped[i] = \
                u * Sab0[i] + ur * (Sca0[i] - Sab0[i]) + \
                v * Sca0[i] + vs * (Sbc1[i] - Sca0[i]) + \
                w * Sbc1[i] + wt * (Sab0[i] - Sbc1[i])
        return mapped
    return TRIANGULAR_COONS_PATCH0


def TRIANGLE_DOMAIN(n, points):
    pa = points[0]
    pb = points[1]
    pc = points[2]
    net = []
    cells = []

    for i in range(0, n + 1):
        net.append([
            (pa[0] + i * (pb[0] - pa[0]) / double(n)),
            (pa[1] + i * (pb[1] - pa[1]) / double(n)),
            (pa[2] + i * (pb[2] - pa[2]) / double(n))
        ])

    for y in range(1, n + 1):
        r0 = (y - 1) * (n + 2) - (y - 1) * y / 2
        r1 = y * (n + 2) - y * (y + 1) / 2
        for x in range(0, n - y +1):
            c0 = r0 + x
            c1 = r1 + x
            net.append([
                pa[0] + 
                x * (pb[0] - pa[0]) / double(n) +
                y * (pc[0] - pa[0]) / double(n),
                pa[1] + 
                x * (pb[1] - pa[1]) / double(n) + 
                y * (pc[1] - pa[1]) / double(n),
                pa[2] + 
                x * (pb[2] - pa[2]) / double(n) + 
                y * (pc[2] - pa[2]) / double(n)
            ])
            if (x > 0):
                cells.append([c1, c0, c1 - 1])

            cells.append([c1, c0 + 1, c0])

        #print net[:2]
        #print cells[:2]
    return MKPOLS([net, cells])
