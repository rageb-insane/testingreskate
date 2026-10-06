"""skate. 'TangentSpace' vertex element (usage 0x34, format UByte4N): the tangent frame
as a normal plus a rotation angle for the tangent, unpacked exactly as the game's
vertex shader does. Ported from the decompiled CalcTangentSpaceNormal_AxisAngle
shipped with FrostyToolsuite (Shaders/UnpackNormals.hlsl).

    byte 0      normal component a, low 8 bits  (a = b0/255*0.7071 - 0.7071 [+0.7071 if flag 16])
    byte 1      normal component b, low 8 bits  (               [+0.7071 if flag 32])
    byte 2      tangent angle, low 8 bits       (angle = b2/255*pi/2 + (flags & 192)*pi/128)
    byte 3      flags: 1 sign of the third normal component, 2/4 axis order,
                8 bitangent flip, 16/32 high bits of a/b, 64|128 angle quadrant
"""
import numpy as np

HALF = 0.707107


def _normalize(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def decode(tf):
    """(n, 4) uint8 -> (normal, tangent, bitangent), each (n, 3)."""
    tf = np.asarray(tf, dtype=np.float64)
    flags = tf[:, 3].astype(np.int64)
    a = tf[:, 0] / 255 * HALF - HALF + (flags & 16) * 0.0441942
    b = tf[:, 1] / 255 * HALF - HALF + (flags & 32) * 0.0220971
    angle = tf[:, 2] / 255 * 1.5708 + (flags & 192) * 0.0245437
    c = np.sqrt(np.abs(1 - (a * a + b * b)))
    c = np.where(flags & 1, -c, c)
    r = np.stack([c, a, b], 1)                               # r2.xyz
    r = np.where((flags & 4)[:, None] != 0, r, r[:, [1, 0, 2]])
    n = np.where((flags & 2)[:, None] != 0, r, r[:, [1, 2, 0]])
    ref = np.where((flags & 2)[:, None] != 0,
                   np.stack([-n[:, 1], n[:, 0], np.zeros(len(n))], 1),
                   np.stack([n[:, 2], np.zeros(len(n)), -n[:, 0]], 1))
    ref = _normalize(ref)
    t = ref * np.cos(np.abs(angle))[:, None] + np.cross(n, ref) * np.sin(np.abs(angle))[:, None]
    bt = np.cross(n, t)
    bt = np.where((flags & 8)[:, None] != 0, -bt, bt)
    return _normalize(n), _normalize(-t), _normalize(bt)


def _split(v, hi_value):
    """[-HALF, HALF] -> (low byte, high flag) so that low/255*HALF - HALF + flag*HALF == v."""
    u = np.clip((v + HALF) / HALF, 0, 2)
    hi = u >= 1
    low = np.clip(np.rint((u - hi) * 255), 0, 255).astype(np.int64)
    return low, np.where(hi, hi_value, 0)


def encode(normal, tangent, bitangent):
    """Unit normal/tangent/bitangent (n, 3) -> (n, 4) uint8, the inverse of decode()."""
    n = _normalize(np.asarray(normal, dtype=np.float64))
    largest = np.argmax(np.abs(n), axis=1)
    # which flags put the reconstructed component c on the largest axis (see decode)
    flags = np.select([largest == 0, largest == 1], [4 | 2, 2], 4).astype(np.int64)
    c = n[np.arange(len(n)), largest]
    a = np.select([largest == 0, largest == 1], [n[:, 1], n[:, 0]], n[:, 0])
    b = np.select([largest == 0, largest == 1], [n[:, 2], n[:, 2]], n[:, 1])
    flags |= (c < 0).astype(np.int64)
    a_low, a_hi = _split(a, 16)
    b_low, b_hi = _split(b, 32)
    flags |= a_hi | b_hi
    # the decoder rebuilds n from the quantised a, b; use that n for the angle
    qa = a_low / 255 * HALF - HALF + a_hi * 0.0441942
    qb = b_low / 255 * HALF - HALF + b_hi * 0.0220971
    qc = np.sqrt(np.abs(1 - (qa * qa + qb * qb))) * np.where(c < 0, -1, 1)
    nq = np.stack([qc, qa, qb], 1)
    nq = np.where((flags & 4)[:, None] != 0, nq, nq[:, [1, 0, 2]])
    nq = np.where((flags & 2)[:, None] != 0, nq, nq[:, [1, 2, 0]])
    ref = np.where((flags & 2)[:, None] != 0,
                   np.stack([-nq[:, 1], nq[:, 0], np.zeros(len(nq))], 1),
                   np.stack([nq[:, 2], np.zeros(len(nq)), -nq[:, 0]], 1))
    ref = _normalize(ref)
    p = -np.asarray(tangent, dtype=np.float64)                 # decode returns -p
    p = _normalize(p - (p * nq).sum(1, keepdims=True) * nq)
    theta = np.arctan2((p * np.cross(nq, ref)).sum(1), (p * ref).sum(1)) % (2 * np.pi)
    quadrant = np.minimum((theta // (np.pi / 2)).astype(np.int64), 3)
    t_low = np.clip(np.rint((theta - quadrant * np.pi / 2) / (np.pi / 2) * 255), 0, 255).astype(np.int64)
    flags |= quadrant * 64
    # bitangent: decode gives cross(n, p_rotated), flipped by flag 8
    p_rot = ref * np.cos(theta)[:, None] + np.cross(nq, ref) * np.sin(theta)[:, None]
    flip = (np.cross(nq, p_rot) * np.asarray(bitangent, dtype=np.float64)).sum(1) < 0
    flags |= np.where(flip, 8, 0)
    return np.stack([a_low, b_low, t_low, flags], 1).astype(np.uint8)
