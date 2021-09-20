import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from wetb.utils.rotation import axis2axis_angle, axis_angle2quaternion, quaternion2matrix, s2matrix, matrix2quaternion,\
    quaternion2axis_angle, axis_angle2axis, matrix2axis


def init_plot(ref_coo='gl', rot=(180, 180)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_coo(ref_coo, np.eye(3) * 1.5, lw=1, alpha=1)
    for xyz in 'xyz':
        getattr(ax, 'set_%slim' % xyz)([-2, 2])
        getattr(ax, 'set_%slabel' % xyz)(xyz)

    ax.view_init(rot[0] + 10, rot[1] - 10)


def plot_coo(coo, A, origo=(0, 0, 0), lw=1, alpha=1):
    ax = plt.gca()
    for v, c, xyz in zip(np.asarray(A).T, 'rgb', 'xyz'):
        ax.quiver(*origo, *v, color=c, lw=lw, alpha=alpha, arrow_length_ratio=.2, zorder=-32)
        ax.text(*(np.array(origo) + v * 1.1), "$%s_{%s}$" % (xyz, coo),
                alpha=alpha, fontsize=12, fontweight='bold', zorder=32)


def plot_axis_rotation(rot_coo, axis, origo=(0, 0, 0), ref_coo=np.eye(3), deg=False):

    axis_angle = axis2axis_angle(axis)
    mat = quaternion2matrix(axis_angle2quaternion(axis_angle, deg))
    mat = np.dot(ref_coo, mat)
    axis = np.dot(ref_coo, axis_angle[:3])
    xyz = np.array([origo, np.array(origo) + axis * 1.5])
    plt.gca().plot(*xyz.T, '-.k', lw=1)
    plot_coo(rot_coo, mat, origo)
    return mat


def plot_matrix_rotation(rot_name, matrix, origo, ref_coo=np.eye(3)):
    mat = np.dot(ref_coo, matrix)
    plot_coo(rot_name, mat, origo)


def plot_body(nodes):
    ax = plt.gca()
    ax.plot(*np.array(nodes).T, alpha=0.2, lw=10)
    ax.plot(*np.array(nodes).T, '.k', alpha=1)


def set_aspect_equal():
    ax = plt.gca()
    xyz_lim = [getattr(ax, 'get_%slim' % xyz)() for xyz in 'xyz']
    max_range = np.max([np.abs(np.subtract(*lim)) for lim in xyz_lim]) / 2
    mid = np.mean(xyz_lim, 1)
    for xyz, m in zip('xyz', mid):
        getattr(ax, "set_%slim" % xyz)([m - max_range, m + max_range])
    plt.tight_layout()


def vs_array(s, shape):
    arr = np.array(s.strip().replace("D", "E").split("\n"), dtype=float)
    if len(np.atleast_1d(shape)) == 2:
        arr = arr.reshape(shape[::-1]).T
    return arr


if __name__ == '__main__':
    TBD = vs_array("""-0.991775138730704
-1.644972253859245D-002
0.126931008126843
1.644972253859245D-002
0.967100554922815
0.253862016253685
-0.126931008126843
0.253862016253685
-0.958875693653519""", (3, 3))
    TUB = vs_array("""-1.00000000000000
0.000000000000000D+000
0.000000000000000D+000
0.000000000000000D+000
1.00000000000000
0.000000000000000D+000
0.000000000000000D+000
0.000000000000000D+000
-1.00000000000000""", (3, 3))
    TDU = vs_array("""0.991775138730704
-1.644972253859245D-002
0.126931008126843
-1.644972253859245D-002
0.967100554922815
0.253862016253685
-0.126931008126843
-0.253862016253685
0.958875693653519""", (3, 3))
    print(np.dot(TBD.T, TUB.T) - TDU)
    print(TDU)
    rot_bd = matrix2axis(TBD)
    rot_bu = matrix2axis(TUB)
    rot_du = matrix2axis(TDU)

    print(np.dot(TUB, matrix2axis(np.dot(TUB.T, TBD.T))))
    print(matrix2axis(TDU))

    print(np.dot(TUB.T, TBD.T).T)
