import numpy as np

from conncomp.sphere import CubeLattice, path_to_antipode


def test_cube_lattice():
    N = 5
    lat = CubeLattice(N)
    assert len(lat) == (2 * N + 1) ** 3 - (2 * N - 1) ** 3
    assert len(lat.index) == len(lat)
    assert np.all(np.abs(lat.points).max(axis=1) == N)
    for i in (0, 17, len(lat) - 1):
        assert lat.points[lat.antipode(i)].tolist() == (-lat.points[i]).tolist()
    face_centre = lat.index[(0, 0, N)]
    assert len(list(lat.neighbours(face_centre))) == 8
    corner = lat.index[(N, N, N)]
    assert len(list(lat.neighbours(corner))) == 6
    assert lat.nearest([1, 2, -10]) == lat.index[(0, 1, -5)]


def test_path_to_antipode_circle():
    lat = CubeLattice(10)
    vals = lat.values([((2, 0, 0), 1), ((0, 2, 0), 1), ((0, 0, 2), -1)])  # x^2 + y^2 - z^2
    path, _ = path_to_antipode(lat, vals, lat.index[(10, 0, 0)])
    assert path[0] == lat.index[(10, 0, 0)] and path[-1] == lat.index[(-10, 0, 0)]
    assert all(vals[i] > 0 for i in path)
    path, visited = path_to_antipode(lat, vals, lat.index[(0, 0, 10)])  # inside the oval
    assert path is None and all(vals[i] < 0 for i in visited)
