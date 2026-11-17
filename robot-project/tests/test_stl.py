import numpy as np

from q8.stl import mesh_stats, read_stl, write_binary_stl

CUBE_ASCII = """solid cube
{facets}
endsolid cube
"""


def _cube_triangles(size=10.0):
    s = size
    v = np.array([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0], [0, 0, s], [s, 0, s], [s, s, s], [0, s, s]], float)
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return np.array([[v[a], v[b], v[c]] for a, b, c in faces])


def _write_ascii(path, tris):
    facets = []
    for t in tris:
        facets.append("facet normal 0 0 0\n outer loop\n" + "".join(
            f"  vertex {x} {y} {z}\n" for x, y, z in t) + " endloop\nendfacet")
    path.write_text(CUBE_ASCII.format(facets="\n".join(facets)))


def test_ascii_and_binary_round_trip(tmp_path):
    tris = _cube_triangles()
    ascii_path = tmp_path / "cube_ascii.stl"
    _write_ascii(ascii_path, tris)
    loaded = read_stl(ascii_path)
    assert loaded.shape == (12, 3, 3)
    bin_path = tmp_path / "cube_bin.stl"
    write_binary_stl(loaded, bin_path)
    assert bin_path.read_bytes()[:5] != b"solid"
    np.testing.assert_allclose(read_stl(bin_path), tris, atol=1e-5)


def test_mesh_stats_volume_and_bbox(tmp_path):
    path = tmp_path / "cube.stl"
    write_binary_stl(_cube_triangles(10.0), path)
    st = mesh_stats(path)
    assert st["triangles"] == 12
    assert abs(st["volume_mm3"] - 1000.0) < 1e-3
    assert st["bbox_mm"] == [10.0, 10.0, 10.0]
