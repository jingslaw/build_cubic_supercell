import numpy as np
from numpy import cross
from numpy.linalg import norm
from math import ceil
from numba import njit
from itertools import product


@njit(cache=True, fastmath=True)
def det3(m):
    return (
            m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
            - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
            + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )


@njit(cache=True, fastmath=True)
def matmul3x3(A, B):
    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i, j] += A[i, k] * B[k, j]
    return C


@njit(cache=True, fastmath=True)
def deviation(cell):
    a, b, c = cell

    abx = a[1]*b[2] - a[2]*b[1]
    aby = a[2]*b[0] - a[0]*b[2]
    abz = a[0]*b[1] - a[1]*b[0]
    ab_norm = (abx**2 + aby**2 + abz**2)**0.5

    acx = a[1]*c[2] - a[2]*c[1]
    acy = a[2]*c[0] - a[0]*c[2]
    acz = a[0]*c[1] - a[1]*c[0]
    ac_norm = (acx**2 + acy**2 + acz**2)**0.5

    bcx = b[1]*c[2] - b[2]*c[1]
    bcy = b[2]*c[0] - b[0]*c[2]
    bcz = b[0]*c[1] - b[1]*c[0]
    bc_norm = (bcx**2 + bcy**2 + bcz**2)**0.5

    surface = 2 * (ab_norm + ac_norm + bc_norm)

    # Volume = |a . (b x c)|
    vol = abs(a[0]*bcx + a[1]*bcy + a[2]*bcz)

    return surface * vol ** (-2/3) / 6 - 1


def find_transform_matrix(
        lattice: np.ndarray,
        num: int = 10,
        start: int = 1,
        tolerance: float = 1e-5,
        fast_search: bool = False,
        write_log: bool = False,
        lower_bound: int | None = None,
        upper_bound: int | None = None
):
    # Validate input parameters
    primitive_lattices = np.asarray(lattice)
    if primitive_lattices.shape != (3, 3):
        raise ValueError("primitive_lattices must be a 3x3 matrix")

    if not isinstance(num, int) or num < 1:
        raise ValueError("times must be an integer ≥ 1")
    if not isinstance(start, int) or start < 1:
        raise ValueError("start must be an integer ≥ 1")
    if start >= num:
        raise ValueError("start must be less than times")

    if not isinstance(tolerance, float) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative float")

    if not isinstance(fast_search, bool):
        raise ValueError("fast_search must be a boolean")

    if lower_bound is not None and not isinstance(lower_bound, int):
        raise ValueError("lower_bound must be an int or None")
    if upper_bound is not None and not isinstance(upper_bound, int):
        raise ValueError("upper_bound must be an int or None")
    if lower_bound is not None and upper_bound is not None:
        if lower_bound >= upper_bound:
            raise ValueError("lower_bound must be less than upper_bound")

    volume = det3(lattice)

    # Precompute projection lengths
    max_length = pow(num, 1.0 / 3) * max(norm(lattice[0]), norm(lattice[1]), norm(lattice[2]))
    r1 = ceil(max_length / (volume / norm(cross(lattice[1], lattice[2]))))
    r2 = ceil(max_length / (volume / norm(cross(lattice[2], lattice[0]))))
    r3 = ceil(max_length / (volume / norm(cross(lattice[0], lattice[1]))))

    if lower_bound:
        r1_min = lower_bound
        r2_min = lower_bound
        r3_min = lower_bound
    else:
        r1_min = -r1
        r2_min = -r2
        r3_min = -r3
    if upper_bound:
        r1_max = upper_bound
        r2_max = upper_bound
        r3_max = upper_bound
    else:
        r1_max = r1
        r2_max = r2
        r3_max = r3

    bias = [np.inf] * (num - start + 1)
    matrixes = [None] * (num - start + 1)

    # Cache indices
    range_r1 = range(r1_min, r1_max + 1)
    range_r2 = range(r2_min, r2_max + 1)
    range_r3 = range(r3_min, r3_max + 1)

    total = ((r1_max - r1_min + 1) ** 2 * (r2_max - r2_min + 1) ** 2 * (r3_max - r3_min + 1) ** 2
             * (r1_max + 1) * (r2_max + 1) * (r3_max + 1))
    i = 0

    for P11 in range(0, r1_max + 1):
        for P22 in range(0, r2_max + 1):
            for P33 in range(0, r3_max + 1):
                for P21, P31, P12, P32, P13, P23 in product(range_r1, range_r1, range_r2, range_r2, range_r3, range_r3):
                    pmatrix = np.array([
                        [P11, P12, P13],
                        [P21, P22, P23],
                        [P31, P32, P33]
                    ])
                    n = int(abs(det3(pmatrix)))
                    i += 1
                    if n > num or n < start:
                        continue
                    # super_cell = np.dot(pmatrix, cell)
                    super_cell = matmul3x3(pmatrix, lattice)

                    result = deviation(super_cell)

                    if fast_search and result < tolerance:
                        with open('log.txt', 'w') as file:
                            file.write('100%')
                        return pmatrix

                    idx = n - start
                    if result < bias[idx]:
                        bias[idx] = result
                        matrixes[idx] = pmatrix

            percent = 100 * i / total
            if write_log:
                with open('log.txt', 'w') as file:
                    file.write(f'{percent:.2f}%\n')
            else:
                print(f'{percent:.2f}%', end='\r')

    with open('log.txt', 'w') as file:
        file.write('100%\n')

    with open('p_matrix.txt', 'w') as file:
        file.write(f'{total} supercells are calculated.\n\n')
        for i in range(len(bias)):
            file.write(f'{i + start}\n')
            file.write(f'bias: {bias[i]:.8f}\n')
            file.write(f'{matrixes[i]}\n\n')
        file.write('cell size and bias\n\n')
        for i in range(len(bias)):
            file.write(f'{i + start} {bias[i]:.7f}\n')

    index = np.argmin(bias)
    return matrixes[index]


if __name__ == "__main__":
    from pymatgen.core.structure import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.transformations.advanced_transformations import SupercellTransformation, CubicSupercellTransformation
    import time

    fp = "primitive_low_sym.vasp"
    for fp in ["primitive_high_sym.vasp", "primitive_low_sym.vasp"]:
        t_start = time.perf_counter()
        primitive = Structure.from_file(filename=fp)
        lattice = primitive.lattice.matrix
        result = find_transform_matrix(lattice, 10, start=1, lower_bound=None, upper_bound=None, write_log=True)
        st = SupercellTransformation(result)
        superstructure = st.apply_transformation(primitive)

        t_end = time.perf_counter()
        print(f"Elapsed time: {t_end - t_start:.6f} seconds")

        print(f'transfer matrix:\n{result}')
        print(superstructure)

        print("\nPymatgen Cubic Supercell Transformation:")
        t_start = time.perf_counter()
        transform = CubicSupercellTransformation(
            max_atoms=20,
            min_length=4
        )
        supercell_pymat = transform.apply_transformation(primitive)
        t_end = time.perf_counter()
        print(f"Elapsed time: {t_end - t_start:.6f} seconds")
        print(supercell_pymat)

    high_sym = Structure.from_file(filename="primitive_high_sym.vasp")
    low_sym = Structure.from_file(filename="primitive_low_sym.vasp")
    sm = StructureMatcher()
    print("Whether primitive_high_sym and primitive_low_sym are equivalent structures:", sm.fit(high_sym, low_sym))
