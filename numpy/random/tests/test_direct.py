import builtins
import csv
import math
import os
import sys
from os.path import join
from pathlib import Path

import pytest

import numpy as np
from numpy.random import (
    MT19937,
    PCG64,
    PCG64DXSM,
    SFC64,
    Generator,
    Philox,
    RandomState,
    SeedSequence,
    default_rng,
)
from numpy.random._common import interface
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_raises,
)

try:
    import cffi  # noqa: F401

    MISSING_CFFI = False
except ImportError:
    MISSING_CFFI = True

try:
    import ctypes  # noqa: F401

    MISSING_CTYPES = False
except ImportError:
    MISSING_CTYPES = False

if sys.flags.optimize > 1:
    # no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1
    # cffi cannot succeed
    MISSING_CFFI = True


pwd = os.path.dirname(os.path.abspath(__file__))


def assert_state_equal(actual, target):
    for key in actual:
        if isinstance(actual[key], dict):
            assert_state_equal(actual[key], target[key])
        elif isinstance(actual[key], np.ndarray):
            assert_array_equal(actual[key], target[key])
        else:
            assert actual[key] == target[key]


def uint32_to_float32(u):
    return ((u >> np.uint32(8)) * (1.0 / 2**24)).astype(np.float32)


def uniform32_from_uint64(x):
    x = np.uint64(x)
    upper = np.array(x >> np.uint64(32), dtype=np.uint32)
    lower = np.uint64(0xffffffff)
    lower = np.array(x & lower, dtype=np.uint32)
    joined = np.column_stack([lower, upper]).ravel()
    return uint32_to_float32(joined)


def uniform32_from_uint53(x):
    x = np.uint64(x) >> np.uint64(16)
    x = np.uint32(x & np.uint64(0xffffffff))
    return uint32_to_float32(x)


def uniform32_from_uint32(x):
    return uint32_to_float32(x)


def uniform32_from_uint(x, bits):
    if bits == 64:
        return uniform32_from_uint64(x)
    elif bits == 53:
        return uniform32_from_uint53(x)
    elif bits == 32:
        return uniform32_from_uint32(x)
    else:
        raise NotImplementedError


def uniform_from_uint(x, bits):
    if bits in (64, 63, 53):
        return uniform_from_uint64(x)
    elif bits == 32:
        return uniform_from_uint32(x)


def uniform_from_uint64(x):
    return (x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


def uniform_from_uint32(x):
    out = np.empty(len(x) // 2)
    for i in range(0, len(x), 2):
        a = x[i] >> 5
        b = x[i + 1] >> 6
        out[i // 2] = (a * 67108864.0 + b) / 9007199254740992.0
    return out


def uniform_from_dsfmt(x):
    return x.view(np.double) - 1.0


def gauss_from_uint(x, n, bits):
    if bits in (64, 63):
        doubles = uniform_from_uint64(x)
    elif bits == 32:
        doubles = uniform_from_uint32(x)
    else:  # bits == 'dsfmt'
        doubles = uniform_from_dsfmt(x)
    gauss = []
    loc = 0
    x1 = x2 = 0.0
    while len(gauss) < n:
        r2 = 2
        while r2 >= 1.0 or r2 == 0.0:
            x1 = 2.0 * doubles[loc] - 1.0
            x2 = 2.0 * doubles[loc + 1] - 1.0
            r2 = x1 * x1 + x2 * x2
            loc += 2

        f = np.sqrt(-2.0 * np.log(r2) / r2)
        gauss.append(f * x2)
        gauss.append(f * x1)

    return gauss[:n]


def test_seedsequence():
    from numpy.random.bit_generator import (
        ISeedSequence,
        ISpawnableSeedSequence,
        SeedlessSeedSequence,
    )

    s1 = SeedSequence(range(10), spawn_key=(1, 2), pool_size=6)
    s1.spawn(10)
    s2 = SeedSequence(**s1.state)
    assert_equal(s1.state, s2.state)
    assert_equal(s1.n_children_spawned, s2.n_children_spawned)

    # The interfaces cannot be instantiated themselves.
    assert_raises(TypeError, ISeedSequence)
    assert_raises(TypeError, ISpawnableSeedSequence)
    dummy = SeedlessSeedSequence()
    assert_raises(NotImplementedError, dummy.generate_state, 10)
    assert len(dummy.spawn(10)) == 10


def test_generator_spawning():
    """ Test spawning new generators and bit_generators directly.
    """
    rng = np.random.default_rng()
    seq = rng.bit_generator.seed_seq
    new_ss = seq.spawn(5)
    expected_keys = [seq.spawn_key + (i,) for i in range(5)]
    assert [c.spawn_key for c in new_ss] == expected_keys

    new_bgs = rng.bit_generator.spawn(5)
    expected_keys = [seq.spawn_key + (i,) for i in range(5, 10)]
    assert [bg.seed_seq.spawn_key for bg in new_bgs] == expected_keys

    new_rngs = rng.spawn(5)
    expected_keys = [seq.spawn_key + (i,) for i in range(10, 15)]
    found_keys = [rng.bit_generator.seed_seq.spawn_key for rng in new_rngs]
    assert found_keys == expected_keys

    # Sanity check that streams are actually different:
    assert new_rngs[0].uniform() != new_rngs[1].uniform()


def test_non_spawnable():
    from numpy.random.bit_generator import ISeedSequence

    class FakeSeedSequence:
        def generate_state(self, n_words, dtype=np.uint32):
            return np.zeros(n_words, dtype=dtype)

    ISeedSequence.register(FakeSeedSequence)

    rng = np.random.default_rng(FakeSeedSequence())

    with pytest.raises(TypeError, match="The underlying SeedSequence"):
        rng.spawn(5)

    with pytest.raises(TypeError, match="The underlying SeedSequence"):
        rng.bit_generator.spawn(5)


class Base:
    dtype = np.uint64
    data2 = data1 = {}

    @classmethod
    def setup_class(cls):
        cls.bit_generator = PCG64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.seed_error_type = TypeError
        cls.invalid_init_types = []
        cls.invalid_init_values = []

    @classmethod
    def _read_csv(cls, filename):
        with open(filename) as csv:
            seed = csv.readline()
            seed = seed.split(',')
            seed = [int(s.strip(), 0) for s in seed[1:]]
            data = []
            for line in csv:
                data.append(int(line.split(',')[-1].strip(), 0))
            return {'seed': seed, 'data': np.array(data, dtype=cls.dtype)}

    def test_raw(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        uints = bit_generator.random_raw(1000)
        assert_equal(uints, self.data1['data'])

        bit_generator = self.bit_generator(*self.data1['seed'])
        uints = bit_generator.random_raw()
        assert_equal(uints, self.data1['data'][0])

        bit_generator = self.bit_generator(*self.data2['seed'])
        uints = bit_generator.random_raw(1000)
        assert_equal(uints, self.data2['data'])

    def test_random_raw(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        uints = bit_generator.random_raw(output=False)
        assert uints is None
        uints = bit_generator.random_raw(1000, output=False)
        assert uints is None

    def test_gauss_inv(self):
        n = 25
        rs = RandomState(self.bit_generator(*self.data1['seed']))
        gauss = rs.standard_normal(n)
        assert_allclose(gauss,
                        gauss_from_uint(self.data1['data'], n, self.bits))

        rs = RandomState(self.bit_generator(*self.data2['seed']))
        gauss = rs.standard_normal(25)
        assert_allclose(gauss,
                        gauss_from_uint(self.data2['data'], n, self.bits))

    def test_uniform_double(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        vals = uniform_from_uint(self.data1['data'], self.bits)
        uniforms = rs.random(len(vals))
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float64)

        rs = Generator(self.bit_generator(*self.data2['seed']))
        vals = uniform_from_uint(self.data2['data'], self.bits)
        uniforms = rs.random(len(vals))
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float64)

    def test_uniform_float(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        vals = uniform32_from_uint(self.data1['data'], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float32)

        rs = Generator(self.bit_generator(*self.data2['seed']))
        vals = uniform32_from_uint(self.data2['data'], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float32)

    def test_repr(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert 'Generator' in repr(rs)
        assert f'{id(rs):#x}'.upper().replace('X', 'x') in repr(rs)

    def test_str(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert 'Generator' in str(rs)
        assert str(self.bit_generator.__name__) in str(rs)
        assert f'{id(rs):#x}'.upper().replace('X', 'x') not in str(rs)

    def test_pickle(self):
        import pickle

        bit_generator = self.bit_generator(*self.data1['seed'])
        state = bit_generator.state
        bitgen_pkl = pickle.dumps(bit_generator)
        reloaded = pickle.loads(bitgen_pkl)
        reloaded_state = reloaded.state
        assert_array_equal(Generator(bit_generator).standard_normal(1000),
                           Generator(reloaded).standard_normal(1000))
        assert bit_generator is not reloaded
        assert_state_equal(reloaded_state, state)

        ss = SeedSequence(100)
        aa = pickle.loads(pickle.dumps(ss))
        assert_equal(ss.state, aa.state)

    def test_pickle_preserves_seed_sequence(self):
        # GH 26234
        # Add explicit test that bit generators preserve seed sequences
        import pickle

        bit_generator = self.bit_generator(*self.data1['seed'])
        ss = bit_generator.seed_seq
        bg_plk = pickle.loads(pickle.dumps(bit_generator))
        ss_plk = bg_plk.seed_seq
        assert_equal(ss.state, ss_plk.state)
        assert_equal(ss.pool, ss_plk.pool)

        bit_generator.seed_seq.spawn(10)
        bg_plk = pickle.loads(pickle.dumps(bit_generator))
        ss_plk = bg_plk.seed_seq
        assert_equal(ss.state, ss_plk.state)
        assert_equal(ss.n_children_spawned, ss_plk.n_children_spawned)

    def test_invalid_state_type(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        with pytest.raises(TypeError):
            bit_generator.state = {'1'}

    def test_invalid_state_value(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        state = bit_generator.state
        state['bit_generator'] = 'otherBitGenerator'
        with pytest.raises(ValueError):
            bit_generator.state = state

    def test_invalid_init_type(self):
        bit_generator = self.bit_generator
        for st in self.invalid_init_types:
            with pytest.raises(TypeError):
                bit_generator(*st)

    def test_invalid_init_values(self):
        bit_generator = self.bit_generator
        for st in self.invalid_init_values:
            with pytest.raises((ValueError, OverflowError)):
                bit_generator(*st)

    def test_benchmark(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        bit_generator._benchmark(1)
        bit_generator._benchmark(1, 'double')
        with pytest.raises(ValueError):
            bit_generator._benchmark(1, 'int32')

    @pytest.mark.skipif(MISSING_CFFI, reason='cffi not available')
    def test_cffi(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        cffi_interface = bit_generator.cffi
        assert isinstance(cffi_interface, interface)
        other_cffi_interface = bit_generator.cffi
        assert other_cffi_interface is cffi_interface

    @pytest.mark.skipif(MISSING_CTYPES, reason='ctypes not available')
    def test_ctypes(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        ctypes_interface = bit_generator.ctypes
        assert isinstance(ctypes_interface, interface)
        other_ctypes_interface = bit_generator.ctypes
        assert other_ctypes_interface is ctypes_interface

    def test_getstate(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        state = bit_generator.state
        alt_state = bit_generator.__getstate__()
        assert isinstance(alt_state, tuple)
        assert_state_equal(state, alt_state[0])
        assert isinstance(alt_state[1], SeedSequence)

class TestPhilox(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = Philox
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(
            join(pwd, './data/philox-testset-1.csv'))
        cls.data2 = cls._read_csv(
            join(pwd, './data/philox-testset-2.csv'))
        cls.seed_error_type = TypeError
        cls.invalid_init_types = []
        cls.invalid_init_values = [(1, None, 1), (-1,), (None, None, 2 ** 257 + 1)]

    def test_set_key(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        state = bit_generator.state
        keyed = self.bit_generator(counter=state['state']['counter'],
                                   key=state['state']['key'])
        assert_state_equal(bit_generator.state, keyed.state)


class TestPCG64(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = PCG64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, './data/pcg64-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/pcg64-testset-2.csv'))
        cls.seed_error_type = (ValueError, TypeError)
        cls.invalid_init_types = [(3.2,), ([None],), (1, None)]
        cls.invalid_init_values = [(-1,)]

    def test_advance_symmetry(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        state = rs.bit_generator.state
        step = -0x9e3779b97f4a7c150000000000000000
        rs.bit_generator.advance(step)
        val_neg = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(2**128 + step)
        val_pos = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(10 * 2**128 + step)
        val_big = rs.integers(10)
        assert val_neg == val_pos
        assert val_big == val_pos

    def test_advange_large(self):
        rs = Generator(self.bit_generator(38219308213743))
        pcg = rs.bit_generator
        state = pcg.state["state"]
        initial_state = 287608843259529770491897792873167516365
        assert state["state"] == initial_state
        pcg.advance(sum(2**i for i in (96, 64, 32, 16, 8, 4, 2, 1)))
        state = pcg.state["state"]
        advanced_state = 135275564607035429730177404003164635391
        assert state["state"] == advanced_state


class TestPCG64DXSM(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = PCG64DXSM
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, './data/pcg64dxsm-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/pcg64dxsm-testset-2.csv'))
        cls.seed_error_type = (ValueError, TypeError)
        cls.invalid_init_types = [(3.2,), ([None],), (1, None)]
        cls.invalid_init_values = [(-1,)]

    def test_advance_symmetry(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        state = rs.bit_generator.state
        step = -0x9e3779b97f4a7c150000000000000000
        rs.bit_generator.advance(step)
        val_neg = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(2**128 + step)
        val_pos = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(10 * 2**128 + step)
        val_big = rs.integers(10)
        assert val_neg == val_pos
        assert val_big == val_pos

    def test_advange_large(self):
        rs = Generator(self.bit_generator(38219308213743))
        pcg = rs.bit_generator
        state = pcg.state
        initial_state = 287608843259529770491897792873167516365
        assert state["state"]["state"] == initial_state
        pcg.advance(sum(2**i for i in (96, 64, 32, 16, 8, 4, 2, 1)))
        state = pcg.state["state"]
        advanced_state = 277778083536782149546677086420637664879
        assert state["state"] == advanced_state


class TestMT19937(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = MT19937
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, './data/mt19937-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/mt19937-testset-2.csv'))
        cls.seed_error_type = ValueError
        cls.invalid_init_types = []
        cls.invalid_init_values = [(-1,)]

    def test_seed_float_array(self):
        assert_raises(TypeError, self.bit_generator, np.array([np.pi]))
        assert_raises(TypeError, self.bit_generator, np.array([-np.pi]))
        assert_raises(TypeError, self.bit_generator, np.array([np.pi, -np.pi]))
        assert_raises(TypeError, self.bit_generator, np.array([0, np.pi]))
        assert_raises(TypeError, self.bit_generator, [np.pi])
        assert_raises(TypeError, self.bit_generator, [0, np.pi])

    def test_state_tuple(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        bit_generator = rs.bit_generator
        state = bit_generator.state
        desired = rs.integers(2 ** 16)
        tup = (state['bit_generator'], state['state']['key'],
               state['state']['pos'])
        bit_generator.state = tup
        actual = rs.integers(2 ** 16)
        assert_equal(actual, desired)
        tup = tup + (0, 0.0)
        bit_generator.state = tup
        actual = rs.integers(2 ** 16)
        assert_equal(actual, desired)


class TestSFC64(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = SFC64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(
            join(pwd, './data/sfc64-testset-1.csv'))
        cls.data2 = cls._read_csv(
            join(pwd, './data/sfc64-testset-2.csv'))
        cls.seed_error_type = (ValueError, TypeError)
        cls.invalid_init_types = [(3.2,), ([None],), (1, None)]
        cls.invalid_init_values = [(-1,)]

    def test_legacy_pickle(self):
        # Pickling format was changed in 2.0.x
        import gzip
        import pickle

        expected_state = np.array(
            [
                9957867060933711493,
                532597980065565856,
                14769588338631205282,
                13
            ],
            dtype=np.uint64
        )

        base_path = os.path.split(os.path.abspath(__file__))[0]
        pkl_file = os.path.join(base_path, "data", "sfc64_np126.pkl.gz")
        with gzip.open(pkl_file) as gz:
            sfc = pickle.load(gz)

        assert isinstance(sfc, SFC64)
        assert_equal(sfc.state["state"]["state"], expected_state)


class TestDefaultRNG:
    def test_seed(self):
        for args in [(), (None,), (1234,), ([1234, 5678],)]:
            rg = default_rng(*args)
            assert isinstance(rg.bit_generator, PCG64)

    def test_passthrough(self):
        bg = Philox()
        rg = default_rng(bg)
        assert rg.bit_generator is bg
        rg2 = default_rng(rg)
        assert rg2 is rg
        assert rg2.bit_generator is bg

    @pytest.mark.thread_unsafe(
        reason="np.random.set_bit_generator affects global state"
    )
    def test_coercion_RandomState_Generator(self):
        # use default_rng to coerce RandomState to Generator
        rs = RandomState(1234)
        rg = default_rng(rs)
        assert isinstance(rg.bit_generator, MT19937)
        assert rg.bit_generator is rs._bit_generator

        # RandomState with a non MT19937 bit generator
        _original = np.random.get_bit_generator()
        bg = PCG64(12342298)
        np.random.set_bit_generator(bg)
        rs = np.random.mtrand._rand
        rg = default_rng(rs)
        assert rg.bit_generator is bg

        # vital to get global state back to original, otherwise
        # other tests start to fail.
        np.random.set_bit_generator(_original)


class DistributionPropertiesBase:
    """
    Base class for testing mean/variance-based distribution properties using
    external CSV datasets that encode (mean, var, seed) triplets.
    Subclasses must implement:
    - get_data_filename(self) -> str
    - draw_samples(self, mu: float, var: float, seed: int, size: int) -> np.ndarray
    Optionally override:
    - null_mean(self, mu: float, var: float) -> float
    """
    class HexFile:
        """
        Lightweight reader for CSV files where each row contains one or more
        64-bit words encoded as hex strings (e.g. "0x0123...").
        Expected CSV format:
            header_row
            <hexword>,...
            <hexword>,...
            ...
        Decoding behavior:
        - Each selected hex word is parsed as a 64-bit integer (base 16).
        - The integer is converted to 8 big-endian bytes.
        - Bytes from multiple words are concatenated in row order.
        - Trailing NUL bytes (b'\\0') are stripped.
        - The result is decoded to UTF-8 using 'ignore' for invalid sequences.
        Examples:
            hf = HexFile("data.csv")
            hf[0]                -> decode first column of row 0 (string)
            hf[0, 0]             -> raw cell value (hex string), e.g. "0xAB..."
            hf[0:10, 0]          -> decode column 0 for rows 0..9 (string)
            hf[5:8]              -> decode column 0 for rows 5..7 (string)
            len(hf)             -> number of data rows (excluding header)
        """

        def __init__(self, path) -> None:
            self.path = Path(path)
            self.header: List[str]
            self.rows: List[Sequence[str]]

            with self.path.open(newline="") as fh:
                reader = csv.reader(fh)
                try:
                    self.header = next(reader)
                except StopIteration as ex:
                    raise ValueError(f"CSV file {self.path} is empty") from ex
                # store rows as sequences of strings
                self.rows = [tuple(row) for row in reader]

        def __len__(self) -> int:
            """Number of data rows (does not include header)."""
            return len(self.rows)

        # Public API: support indexing similar to numpy-style:
        #   hf[i] -> decode first column of row i (string) OR when i is a slice: decode
        #            first column of rows slice
        #   hf[i, j] -> raw cell (string) if j is int, or decoded concatenation if i is
        #   slice
        def __getitem__(self, key):
            # Tuple indexing: (row_index_or_slice, column_index)
            if isinstance(key, tuple):
                row_index, col_index = key
                if isinstance(row_index, slice):
                    # decode column col_index across sliced rows
                    words = [
                        self._cell_as_hex(r, col_index)
                        for r in range(*row_index.indices(len(self.rows)))
                    ]
                    return self._decode_hex_words(words)
                # single-row + column -> return raw cell string
                return self._cell_as_hex(row_index, col_index)

            # Single-indexing:
            if isinstance(key, int):
                # decode first column of row `key`
                return self._decode_hex_words([self._cell_as_hex(key, 0)])

            if isinstance(key, slice):
                # decode first column for the slice of rows
                words = [
                    self._cell_as_hex(r, 0) for r in range(*key.indices(len(self.rows)))
                ]
                return self._decode_hex_words(words)

            raise TypeError(f"Unsupported index type: {type(key)!r}")

        def _cell_as_hex(self, row: int, col: int) -> str:
            """Return the raw hex string from (row, col).
            Raises IndexError on bad indices."""
            # Normalize negative indices
            if row < 0:
                row = len(self.rows) + row
            if col < 0:
                # allow negative column indexing relative to row length
                col = len(self.rows[row]) + col
            try:
                return self.rows[row][col]
            except IndexError as ex:
                raise IndexError(f"Index out of range: row={row}, col={col}") from ex

        @staticmethod
        def _decode_hex_words(hex_words) -> str:
            """
            Decode an iterable of hex-strings into a UTF-8 string.
            Each hex word is parsed as a 64-bit integer (base 16), converted to
            8 big-endian bytes and concatenated. Trailing NUL bytes are removed
            before decoding with 'utf-8' using 'ignore' for invalid sequences.
            """
            out = bytearray()
            for hw in hex_words:
                # Accept formats like "0x..." or plain hex like "ABCDEF..."
                hw_clean = hw.strip()
                if hw_clean.startswith(("0x", "0X")):
                    hw_clean = hw_clean[2:]
                if not hw_clean:
                    continue
                # parse as integer (base 16)
                try:
                    val = int(hw_clean, 16)
                except ValueError as ex:
                    raise ValueError(f"Invalid hex word: {hw!r}") from ex
                out += val.to_bytes(8, byteorder="big", signed=False)

            # strip trailing null bytes and decode
            out = out.rstrip(b"\0")
            return out.decode("utf-8", errors="ignore")

        def iter_decoded_column(self, col: int = 0):
            """Generator yielding decoded strings for each row's given column."""
            for r in range(len(self.rows)):
                yield self._decode_hex_words([self._cell_as_hex(r, col)])

    def _prepare_data_file(self, fname=None) -> None:
        # retrieve the data file
        fname: str = fname or join(pwd, "data", "normal-testset-1.csv")
        data_file = self.HexFile(fname)

        # set the handler function according to the header
        header: str = data_file[0]
        module_name = data_file[1]
        file_content_handler = globals()[module_name].__dict__[header]

        # quick sanity check
        # we read random data from the file and validate availability
        # random sample some intervals and perform lookups
        lo1, lo2, hi1, hi2 = (
            default_rng(9494949494).integers(0, 200),
            default_rng(6666666666).integers(0, 200),
            default_rng(505050505050).integers(0, 200),
            default_rng(77777777).integers(0, 200),
        )

        try:
            file_content_handler(data_file[lo1:hi1, 0])
            file_content_handler(data_file[lo2:hi2, 0])
        except FileNotFoundError as e:
            raise RuntimeError(
                "Could not find the required data file for testing or it is corrupted."
                f" Please ensure that '{fname}' is present in the 'data'"
                " directory."
            ) from e
        except Exception as e:
            pass

        return module_name, file_content_handler

    @classmethod
    def _read_mean_var_seed_triplets(cls, filename):
        triplets = []
        with open(filename, "r") as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
            enc_token, seed_token = parts[-2], parts[-1]
            if enc_token.lower().startswith("0x"):
                h = enc_token[2:].replace("_", "")
                u64 = int(h, 16)
                hi: int = (u64 >> 32) & 0xffffffff
                lo: int = u64 & 0xffffffff
                mean = np.frombuffer(np.uint32(hi).tobytes(), dtype=np.float32)[
                    0
                ].item()
                var = np.frombuffer(np.uint32(lo).tobytes(), dtype=np.float32)[0].item()
                seed = int(seed_token, 16)
                triplets.append((float(mean), float(var), seed))
        return triplets

    # --- Hooks for subclasses ---
    def get_data_filename(self) -> str:
        raise NotImplementedError

    def draw_samples(self, mu: float, var: float, seed: int, size: int):
        raise NotImplementedError

    def null_mean(self, mu: float, var: float) -> float:
        # Default: shift mean by +0.5 * sigma
        return mu + 0.5 * np.sqrt(var)

    # --- Generic test using hooks ---
    def test_mean_hypothesis_large_population(self):
        # Prepare file
        fname: str = self.get_data_filename()
        self._prepare_data_file(fname)

        triplets = self._read_mean_var_seed_triplets(fname)
        assert len(triplets) > 0, "No mean/var/seed triplets loaded"

        # Start testing
        Nsamples = 65536
        p_values = []

        for mu, var, seed in triplets:
            if not np.isfinite(mu) or not np.isfinite(var) or var <= 0:
                continue

            samples = self.draw_samples(mu, var, seed, Nsamples)

            sigma = np.sqrt(var)
            null_mean = self.null_mean(mu, var)
            sample_mean = np.mean(samples)
            se = sigma / np.sqrt(Nsamples)
            z = (sample_mean - null_mean) / se
            p = math.erfc(abs(z) / math.sqrt(2))
            p_values.append(p)

        assert any(p < 0.05 for p in p_values), "No hypothesis produced p < 0.05"


class TestNormalDistributionProperties(DistributionPropertiesBase):
    def get_data_filename(self) -> str:
        return join(pwd, "data", "normal-testset-1.csv")

    def draw_samples(self, mu: float, var: float, seed: int, size: int):
        sigma = np.sqrt(var)
        rng: Generator = default_rng(seed)
        return rng.normal(loc=mu, scale=sigma, size=size)

    def null_mean(self, mu: float, var: float) -> float:
        return mu + 0.5 * np.sqrt(var)


class TestLaplaceDistributionProperties(DistributionPropertiesBase):
    def get_data_filename(self) -> str:
        return join(pwd, "data", "normal-testset-1.csv")

    def draw_samples(self, mu: float, var: float, seed: int, size: int):
        b = np.sqrt(max(var, 0.0) / 2.0)
        rng: Generator = default_rng(seed)
        return rng.laplace(loc=mu, scale=b, size=size)

class TestTDistributionProperties(DistributionPropertiesBase):
    df = 10

    def get_data_filename(self) -> str:
        return join(pwd, "data", "normal-testset-1.csv")

    def draw_samples(self, mu: float, var: float, seed: int, size: int):
        df = self.df
        df = int(df) if df and df > 2 else 10
        scale = np.sqrt(max(var, 0.0) * (df - 2.0) / df)
        rng: Generator = default_rng(seed)
        return mu + scale * rng.standard_t(df, size=size)


class TestUniformDistributionProperties(DistributionPropertiesBase):
    def get_data_filename(self) -> str:
        return join(pwd, "data", "normal-testset-1.csv")

    def draw_samples(self, mu: float, var: float, seed: int, size: int):
        half_width = np.sqrt(3.0 * max(var, 0.0))
        a = mu - half_width
        b = mu + half_width
        rng: Generator = default_rng(seed)
        return rng.uniform(low=a, high=b, size=size)


class TestExponentialDistributionProperties(DistributionPropertiesBase):
    def get_data_filename(self) -> str:
        return join(pwd, "data", "normal-testset-1.csv")

    def draw_samples(self, mu: float, var: float, seed: int, size: int):
        # Shifted exponential to match arbitrary mean/variance
        scale = np.sqrt(max(var, 0.0))
        loc = mu - scale
        rng: Generator = default_rng(seed)
        return loc + rng.exponential(scale=scale, size=size)