import unittest

import numpy as np
import tenseal as ts

from flowertune_llm.she.ckks_packing import (
    pack_column_blocks,
    packed_column_transforms,
)


class CkksPackingTest(unittest.TestCase):
    def test_four_columns_share_one_column_major_block(self):
        columns = np.arange(12, dtype=np.float64).reshape(3, 4)

        blocks = pack_column_blocks(columns)

        self.assertEqual(len(blocks), 1)
        values, column_count = blocks[0]
        self.assertEqual(column_count, 4)
        np.testing.assert_array_equal(values, columns.T.reshape(-1))

    def test_tail_block_preserves_its_column_count(self):
        columns = np.arange(15, dtype=np.float64).reshape(3, 5)

        blocks = pack_column_blocks(columns)

        self.assertEqual([column_count for _, column_count in blocks], [4, 1])
        np.testing.assert_array_equal(blocks[1][0], columns[:, 4])

    def test_transforms_recover_each_plaintext_product_column(self):
        rng = np.random.default_rng(7)
        plain_b = rng.standard_normal((9, 3))
        columns = rng.standard_normal((3, 4))
        packed, column_count = pack_column_blocks(columns)[0]

        recovered = np.column_stack(
            [
                packed @ transform
                for transform in packed_column_transforms(plain_b, column_count)
            ]
        )

        np.testing.assert_allclose(recovered, plain_b @ columns)

    def test_ckks_transforms_preserve_column_order(self):
        rng = np.random.default_rng(13)
        plain_b = rng.standard_normal((8, 3))
        columns = rng.standard_normal((3, 4))
        packed, column_count = pack_column_blocks(columns)[0]
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 60],
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        encrypted = ts.ckks_vector(context, packed)

        recovered = np.column_stack(
            [
                np.asarray(
                    encrypted.mm(ts.plain_tensor(transform)).decrypt()
                )
                for transform in packed_column_transforms(plain_b, column_count)
            ]
        )

        np.testing.assert_allclose(recovered, plain_b @ columns, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
