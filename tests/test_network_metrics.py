from io import BytesIO
import unittest

import numpy as np

from flowertune_llm.she.network_metrics import (
    network_byte_totals,
    serialized_ndarray_size,
)


class NetworkMetricsTest(unittest.TestCase):
    def test_serialized_size_matches_flower_numpy_payload(self):
        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        buffer = BytesIO()
        np.save(buffer, array, allow_pickle=False)

        self.assertEqual(serialized_ndarray_size(array), len(buffer.getvalue()))

    def test_totals_include_plaintext_and_ciphertext_in_both_directions(self):
        values = network_byte_totals(
            client_count=4,
            plaintext_upload_bytes=400,
            ciphertext_upload_bytes=800,
            plaintext_download_bytes=50,
            ciphertext_download_bytes=70,
        )

        self.assertEqual(values["model_upload_bytes"], 1200)
        self.assertEqual(values["model_upload_per_client_bytes"], 300)
        self.assertEqual(values["model_download_per_client_bytes"], 120)
        self.assertEqual(values["model_roundtrip_bytes"], 420)
        self.assertEqual(values["model_roundtrip_broadcast_bytes"], 1680)

    def test_client_count_must_be_positive(self):
        with self.assertRaises(ValueError):
            network_byte_totals(
                client_count=0,
                plaintext_upload_bytes=0,
                ciphertext_upload_bytes=0,
                plaintext_download_bytes=0,
                ciphertext_download_bytes=0,
            )


if __name__ == "__main__":
    unittest.main()
