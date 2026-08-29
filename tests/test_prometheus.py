import unittest

from dfxlab.prometheus import select_metrics


class PrometheusTest(unittest.TestCase):
    def test_selects_and_aggregates_label_variants(self) -> None:
        raw = """
# HELP vllm:num_requests_running Running requests.
vllm:num_requests_running{model_name="a"} 2
vllm:num_requests_running{model_name="b"} 3
vllm:kv_cache_usage_perc 0.75
ignored_metric 99
"""
        self.assertEqual(
            select_metrics(raw),
            {
                "vllm:num_requests_running": 5.0,
                "vllm:kv_cache_usage_perc": 0.75,
            },
        )


if __name__ == "__main__":
    unittest.main()

