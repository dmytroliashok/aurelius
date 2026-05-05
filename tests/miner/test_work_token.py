from aurelius.miner.work_token import generate_work_id, recompute_work_id


class TestGenerateWorkId:
    def test_returns_work_id_result(self):
        result = generate_work_id({"name": "test"}, "hotkey123")
        assert len(result.work_id) == 64  # sha256 hex
        int(result.work_id, 16)  # valid hex
        assert len(result.nonce) == 32  # 128-bit hex nonce
        assert result.time_ns.isdigit()

    def test_unique_per_call(self):
        config = {"name": "test"}
        ids = {generate_work_id(config, "hotkey123").work_id for _ in range(100)}
        assert len(ids) == 100  # all unique due to nonce

    def test_different_configs_different_ids(self):
        id1 = generate_work_id({"name": "config_a"}, "hotkey123").work_id
        id2 = generate_work_id({"name": "config_b"}, "hotkey123").work_id
        assert id1 != id2

    def test_different_hotkeys_different_ids(self):
        config = {"name": "test"}
        id1 = generate_work_id(config, "hotkey_a").work_id
        id2 = generate_work_id(config, "hotkey_b").work_id
        assert id1 != id2

    def test_recompute_matches(self):
        config = {"name": "test"}
        result = generate_work_id(config, "hotkey123")
        recomputed = recompute_work_id(config, "hotkey123", result.time_ns, result.nonce)
        assert recomputed == result.work_id

    def test_recompute_different_config_fails(self):
        result = generate_work_id({"name": "test"}, "hotkey123")
        recomputed = recompute_work_id({"name": "other"}, "hotkey123", result.time_ns, result.nonce)
        assert recomputed != result.work_id
