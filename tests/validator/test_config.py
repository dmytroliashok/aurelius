from aurelius.config import Config, LocalConfig


class TestLocalConfig:
    def test_backward_compatible_alias(self):
        assert Config is LocalConfig

    def test_default_network(self):
        assert LocalConfig.NETWORK in ("finney", "test", "local")

    def test_default_netuid(self):
        assert isinstance(LocalConfig.NETUID, int)

    def test_default_wallet(self):
        assert LocalConfig.WALLET_NAME == "default"
        assert LocalConfig.WALLET_HOTKEY == "default"

    def test_llm_defaults(self):
        assert LocalConfig.LLM_MODEL == "deepseek-chat"
        assert "deepseek" in LocalConfig.LLM_BASE_URL

    def test_api_defaults(self):
        assert "localhost" in LocalConfig.CENTRAL_API_URL
