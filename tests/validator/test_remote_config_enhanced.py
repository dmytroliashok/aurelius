from unittest.mock import AsyncMock

from aurelius.validator.remote_config import RemoteConfig


class TestRemoteConfigStaleness:
    def test_initially_stale(self):
        rc = RemoteConfig()
        assert rc.is_stale is True

    async def test_not_stale_after_refresh(self):
        api = AsyncMock()
        api.get_remote_config.return_value = {"polling_interval_seconds": 120}
        rc = RemoteConfig(api_client=api, refresh_interval=0.1)
        await rc.refresh()
        assert rc.is_stale is False

    async def test_api_available_after_success(self):
        api = AsyncMock()
        api.get_remote_config.return_value = {}
        rc = RemoteConfig(api_client=api, refresh_interval=0.0)
        await rc.refresh()
        assert rc.api_available is True

    async def test_api_unavailable_after_failure(self):
        api = AsyncMock()
        api.get_remote_config.side_effect = ConnectionError("offline")
        rc = RemoteConfig(api_client=api, refresh_interval=0.0)
        await rc.refresh()
        assert rc.api_available is False

    def test_concordia_properties(self):
        rc = RemoteConfig()
        assert rc.concordia_timeout == 600
        assert rc.concordia_llm_model == "deepseek-chat"
        assert rc.concordia_image_tag == "v2.0.0"
        assert rc.classifier_update_interval_hours == 6
