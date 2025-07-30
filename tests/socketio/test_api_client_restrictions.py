"""Tests for API client connection restrictions."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from seed_vc.socketio.api import (
    APIRouterVCModel,
    _get_connected_clients_count,
    require_no_connected_clients_method,
)
from seed_vc.socketio.schemas import (
    ModelParametersRequest,
    ModelReloadRequest,
)


class TestAPIClientRestrictions:
    """Test cases for API client connection restrictions."""

    def setup_method(self):
        """Setup test fixtures."""
        # Mock VoiceConverter
        self.mock_model = MagicMock()
        self.mock_model.model_set = [{"sampling_rate": 44100}]
        self.mock_model.block_time = 0.18
        self.mock_model.crossfade_time = 0.04
        self.mock_model.extra_time_ce = 0.25
        self.mock_model.extra_time = 0.5
        self.mock_model.extra_time_right = 0.02
        self.mock_model.diffusion_steps = 4
        self.mock_model.max_prompt_length = 15.0
        self.mock_model.inference_cfg_rate = 0.7
        self.mock_model.use_vad = True

        # Create API router instance
        self.api_router = APIRouterVCModel(self.mock_model)

    def test_get_connected_clients_count_no_clients(self):
        """Test getting connected clients count when no clients connected."""
        with patch("seed_vc.socketio.server.client_converters", {}):
            with patch("seed_vc.socketio.server.converter_lock", MagicMock()):
                count = _get_connected_clients_count()
                assert count == 0

    def test_get_connected_clients_count_with_clients(self):
        """Test getting connected clients count when clients are connected."""
        mock_clients = {"client1": MagicMock(), "client2": MagicMock()}
        with patch("seed_vc.socketio.server.client_converters", mock_clients):
            with patch("seed_vc.socketio.server.converter_lock", MagicMock()):
                count = _get_connected_clients_count()
                assert count == 2

    def test_get_connected_clients_count_import_error(self):
        """Test getting connected clients count when server module unavailable."""
        with patch("builtins.__import__", side_effect=ImportError):
            count = _get_connected_clients_count()
            assert count == 0

    def test_require_no_connected_clients_decorator_no_clients(self):
        """Test decorator allows execution when no clients connected."""

        class TestRouter(APIRouterVCModel):
            @require_no_connected_clients_method
            def test_method(self):
                return "success"

        router = TestRouter(self.mock_model, client_count_checker=lambda: 0)
        result = router.test_method()
        assert result == "success"

    def test_require_no_connected_clients_decorator_with_clients(self):
        """Test decorator blocks execution when clients are connected."""

        class TestRouter(APIRouterVCModel):
            @require_no_connected_clients_method
            def test_method(self):
                return "success"

        router = TestRouter(self.mock_model, client_count_checker=lambda: 2)
        with pytest.raises(HTTPException) as exc_info:
            router.test_method()

        assert exc_info.value.status_code == 409
        assert "2 client(s) are connected" in str(exc_info.value.detail)
        assert "disconnect all clients first" in str(exc_info.value.detail)

    def test_update_model_parameters_no_clients_allowed(self):
        """Test update_model_parameters works when no clients connected."""
        request = ModelParametersRequest(diffusion_steps=8)

        with patch("seed_vc.socketio.api._get_connected_clients_count", return_value=0):
            response = self.api_router.update_model_parameters(request)

            assert response.status_code == 200
            response_data = response.body.decode()
            assert "updated successfully" in response_data

    def test_update_model_parameters_with_clients_blocked(self):
        """Test update_model_parameters blocked when clients connected."""
        request = ModelParametersRequest(diffusion_steps=8)

        # Create API router with client count checker that returns 1
        api_router_with_checker = APIRouterVCModel(self.mock_model, client_count_checker=lambda: 1)

        with pytest.raises(HTTPException) as exc_info:
            api_router_with_checker.update_model_parameters(request)

        assert exc_info.value.status_code == 409
        assert "1 client(s) are connected" in str(exc_info.value.detail)

    def test_reload_model_no_clients_allowed(self):
        """Test reload_model works when no clients connected."""
        request = ModelReloadRequest()

        # Mock the model loading methods
        self.mock_model._load_models.return_value = [{"sampling_rate": 44100}]
        self.mock_model._init_buffers.return_value = None

        with patch("seed_vc.socketio.api._get_connected_clients_count", return_value=0):
            response = self.api_router.reload_model(request)

            assert response.status_code == 200
            response_data = response.body.decode()
            assert "reloaded successfully" in response_data

    def test_reload_model_with_clients_blocked(self):
        """Test reload_model blocked when clients connected."""
        request = ModelReloadRequest()

        # Create API router with client count checker that returns 3
        api_router_with_checker = APIRouterVCModel(self.mock_model, client_count_checker=lambda: 3)

        with pytest.raises(HTTPException) as exc_info:
            api_router_with_checker.reload_model(request)

        assert exc_info.value.status_code == 409
        assert "3 client(s) are connected" in str(exc_info.value.detail)

    def test_other_api_methods_not_restricted(self):
        """Test that other API methods are not restricted by client connections."""
        # Test reference audio update (should work with clients connected)
        from seed_vc.socketio.schemas import ReferenceAudioRequest

        with patch("seed_vc.socketio.api._get_connected_clients_count", return_value=2):
            with patch.object(
                self.api_router, "_validate_file_path", return_value="/test/path.wav"
            ):
                with patch("librosa.load", return_value=([0.1, 0.2, 0.3], 44100)):
                    request = ReferenceAudioRequest(file_path="/test/path.wav")

                    # This should work even with clients connected
                    response = self.api_router.update_reference_audio(request)
                    assert response.status_code == 200

        # Test conversion mode change (should work with clients connected)
        from seed_vc.socketio.schemas import ConversionModeRequest

        with patch("seed_vc.socketio.api._get_connected_clients_count", return_value=2):
            request = ConversionModeRequest(mode="passthrough")

            # This should work even with clients connected
            response = self.api_router.change_conversion_mode(request)
            assert response.status_code == 200

    def test_multiple_restrictions_can_be_added_easily(self):
        """Test that the decorator can be easily applied to additional methods."""

        class ExtendedAPIRouter(APIRouterVCModel):
            @require_no_connected_clients_method
            def custom_restricted_method(self):
                """Custom method that requires no connected clients."""
                return {"status": "success"}

            def custom_unrestricted_method(self):
                """Custom method that allows connected clients."""
                return {"status": "success"}

        # Test restricted method with clients connected
        router_with_clients = ExtendedAPIRouter(self.mock_model, client_count_checker=lambda: 1)
        with pytest.raises(HTTPException) as exc_info:
            router_with_clients.custom_restricted_method()
        assert exc_info.value.status_code == 409

        # Test unrestricted method with clients connected
        result = router_with_clients.custom_unrestricted_method()
        assert result["status"] == "success"

        # Test restricted method without clients
        router_no_clients = ExtendedAPIRouter(self.mock_model, client_count_checker=lambda: 0)
        result = router_no_clients.custom_restricted_method()
        assert result["status"] == "success"

    def test_error_message_formatting(self):
        """Test that error messages are properly formatted for different client counts."""

        class TestRouter(APIRouterVCModel):
            @require_no_connected_clients_method
            def test_method(self):
                return "success"

        # Test singular message
        router = TestRouter(self.mock_model, client_count_checker=lambda: 1)
        with pytest.raises(HTTPException) as exc_info:
            router.test_method()
        assert "1 client(s) are connected" in str(exc_info.value.detail)

        # Test plural message
        router = TestRouter(self.mock_model, client_count_checker=lambda: 5)
        with pytest.raises(HTTPException) as exc_info:
            router.test_method()
        assert "5 client(s) are connected" in str(exc_info.value.detail)
