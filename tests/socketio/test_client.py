"""Tests for Socket.IO client connection error handling."""

from unittest.mock import patch

import socketio

import seed_vc.socketio.client as client_module
from seed_vc.socketio.client import on_connect_error, on_connection_error
from seed_vc.socketio.schemas import ConnectionErrorType


class TestClientConnectionErrors:
    """Test cases for client connection error handling."""

    def setup_method(self):
        """Setup test state."""
        # Clear global connection error details
        client_module.connection_error_details.clear()

    def test_chunk_size_mismatch_error_handling(self):
        """Test that chunk size mismatch errors are properly handled."""
        error_data = {
            "error": ConnectionErrorType.CHUNK_SIZE_MISMATCH.value,
            "client_chunk_size": 1000,
            "expected_chunk_size": 7938,
            "block_time": 0.18,
            "sample_rate": 44100,
        }

        # Simulate server sending connection_error
        on_connection_error(error_data)

        # Check that error details are stored
        assert (
            client_module.connection_error_details["error"]
            == ConnectionErrorType.CHUNK_SIZE_MISMATCH.value
        )
        assert client_module.connection_error_details["client_chunk_size"] == 1000
        assert client_module.connection_error_details["expected_chunk_size"] == 7938

    def test_max_clients_reached_error_handling(self):
        """Test that max clients reached errors are properly handled."""
        error_data = {
            "error": ConnectionErrorType.MAX_CLIENTS_REACHED.value,
            "message": "Maximum number of clients (1) already connected",
        }

        # Simulate server sending connection_error
        on_connection_error(error_data)

        # Check that error details are stored
        assert (
            client_module.connection_error_details["error"]
            == ConnectionErrorType.MAX_CLIENTS_REACHED.value
        )
        assert client_module.connection_error_details["message"] == (
            "Maximum number of clients (1) already connected"
        )

    def test_connection_error_logging_chunk_size_mismatch(self, caplog):
        """Test that chunk size mismatch errors are logged properly."""
        # Set up error details
        client_module.connection_error_details.update(
            {
                "error": ConnectionErrorType.CHUNK_SIZE_MISMATCH.value,
                "client_chunk_size": 1000,
                "expected_chunk_size": 7938,
                "block_time": 0.18,
                "sample_rate": 44100,
            }
        )

        with patch("seed_vc.socketio.client.sio") as mock_sio:
            # Mock connection failure
            mock_sio.connect.side_effect = socketio.exceptions.ConnectionError("Connection failed")

            # Import here to avoid circular import and capture the actual error handling
            from seed_vc.socketio.client import main

            with patch("sys.argv", ["client.py", "--host", "localhost", "--port", "5000"]):
                with patch("seed_vc.socketio.client.logger") as mock_logger:
                    try:
                        main()
                    except SystemExit:
                        pass  # main() calls return which may cause SystemExit in test

                    # Check that appropriate error messages were logged
                    mock_logger.error.assert_any_call(
                        "❌ Failed to connect to server: %s", mock_sio.connect.side_effect
                    )

                    # Check chunk size mismatch specific logging
                    calls = [
                        call
                        for call in mock_logger.error.call_args_list
                        if "Chunk size mismatch" in str(call)
                    ]
                    assert len(calls) > 0

    def test_connection_error_logging_max_clients_reached(self, caplog):
        """Test that max clients reached errors are logged properly."""
        # Set up error details
        client_module.connection_error_details.update(
            {
                "error": ConnectionErrorType.MAX_CLIENTS_REACHED.value,
                "message": "Maximum number of clients (1) already connected",
            }
        )

        with patch("seed_vc.socketio.client.sio") as mock_sio:
            # Mock connection failure
            mock_sio.connect.side_effect = socketio.exceptions.ConnectionError("Connection failed")

            # Import here to avoid circular import and capture the actual error handling
            from seed_vc.socketio.client import main

            with patch("sys.argv", ["client.py", "--host", "localhost", "--port", "5000"]):
                with patch("seed_vc.socketio.client.logger") as mock_logger:
                    try:
                        main()
                    except SystemExit:
                        pass  # main() calls return which may cause SystemExit in test

                    # Check that appropriate error messages were logged
                    mock_logger.error.assert_any_call(
                        "❌ Failed to connect to server: %s", mock_sio.connect.side_effect
                    )

                    # Check max clients specific logging
                    calls = [
                        call
                        for call in mock_logger.error.call_args_list
                        if "Connection rejected" in str(call)
                    ]
                    assert len(calls) > 0

                    wait_calls = [
                        call
                        for call in mock_logger.error.call_args_list
                        if "wait for another client" in str(call)
                    ]
                    assert len(wait_calls) > 0

    def test_unknown_error_handling(self):
        """Test that unknown errors are handled gracefully."""
        error_data = {"error": "unknown_error", "message": "Some unknown error occurred"}

        # Simulate server sending connection_error
        on_connection_error(error_data)

        # Check that error details are stored
        assert client_module.connection_error_details["error"] == "unknown_error"
        assert client_module.connection_error_details["message"] == "Some unknown error occurred"

    def test_connect_error_max_clients_reached(self):
        """Test that connect_error event handles max_clients_reached properly."""
        error_data = {
            "error": ConnectionErrorType.MAX_CLIENTS_REACHED.value,
            "message": "Maximum number of clients (1) already connected",
        }

        # Simulate server sending connect_error (when connection is refused)
        on_connect_error(error_data)

        # Check that error details are stored
        assert (
            client_module.connection_error_details["error"]
            == ConnectionErrorType.MAX_CLIENTS_REACHED.value
        )
        assert client_module.connection_error_details["message"] == (
            "Maximum number of clients (1) already connected"
        )

    def test_connect_error_chunk_size_mismatch(self):
        """Test that connect_error event handles chunk_size_mismatch properly."""
        error_data = {
            "error": ConnectionErrorType.CHUNK_SIZE_MISMATCH.value,
            "client_chunk_size": 1000,
            "expected_chunk_size": 7938,
            "block_time": 0.18,
            "sample_rate": 44100,
        }

        # Simulate server sending connect_error (when connection is refused)
        on_connect_error(error_data)

        # Check that error details are stored
        assert (
            client_module.connection_error_details["error"]
            == ConnectionErrorType.CHUNK_SIZE_MISMATCH.value
        )
        assert client_module.connection_error_details["client_chunk_size"] == 1000
        assert client_module.connection_error_details["expected_chunk_size"] == 7938
