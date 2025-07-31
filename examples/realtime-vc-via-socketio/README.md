# Real-Time Voice Conversion Using SocketIO

This example explains how to use SocketIO for real-time voice conversion with both server and client components.

## Environment Setup

Install the necessary modules for SocketIO.
On the **client** machine, you can install only the required minimal modules using the `--only` option.

```bash
$ cd /path/to/custom-seed-vc/

# Install modules for the server
$ poetry install

# Install minimal modules for the client
$ poetry install --only client
```

## How to Use

### Starting the Server

To start the server, run the following commands. It may take some time to initialize.

```bash
$ cd /path/to/custom-seed-vc/
$ poetry run python seed_vc/socketio/server.py
```

Once the server is up, you'll see output like this:

```bash
$ poetry run python seed_vc/socketio/server.py
[15:09:22] [SERVER] [INFO] 🚀 Starting server imports...
[15:09:22] [SERVER] [INFO] ⏳ Importing seed_vc modules (this may take a while)...
[15:09:37] [SERVER] [INFO] ✅ All imports completed!
[15:09:37] [SERVER] [INFO] 🎙️  Starting voice conversion server on 0.0.0.0:5000 ...
[15:09:37] [SERVER] [INFO] 🔄 Initializing global VoiceConverter...
[15:09:53] [SERVER] [INFO] ✅ Global VoiceConverter ready!
[15:09:53] [SERVER] [INFO] 🌟 Ready to accept connections!
```

### Starting the Client

To start the client, run the following.
Make sure the server is running beforehand.

```bash
$ cd /path/to/custom-seed-vc/
$ poetry run python seed_vc/socketio/client.py
```

When the client successfully connects to the server, you'll see something like:

```bash
$ poetry run python seed_vc/socketio/client.py
[15:11:57] [CLIENT] [INFO] 🔗 Connecting to http://localhost:5000
[15:11:57] [CLIENT] [INFO] 🔗 Connected to server
[15:11:57] [CLIENT] [INFO] 🎧 Streaming... (Ctrl+C to stop)
```

At this point, audio from the client’s microphone will be sent to the server, converted in real time, and the transformed audio will be played through the client’s speakers.

If you experience lag between the server and client, restarting just the client might help reduce it.

### Changing Settings via FastAPI

While the server is running, you can use the API to change voice conversion settings.
The following APIs are available:

* **Switch Conversion Modes**:

  * `convert`: Default mode, performs voice conversion
  * `passthrough`: Outputs input audio without conversion
  * `silence`: Outputs silence

```bash
$ curl -X POST "http://localhost:5000/api/v1/mode" \
    -H "Content-Type: application/json" \
    -d '{"mode": "passthrough"}'
```

* **Change Reference Audio**:

```bash
$ curl -X POST "http://localhost:5000/api/v1/reference" \
    -H "Content-Type: application/json" \
    -d '{"file_path": "assets/examples/reference/trump_0.wav"}'

# For security, only files under assets/examples/reference/ can be used by default.
# To allow other directories, use --allowed-audio-dirs when starting the server:
$ poetry run python seed_vc/socketio/server.py --allowed-audio-dirs /path/to/your/audio/dir
```

* **Adjust Voice Conversion Parameters**:

```bash
$ curl -X POST "http://localhost:5000/api/v1/parameters" \
    -H "Content-Type: application/json" \
    -d '{"block_time": 0.18,"extra_time_ce": 0.5}'
```

* **Reload the Voice Conversion Model**:

```bash
# Reload the default voice conversion model
$ curl -X POST "http://localhost:5000/api/v1/reload" \
    -H "Content-Type: application/json" \
    -d '{}'

# Load a fine-tuned model
$ curl -X POST "http://localhost:5000/api/v1/reload" \
    -H "Content-Type: application/json" \
    -d '{"checkpoint_path": "examples/fine-tuning/runs/my_run/ft_model.pth", "config_path": "examples/fine-tuning/runs/my_run/config_dit_mel_seed_uvit_xlsr_tiny.yml"}'
```

To see full API documentation, go to:
`http://localhost:5000/docs` in your browser while the server is running.

## Running with Docker

You can also run the server using Docker. Follow the steps below:

### Build the Docker Image

```bash
$ export COMPOSE_FILE=docker/socketio/docker-compose.yml
$ docker compose build
```

### Start and Stop Docker Container

```bash
# Start the container (default port is 5000)
$ docker compose up -d

# Stop the container
$ docker compose down
```

After the container is running, you can start the client as usual to connect to the server inside the container for real-time voice conversion.
