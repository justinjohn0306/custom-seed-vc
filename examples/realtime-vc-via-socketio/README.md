# SocketIOを用いたリアルタイム音声変換

本Exampleでは、SocketIOを用いた、リアルタイム音声変換のサーバーとクライアントの利用方法を説明します。

## 環境構築

SocketIO用のモジュールをインストールしてください。
Client側のマシンでは、`--only`オプションを使うことで必要最小限のモジュールのみをインストールできます。

```bash
$ cd /path/to/custom-seed-vc/

# Server側でのモジュールをインストール
$ poetry install

# Client側でのモジュールをインストール
$ poetry install --only client
```

## 使い方

### Serverの起動

Serverを起動するには、以下のコマンドを実行します。
起動には時間がかかる場合があります。

```bash
$ cd /path/to/custom-seed-vc/
$ poetry run python seed_vc/socketio/server.py
```

起動が完了すると次のようなメッセージが表示されます。
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

### Clientの起動

Clientを起動するには、以下のコマンドを実行します。
Serverが起動していることを確認してから実行してください。

```bash
$ cd /path/to/custom-seed-vc/
$ poetry run python seed_vc/socketio/client.py
```

起動が完了して、Serverと接続できると次のようなメッセージが表示されます。

```bash
$ poetry run python seed_vc/socketio/client.py
[15:11:57] [CLIENT] [INFO] 🔗 Connecting to http://localhost:5000
[15:11:57] [CLIENT] [INFO] 🔗 Connected to server
[15:11:57] [CLIENT] [INFO] 🎧 Streaming... (Ctrl+C to stop)
```

この状態で、Client側のマイクから音声を入力すると、Server側で音声変換が行われ、変換された音声がClient側のスピーカーから再生されます。

ServerとClient間にラグがあると感じる場合は一度、Client側だけを再起動することで改善される場合があります。

### FastAPIによる設定の変更

Serverを起動した状態でAPIを叩くと、音声変換の設定を変更することができます。
次のようなAPIが用意されています。

- 変換モードの変更
    - convert : 音声変換を行うモード（デフォルト）
    - passthrough : 音声変換を行わず、入力音声をそのまま出力するモード
    - silence : 音声を無音にするモード
```bash
$ curl -X POST "http://localhost:5000/api/v1/mode" \
    -H "Content-Type: application/json" \
    -d '{"mode": "passthrough"}'
```

- リファレンス音声の変更
```bash
$ curl -X POST "http://localhost:5000/api/v1/reference" \
    -H "Content-Type: application/json" \
    -d '{"file_path": "assets/examples/reference/trump_0.wav"}'

# セキュリティの関係からデフォルトでは assets/examples/reference/ 以下の音声ファイルのみを指定できます。
# 別のディレクトリの音声ファイルを指定したい場合は、server.pyを起動する際に、`--allowed-audio-dirs`オプションをつけて起動してください。
$ poetry run python seed_vc/socketio/server.py --allowed-audio-dirs /path/to/your/audio/dir
```

- 音声変換モデルの各種パラメータの変更
```bash
$ curl -X POST "http://localhost:5000/api/v1/parameters" \
    -H "Content-Type: application/json" \
    -d '{"block_time": 0.18,"extra_time_ce": 0.5}'
```

- 音声変換モデルの再読み込み
```bash
# デフォルトの音声変換モデルを再読み込み
$ curl -X POST "http://localhost:5000/api/v1/reload" \
    -H "Content-Type: application/json" \
    -d '{}'

# ファインチューニングした音声変換モデルを読み込み
$ curl -X POST "http://localhost:5000/api/v1/reload" \
    -H "Content-Type: application/json" \
    -d '{"checkpoint_path": "examples/fine-tuning/runs/my_run/ft_model.pth", "config_path": "examples/fine-tuning/runs/my_run/config_dit_mel_seed_uvit_xlsr_tiny.yml"}'
```

詳しいAPIの仕様は、Serverを起動した状態で、ブラウザから`http://localhost:5000/docs`にアクセスすることで確認できます。

## Dockerを用いた実行

Dockerを用いてServerを実行することもできます。以下の手順で実行できます。

### Dockerイメージのビルド

```bash
$ export COMPOSE_FILE=docker/socketio/docker-compose.yml
$ docker compose build
```

### Dockerコンテナの起動・停止

```bash
# コンテナの起動
# デフォルトではポート5000で起動
$ docker compose up -d

# コンテナの停止
$ docker compose down
```

コンテナの起動後、通常と同じようにClientを起動してコンテナ上のサーバーに接続することでリアルタイム音声変換を行うことができます。
