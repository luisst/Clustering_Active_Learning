import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path
import urllib.parse

import os
import shutil
import socket

# ---------- CONFIG ----------
PORT = 8000
HTML_FILE = "wavesurfer.html"
AUDIO_FILE = r"C:\Users\luis2\Dropbox\Source_2025\GT_apps\IC-Arienna_ID-76_00236.wav"
# ----------------------------
def get_free_port(default=8000):
    try:
        with socketserver.TCPServer(("localhost", default), None) as s:
            return default
    except OSError:
        # find another free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            return s.getsockname()[1]

def start_server(port, server_root):
    Handler = http.server.SimpleHTTPRequestHandler
    # Change to the server root directory so relative paths work
    os.chdir(server_root)

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"[INFO] Serving {server_root} at http://localhost:{port}")
        httpd.serve_forever()

def main():
    # The server root will be the folder where this script is
    server_root = Path(r"C:\Users\luis2\Dropbox\Source_2025\GT_apps").resolve()

    print(f"Server root: {server_root}")

    # Ensure HTML exists inside GT_apps
    html_path = server_root / HTML_FILE
    if not html_path.exists():
        raise FileNotFoundError(f"{html_path} not found. Place wavesurfer.html inside GT_apps/")

    # Handle audio file
    audio_path = Path(AUDIO_FILE).resolve()

    print(f"Audio file: {audio_path}")

    # Copy audio into server root if it's not already there
    if not str(audio_path).startswith(str(server_root)):
        local_copy = server_root / audio_path.name
        if not local_copy.exists():
            print(f"Copying {audio_path} -> {local_copy}")
            shutil.copy(audio_path, local_copy)
        audio_path = local_copy

    port = get_free_port(PORT)

    # Start server
    thread = threading.Thread(target=start_server, args=(port, server_root), daemon=True)
    thread.start()

    # Build URL
    audio_url = urllib.parse.quote(audio_path.name)
    url = f"http://localhost:{port}/{html_path.name}?audio={audio_url}"

    # Open browser
    print(f"[INFO] Opening browser at: {url}")
    webbrowser.open(url)

    # Keep alive (important for VSCode)
    try:
        thread.join()
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped.")


if __name__ == "__main__":
    main()
