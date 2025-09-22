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


if __name__ == "__main__":

    html_marker = "ws_ActiveLearning1.html"
    html_activelearning = "ws_ActiveLearning1.html"

    base_path_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline','MiniClusters')
    stg1_long_wavs_ex = base_path_ex.joinpath('input_wavs')
    stg1_mp4_candidate_ex = base_path_ex.joinpath('input_mp4s')
    stg4_al_input_ex = base_path_ex.joinpath('STG_4','AL_input')

    # For loop read csv files in stg4_al_input_ex
    # Each csv file has columns: cluster_id, start_time, end_time

    for current_csv_file in stg4_al_input_ex.glob("*.csv"):
        long_wav = current_csv_file.stem
        print(f"Processing {current_csv_file}: long_wav = {long_wav}")

        current_media_path = stg1_long_wavs_ex.joinpath(f"{long_wav}.wav")

        mp4_flag = False

        # Verify if mp4 candidate folder exists and has *.mp4 files
        if stg1_mp4_candidate_ex.exists() and any(stg1_mp4_candidate_ex.glob("*.mp4")):
            print(f"MP4 candidate folder is ready: {stg1_mp4_candidate_ex}")
            mp4_flag = True
        else:
            print(f"Warning: {stg1_mp4_candidate_ex} does not exist or has no mp4 files.")

        # The server root will be the folder where this script is
        server_root = Path(r"C:\Users\luis2\Dropbox\Source_2025\GT_apps\ALwebapp").resolve()

        print(f"Server root: {server_root}")

        # Ensure HTML exists inside GT_apps
        html_path = server_root / html_marker
        if not html_path.exists():
            raise FileNotFoundError(f"{html_path} not found. Place wavesurfer.html inside GT_apps/")

        # Handle audio file
        audio_path = Path(current_media_path).resolve()

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
        url = f"http://localhost:{port}/{html_path.name}?media={audio_url}"

        # Open browser
        print(f"[INFO] Opening browser at: {url}")
        webbrowser.open(url)

        # Keep alive (important for VSCode)
        try:
            thread.join()
        except KeyboardInterrupt:
            print("\n[INFO] Server stopped.")