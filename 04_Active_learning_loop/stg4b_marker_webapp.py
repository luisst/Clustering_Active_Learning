import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path
import urllib.parse
import os
import shutil
import socket
import argparse

# ---------- CONFIG ----------
PORT = 8000

# ----------------------------
def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def get_free_port(default=8000):
    try:
        with socketserver.TCPServer(("localhost", default), None) as s:
            return default
    except OSError:
        # find another free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            return s.getsockname()[1]

def start_server(port, server_root, stop_event):
    Handler = http.server.SimpleHTTPRequestHandler
    # Change to the server root directory so relative paths work
    os.chdir(server_root)

    class StoppableHTTPServer(socketserver.TCPServer):
        def serve_forever(self):
            while not stop_event.is_set():
                try:
                    self.handle_request()
                except OSError:
                    # Server socket was closed
                    break

    with StoppableHTTPServer(("", port), Handler) as httpd:
        httpd.timeout = 1  # Check stop_event every second
        print(f"[INFO] Serving {server_root} at http://localhost:{port}")
        httpd.serve_forever()

if __name__ == "__main__":

    html_marker = "AL1_marker.html"

    base_path_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline','MiniClusters')
    stg1_mp4_candidate_ex = base_path_ex.joinpath('input_mp4s')
    stg4_al_input_ex = base_path_ex.joinpath('STG_4','AL_input')

    parser = argparse.ArgumentParser()
    parser.add_argument('--stg1_mp4_candidate', default=stg1_mp4_candidate_ex, help='Stg1 MP4 candidate folder path')
    parser.add_argument('--stg4_al_folder', type=valid_path, default=stg4_al_input_ex, help='Stg4 AL input folder path')

    args = parser.parse_args()
    stg1_mp4_candidate = Path(args.stg1_mp4_candidate)
    stg4_al_folder = Path(args.stg4_al_folder)

    remote_server_root = stg4_al_folder.parent

    # For loop read csv files in stg4_al_folder
    # Each csv file has columns: cluster_id, start_time, end_time

    # List all csv files in the directory
    csv_files = list(stg4_al_folder.glob("*.csv"))

    for i, current_csv_file in enumerate(csv_files):
        # current_csv_file = csv_files[0]  # For testing, just take the first file
        long_video = current_csv_file.stem

        mp4_flag = False

        # Verify if mp4 candidate folder exists and has *.mp4 files
        if stg1_mp4_candidate.exists() and any(stg1_mp4_candidate.glob("*.mp4")):
            print(f"MP4 candidate folder is ready: {stg1_mp4_candidate}")
            mp4_flag = True
        else:
            print(f"Warning: {stg1_mp4_candidate} does not exist or has no mp4 files.")

        if mp4_flag:

            # Delete _ALinput suffix if exists
            if long_video.endswith("_ALinput"):
                long_video = long_video.split("_ALinput")[0]

            print(f"Processing {current_csv_file}: long_video = {long_video}")

            current_media_path = stg1_mp4_candidate.joinpath(f"{long_video}.wav")

            # Verify if long wav exists
            if not current_media_path.exists():
                print(f"Warning: {current_media_path} does not exist. Skipping.")
                continue

            # Set server root to the mp4 candidate folder
            server_root = remote_server_root.resolve()

            print(f"Server root: {server_root}")

            # Source HTML path (where this script is located)
            source_html_path = Path.home().joinpath('Dropbox', 'Source_2025', '04_Active_learning_loop') / html_marker
            
            # Ensure source HTML exists
            if not source_html_path.exists():
                raise FileNotFoundError(f"{source_html_path} not found. Place {html_marker} in the Active Learning Loop folder.")

            # Copy HTML file to mp4 folder if it's not already there
            html_path = server_root / html_marker
            if not html_path.exists():
                print(f"Copying {source_html_path} -> {html_path}")
                shutil.copy(source_html_path, html_path)

            print(f"Video file: {current_media_path}")

            port = get_free_port(PORT)

            # Create stop event for this server
            stop_event = threading.Event()

            # Start server
            thread = threading.Thread(target=start_server, args=(port, server_root, stop_event), daemon=True)
            thread.start()

            # Build URL
            video_url = urllib.parse.quote(current_media_path.name)
            url = f"http://localhost:{port}/{html_path.name}?video={video_url}"

            # Open browser
            print(f"[INFO] Opening browser at: {url}")
            webbrowser.open(url)

            try:
                if i < len(csv_files) - 1:  # Not the last file
                    input(f"\n[INFO] Press Enter to stop server and continue to next file ({i+2}/{len(csv_files)})...")
                else:  # Last file
                    input(f"\n[INFO] Press Enter to stop server and exit...")
                
                # Stop the server
                print("[INFO] Stopping server...")
                stop_event.set()
                thread.join(timeout=2)  # Wait up to 2 seconds for thread to finish
                
            except KeyboardInterrupt:
                print("\n[INFO] Shutting down...")
                stop_event.set()
                break

    print("[INFO] All files processed. Exiting.")