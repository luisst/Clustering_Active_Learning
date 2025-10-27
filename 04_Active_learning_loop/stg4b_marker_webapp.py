import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path
import urllib.parse
import os
import shutil
import socket

import time
import argparse

# ---------- CONFIG ----------
PORT = 8000

# ----------------------------
def get_free_port(default=8000):
    # Try a range of ports starting from default
    for port in range(default, default + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    
    # If no port found in range, let OS assign one
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def start_server(port, server_root, stop_event):
    # Custom handler that properly supports HTTP Range requests for video seeking
    class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Enable CORS and proper caching for video files
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Access-Control-Allow-Origin', '*')
            super().end_headers()

        def send_head(self):
            """Override to properly handle Range requests for video files"""
            path = self.translate_path(self.path)

            # Check if file exists
            try:
                f = open(path, 'rb')
            except OSError:
                return super().send_head()

            # Get file stats
            fs = os.fstat(f.fileno())
            file_len = fs.st_size

            # Check for Range header
            range_header = self.headers.get('Range')

            if range_header:
                # Parse range header (format: "bytes=start-end")
                try:
                    range_match = range_header.replace('bytes=', '').split('-')
                    start = int(range_match[0]) if range_match[0] else 0
                    end = int(range_match[1]) if range_match[1] else file_len - 1

                    # Validate range
                    if start >= file_len:
                        f.close()
                        self.send_error(416, "Requested Range Not Satisfiable")
                        return None

                    # Adjust end if necessary
                    if end >= file_len:
                        end = file_len - 1

                    # Send 206 Partial Content response
                    self.send_response(206)
                    self.send_header('Content-Type', self.guess_type(path))
                    self.send_header('Content-Range', f'bytes {start}-{end}/{file_len}')
                    self.send_header('Content-Length', str(end - start + 1))
                    self.send_header('Last-Modified', self.date_time_string(fs.st_mtime))
                    self.end_headers()

                    # Seek to start position
                    f.seek(start)
                    return f

                except (ValueError, IndexError):
                    # Invalid range header, fall back to full file
                    pass

            # No range header or invalid range - send full file
            self.send_response(200)
            self.send_header('Content-Type', self.guess_type(path))
            self.send_header('Content-Length', str(file_len))
            self.send_header('Last-Modified', self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f

        def copyfile(self, source, outputfile):
            """Copy file with range support"""
            # Read range from headers if it was a 206 response
            if hasattr(self, '_range_bytes'):
                # Copy only the requested range
                shutil.copyfileobj(source, outputfile, length=self._range_bytes)
            else:
                # Copy entire file
                super().copyfile(source, outputfile)

    Handler = RangeRequestHandler
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
        print(f"[INFO] HTTP Range requests enabled for video seeking")
        httpd.serve_forever()

def valid_path(path):
    if os.path.exists(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")



if __name__ == "__main__":

    html_marker = "AL1_marker.html"

    base_path_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline','TestAO-Irma')
    stg1_mp4_candidate_ex = base_path_ex.joinpath('input_mp4s')
    stg4_al_input_ex = base_path_ex.joinpath('STG_4','STG4_LP1')

    source_path_ex = Path.home().joinpath('Dropbox', 'Source_2025', '04_Active_learning_loop')

    parser = argparse.ArgumentParser()
    parser.add_argument('--stg1_mp4_candidate', default=stg1_mp4_candidate_ex, help='Stg1 MP4 candidate folder path')
    parser.add_argument('--stg4_al_folder', type=valid_path, default=stg4_al_input_ex, help='Stg4 AL input folder path')
    parser.add_argument('--source_path', type=valid_path, default=source_path_ex, help='Source path for HTML file')


    args = parser.parse_args()
    stg1_mp4_candidate = Path(args.stg1_mp4_candidate)
    stg4_al_folder = Path(args.stg4_al_folder)
    source_path = Path(args.source_path)

    json_file_path = source_path.joinpath("images_june4.json")

    remote_server_root = base_path_ex


    # For loop read csv files in stg4_al_folder
    # Each csv file has columns: cluster_id, start_time, end_time

    # List all csv files in the directory
    csv_files = list(stg4_al_folder.glob("*.csv"))

    print(f"[INFO] Found {len(csv_files)} CSV files in {stg4_al_folder}")
    
    # Print each csv file name
    for csv_file in csv_files:
        print(f" - {csv_file.name}")


    for i, current_csv_file in enumerate(csv_files):
        current_csv_file = csv_files[i]  # For testing, just take the first file
        long_media = current_csv_file.stem

        mp4_flag = False

        # Verify if mp4 candidate folder exists and has *.mp4 files
        if stg1_mp4_candidate_ex.exists() and any(stg1_mp4_candidate_ex.glob("*.mp4")):
            print(f"MP4 candidate folder is ready: {stg1_mp4_candidate_ex}")
            mp4_flag = True
        else:
            print(f">>> Warning: {stg1_mp4_candidate_ex} does not exist or has no mp4 files.")
            continue

        # Delete _ALinput suffix if exists
        if long_media.endswith("_ALinput"):
            long_media = long_media.split("_ALinput")[0]

        current_media_path = stg1_mp4_candidate_ex.joinpath(f"{long_media}.mp4")
        print(f"Processing {current_csv_file.name}: long_media = {current_media_path.name}")


        # # Verify if long wav exists
        # if not current_media_path.exists():
        #     print(f">>> Error: {current_media_path} does not exist. Skipping.")
        #     continue
        # Set server root to the mp4 candidate folder
        server_root = remote_server_root.resolve()

        print(f"Server root: {server_root}")

        # Source HTML path (where this script is located)
        source_html_path = source_path / html_marker
        
        # Ensure source HTML exists
        if not source_html_path.exists():
            raise FileNotFoundError(f"{source_html_path} not found. Place {html_marker} in the Active Learning Loop folder.")

        # Copy HTML file to mp4 folder if it's not already there
        html_path = server_root / html_marker
        
        print(f"Copying {source_html_path} -> {html_path}")
        # Copy and replace existing file
        shutil.copy(source_html_path, html_path)

        # Copy images_june4.json if exists
        if json_file_path.exists():
            json_dest_path = server_root / json_file_path.name
            print(f"Copying {json_file_path} -> {json_dest_path}")
            shutil.copy(json_file_path, json_dest_path)

        # Wait until the HTML file is copied
        while not html_path.exists():
            pass

        # If json file was copied, wait until it exists
        if json_file_path.exists():
            json_dest_path = server_root / json_file_path.name
            while not json_dest_path.exists():
                pass

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

            # Add a small delay to ensure port is fully released
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")
            stop_event.set()
            # break

    print("[INFO] All files processed. Exiting.")