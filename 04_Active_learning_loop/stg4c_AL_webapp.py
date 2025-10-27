
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

if __name__ == "__main__":

    html_AL = "AL2_LP_webapp.html"

    LP_METHOD_NAME = "LP1"
    DATASET_NAME = "TestAO-Irma"

    base_path_ex = Path.home().joinpath('Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline',DATASET_NAME)
    # base_path_ex = Path.home().joinpath('Library','CloudStorage','Dropbox','DATASETS_AUDIO','Unsupervised_Pipeline',DATASET_NAME)
    stg1_mp4_candidate_ex = base_path_ex.joinpath('input_mp4s')
    stg1_wav_candidate_ex = base_path_ex.joinpath('input_wavs')
    stg4_al_folder_ex = base_path_ex.joinpath('STG_4',f'STG4_{LP_METHOD_NAME}')
    remote_server_root = base_path_ex
    source_path = Path.home().joinpath('Dropbox', 'Source_2025', '04_Active_learning_loop')

    # For loop read csv files in stg4_al_folder_ex
    # Each csv file has columns: cluster_id, start_time, end_time

    # List all csv files in the directory
    csv_files = list(stg4_al_folder_ex.glob("*.csv"))

    for i, current_csv_file in enumerate(csv_files):
        current_csv_file = csv_files[i]  # For testing, just take the first file
        long_media = current_csv_file.stem

        mp4_flag = False

        # Verify if mp4 candidate folder exists and has *.mp4 files
        if stg1_mp4_candidate_ex.exists() and any(stg1_mp4_candidate_ex.glob("*.mp4")):
            print(f"MP4 candidate folder is ready: {stg1_mp4_candidate_ex}")
            mp4_flag = True
        else:
            print(f"Warning: {stg1_mp4_candidate_ex} does not exist or has no mp4 files.")

        # Delete _ALinput suffix if exists
        if long_media.endswith("_ALinput"):
            long_media = long_media.split("_ALinput")[0]

        current_media_path = ''
        if mp4_flag:
            current_media_path = stg1_mp4_candidate_ex.joinpath(f"{long_media}.mp4")
        else:
            current_media_path = stg1_wav_candidate_ex.joinpath(f"{long_media}.wav")

        print(f"Processing {current_csv_file.name}: long_media = {current_media_path.name}")


        # # Verify if long wav exists
        # if not current_media_path.exists():
        #     print(f">>> Error: {current_media_path} does not exist. Skipping.")
        #     continue
        # Set server root to the mp4 candidate folder
        server_root = remote_server_root.resolve()

        print(f"Server root: {server_root}")

        # Source HTML path (where this script is located)
        source_html_path = source_path / html_AL
        
        # Ensure source HTML exists
        if not source_html_path.exists():
            raise FileNotFoundError(f"{source_html_path} not found. Place {html_AL} in the Active Learning Loop folder.")

        # Copy HTML file to mp4 folder if it's not already there
        html_path = server_root / html_AL
        print(f"Copying {source_html_path} -> {html_path}")
        shutil.copy2(source_html_path, html_path)

        # Wait until the HTML file is copied
        while not html_path.exists():
            pass

        print(f"Video file: {current_media_path}")

        port = get_free_port(PORT)

        # Create stop event for this server
        stop_event = threading.Event()

        # Start server
        thread = threading.Thread(target=start_server, args=(port, server_root, stop_event), daemon=True)
        thread.start()

        # Build URL
        media_url = urllib.parse.quote(current_media_path.name)
        url = f"http://localhost:{port}/{html_path.name}?media={media_url}&method={LP_METHOD_NAME}"

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
            # break

    print("[INFO] All files processed. Exiting.")