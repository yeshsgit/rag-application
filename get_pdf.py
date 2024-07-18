import os
import requests
import time
import threading


class get_pdf():
    def __init__(self, filepath, url):
        self.filepath = filepath
        self.url = url
        self.animate = False

    def download_pdf(self):
        """Downloads pdf to url in config file"""
        if not os.path.exists(self.filepath):
            self.animate = True
            animation_thread = threading.Thread(target=self.animate_download)
            animation_thread.start()

            filename = self.filepath
            # Sends get request to url
            response = requests.get(self.url, timeout=10)
            # Checks status code
            if response.status_code == 200:
                # Opens file and saves it
                with open(filename, 'wb') as file:
                    file.write(response.content)
                self.animate = False
                print(f"\n[INFO] Successfully downloaded pdf at url {self.url} as {filename}")
            else:
                print(f"[INFO] Failed to download file. Status code = {response.status_code}")
        else:
            print(f"[INFO] File {self.filepath} exists")

    def animate_download(self):
        while self.animate:
            for i in range(4):
                if not self.animate:
                    break
                dots = '.' * i
                print(f"\r[INFO] File {self.filepath} doesn't exist downloading{dots}", end='', flush=True)
                time.sleep(0.5)
