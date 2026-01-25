import concurrent.futures
import threading
from PIL import Image
from queue import Queue, Empty
import time

class AsyncImageWriter:
    def __init__(self, num_processes=0, num_threads=4):
        self.num_processes = num_processes
        self.num_threads = num_threads

        if self.num_processes > 0:
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes)
        elif self.num_threads > 0:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads)
        else:
            self.executor = None

        self.futures = Queue()
        self.running = True

    def save_image(self, image, fpath):
        if self.executor is not None:
            # Check for completed futures to avoid memory leak if queue gets too large?
            # Actually, standard pattern is fire and forget, but we need to verify completion eventually.
            # We put future in a queue to wait for them later.
            future = self.executor.submit(self._save_image_task, image, fpath)
            self.futures.put(future)
        else:
            self._save_image_task(image, fpath)

    @staticmethod
    def _save_image_task(image, fpath):
        # image might be a PIL Image or numpy array. If numpy, convert?
        # Assuming PIL Image for now as per my previous code usage.
        # But if passed across processes, it must be picklable. PIL Images are picklable.
        try:
            image.save(fpath)
        except Exception as e:
            print(f"Error saving image to {fpath}: {e}")
            raise e

    def wait_until_done(self):
        while not self.futures.empty():
            try:
                future = self.futures.get(timeout=0.1)
                future.result() # This will raise exception if task failed
            except Empty:
                continue

    def stop(self):
        self.wait_until_done()
        if self.executor:
            self.executor.shutdown(wait=True)
        self.running = False
