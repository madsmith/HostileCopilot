import numpy as np

from .audio_data import AudioData

class PlaybackBuffer:
    def __init__(self, rate: int):
        self.rate = rate
        self.chunks: list[AudioData] = []

    def append(self, frames: AudioData):
        assert isinstance(frames, AudioData), "frames must be of type AudioData"
        self.chunks.append(frames)

    def prune_older_than(self, someTime: float):
        self.chunks = [chunk for chunk in self.chunks if chunk.end_time() > someTime]

    def __len__(self):
        return len(self.chunks)
    
    def extract_frames(self, start: float, frame_count: int) -> np.ndarray:
        """
        Extracts `frame_count` samples of audio starting from `start` time.
        Pads with zeros where data is missing.
        """
        output = np.zeros(frame_count, dtype=np.int16)
        end = start + (frame_count / self.rate)

        for chunk in self.chunks:
            chunk_start = chunk.timestamp
            chunk_end = chunk.end_time()

            if chunk_end < start:
                continue
            if chunk_start > end:
                continue

            chunk_data = chunk.as_array()

            # Calculate overlap
            overlap_start_time = max(start, chunk_start)
            overlap_end_time = min(end, chunk_end)

            out_start_idx = int((overlap_start_time - start) * self.rate)
            out_end_idx = int((overlap_end_time - start) * self.rate)

            in_start_idx = int((overlap_start_time - chunk_start) * self.rate)
            in_end_idx = int((overlap_end_time - chunk_start) * self.rate)

            copy_len = min(out_end_idx - out_start_idx, in_end_idx - in_start_idx)

            output[out_start_idx:out_start_idx + copy_len] = chunk_data[in_start_idx:in_start_idx + copy_len]

        return output

    def extract_window(self, start: float, end: float) -> np.ndarray:
        """
        Extracts a continuous buffer of audio corresponding to the playback window [start, end].
        If there are gaps in the chunks, fills with zeros.
        """
        total_samples = int((end - start) * self.rate)
        output = np.zeros(total_samples, dtype=np.int16)

        for chunk in self.chunks:
            chunk_start = chunk.timestamp
            chunk_end = chunk.end_time()

            if chunk_end < start:
                continue
            if chunk_start > end:
                continue

            chunk_data = chunk.as_array()

            # Calculate overlap
            overlap_start_time = max(start, chunk_start)
            overlap_end_time = min(end, chunk_end)

            out_start_idx = int((overlap_start_time - start) * self.rate)
            out_end_idx = int((overlap_end_time - start) * self.rate)

            in_start_idx = int((overlap_start_time - chunk_start) * self.rate)
            in_end_idx = int((overlap_end_time - chunk_start) * self.rate)

            # Calculate actual number of samples to copy (ensure bounds agree)
            copy_len = min(out_end_idx - out_start_idx, in_end_idx - in_start_idx)

            output[out_start_idx:out_start_idx + copy_len] = chunk_data[in_start_idx:in_start_idx + copy_len]

        return output

    def dump_windows(self, prefix=""):
        """
        Print the start and end times of each series of contiguous chunks of audio data
        """
        output = ""

        if len(self.chunks) == 0:
            return f"{prefix}No audio data"

        start_time = self.chunks[0].timestamp
        end_time = self.chunks[0].end_time()
        chunk_count = 1

        for chunk in self.chunks[1:]:
            # Check for a gap 
            gap = chunk.timestamp - end_time
            if gap > 1/self.rate:
                output += f"{prefix}Playback Window: {start_time:.3f} - {end_time:.3f} [{chunk_count} chunks]\n"
                start_time = chunk.timestamp
                chunk_count = 0
            chunk_count += 1
            end_time = max(end_time, chunk.end_time())

        output += f"{prefix}Playback Window: {start_time:.3f} - {end_time:.3f} [{chunk_count} chunks]"
        
        return output