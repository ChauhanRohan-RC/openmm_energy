import os
import threading


# =======================================================
# IndexStreamBuffer
# =======================================================
class IndexStreamBuffer:
    """
    A data structure to write the values in order of indices

    Data can come in any order asynchronously, and the job of this class is to look of
    consecutive chunk of indices, and if found, write them to the output file and release memory

    It is thread-safe, but not process safe (for that, replace threading.Lock with mp.Lock, will be slow)
    .close() must be called to flush remaining indices safely.

    -> It stores mappings of index -> value in a dict
    -> checks if a contiguous chunk of consecutive indices exists
    -> dumps the block to output file, and free up memory

    See .insert(index: int, value: object)
        .close()
    """

    TAG = "IndexStreamBuffer"

    def __init__(self,
                 output_file_path: str,
                 chunk_size: int,
                 keep_file_open: bool = False,
                 string_converter_callback=None,
                 pre_chunk_write_callback=None,
                 post_chunk_write_callback=None):

        """
        :parameter output_file_path: output file path
        :parameter chunk_size: a contiguous block of indices to write to output file
        :parameter string_converter_callback: a function that converts actual data to strings for writing
        :parameter pre_chunk_write_callback: a function that is called before a chunk is written to output file.
                   It takes chunk_index as input, and may return a string  to be written before the chunk
        :parameter post_chunk_write_callback: a function that is called after a chunk is written to output file.
                   It takes chunk_index and chunk_size (number of entries in the chunk)_as input, and returns nothing
        :parameter keep_file_open:  if true, file descriptor is kept open for the entire time until close() is called
        """

        if output_file_path is None or len(output_file_path.strip()) == 0:
            raise ValueError(f"{self.__class__.TAG}: output_file_path cannot be empty")
        if chunk_size <= 0:
            raise ValueError(f"{self.__class__.TAG}: Chunk size must be greater than zero. Given: {chunk_size}")

        self.out_file_path = output_file_path
        self.chunk_size = chunk_size
        self.string_converter_callback = string_converter_callback if string_converter_callback is not None else str
        self.pre_chunk_write_callback = pre_chunk_write_callback
        self.post_chunk_write_callback = post_chunk_write_callback
        self.keep_file_open = keep_file_open

        # internal state
        self._written_chunk_count: int = 0
        self._out_fd = None  # only when keep_file_open is true
        self._is_closed: bool = False
        self._lock = threading.Lock()

        self._data: dict = {}
        self._next_index: int = 0
        # self._next_range_set: set = set()
        self._set_next_index(0)  # init next index

    def _consider_delete_file(self):
        if self._written_chunk_count == 0 and os.path.exists(self.out_file_path):
            try:
                os.remove(self.out_file_path)
            except Exception as e:
                raise RuntimeError("{self.__class__.TAG}: Could not remove file: " + self.out_file_path) from e

    def _set_next_index(self, next_index: int):
        self._next_index: int = next_index
        self._next_range_set: set = set(range(self._next_index, self._next_index + self.chunk_size))

    def _check_closed(self):
        if self._is_closed:
            raise RuntimeError("{self.__class__.TAG}: Index buffer already is closed")

    def is_closed(self) -> bool:
        return self._is_closed

    def written_chunk_count(self) -> int:
        return self._written_chunk_count

    def size(self) -> int:
        return len(self._data)

    def insert(self, index: int, value: object):
        self._check_closed()
        if index < 0:
            raise ValueError(f"{self.__class__.TAG}: Index must be greater than or equal to 0, given: {index}")

        with self._lock:
            # if DEBUG and index in self._data:
            #     log_warn(f"{self.__class__.TAG}: INDEX {index} already present !!!")

            self._data[index] = value
            self._consider_flush_unsafe()

    def __write_indices_to_fd(self, fd, indices_sorted, pre_string = None):
        # Pre string
        if pre_string is not None and isinstance(pre_string, str) and len(pre_string) > 0:
            fd.write(pre_string)

        # actual data
        for i in indices_sorted:
            fd.write(self.string_converter_callback(self._data.pop(i)))

    def _write_chunk_indices(self, indices_sorted, indices_size: int, execute_pre_write_callback: bool = True):
        chunk_index = self._written_chunk_count

        pre_string = None
        if execute_pre_write_callback and self.pre_chunk_write_callback is not None:
            pre_string = self.pre_chunk_write_callback(chunk_index)

        if self.keep_file_open:
            if self._out_fd is None:
                self._out_fd = open(self.out_file_path, "w")
            self.__write_indices_to_fd(self._out_fd, indices_sorted, pre_string=pre_string)
        else:
            self._consider_delete_file()
            with open(self.out_file_path, "a") as out_fd:
                self.__write_indices_to_fd(out_fd, indices_sorted, pre_string=pre_string)

        self._written_chunk_count += 1
        self._set_next_index(self._next_index + indices_size)

        # Post-write callback
        if self.post_chunk_write_callback is not None:
            self.post_chunk_write_callback(chunk_index, indices_size)

    # NOT THREAD SAFE
    def _consider_flush_unsafe(self):
        self._check_closed()

        while len(self._data) >= self.chunk_size:
            # Check if the current range exists
            range_exists = self._next_range_set.issubset(self._data.keys())
            if not range_exists:
                return

            # write chunk to file
            indices = range(self._next_index, self._next_index + self.chunk_size)
            self._write_chunk_indices(indices, indices_size=self.chunk_size)

    def close(self):
        if self._is_closed:
            return

        with self._lock:
            if self._is_closed:
                return
            self._is_closed = True

            # force flush remaining indices in order
            if len(self._data) > 0:
                sorted_indices = sorted(self._data.keys())
                self._write_chunk_indices(sorted_indices, indices_size=len(sorted_indices))
            self._data.clear()

            # Close file descriptor
            if self._out_fd is not None:
                try:
                    self._out_fd.close()
                except Exception as e:
                    print(f"{self.__class__.TAG}: WARNING Could not close output file: {self.out_file_path}", e)


if __name__ == '__main__':
    # ------------------------------------------------
    # TEST IMPLEMENTATION
    # ----------------------------------------------------

    def convert_to_str(entry) -> str:
        return str(entry)

    def on_pre_chunk_write(chunk_index: int) -> str | None:
        print(f"PRE_CHUNK_WRITE: {chunk_index}")
        if chunk_index == 0:
            return "# Comment Line 1\n# COmment Line 2\n# Comment Line 3\n"
        return None

    def on_post_chunk_write(chunk_index: int, chunk_size: int):
        print(f"POST_CHUNK_WRITE: {chunk_index} | chunk size: {chunk_size}")


    idx_stream_buffer = IndexStreamBuffer(output_file_path="test_index_streamer.txt",
                                          chunk_size=9,
                                          keep_file_open=False,
                                          # string_converter_callback=convert_to_str,
                                          pre_chunk_write_callback=on_pre_chunk_write,
                                          post_chunk_write_callback=on_post_chunk_write)

    import random
    indices = list(range(0, 9300))
    random.shuffle(indices)     # in place shuffle
    for i in indices:
        idx_stream_buffer.insert(i, f"VALUE LINE {i}\n")

    idx_stream_buffer.close()
