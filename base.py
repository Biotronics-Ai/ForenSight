import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, AsyncGenerator, Iterable
from tqdm.auto import tqdm


# ===========================================================
# BASE COMPONENT (common logging for all classes)
# ===========================================================

class BaseComponent(ABC):
    """Looging begavior for all classes."""

    def _log_state(self, state: str, message: str = "", error: str = None):
        """Logs the message and state in a uniform format."""
        cls = self.__class__.__name__
        if error:
            print(f"[{cls}] ❌ {state}: {message} | ERROR: {error}")
        else:
            print(f"[{cls}] ✅ {state}: {message}")

    def _progress(self, iterable: Iterable, **tqdm_kwargs):
        """
        Lightweight progress wrapper. Uses tqdm when available; otherwise returns the
        iterable unchanged. Keeps a minimal surface so subclasses can instrument loops
        without hard dependency.
        """
        if tqdm is None:
            return iterable
        return tqdm(iterable, **tqdm_kwargs)



# ===========================================================
# BASE PARSER (template + logging)
# ===========================================================

class BaseParser(BaseComponent):
    """Base Abstract class for reading the DNA file formats."""

    async def read(self, filepath: str):
        """Template method: main reading workflow with logging and N stripping."""
        try:
            self._log_state("STARTED", f"Reading {filepath}")
            raw_data = await self._read_file(filepath)
            self._log_state("READ_SUCCESS", filepath)

            sequence = self._extract_sequence(raw_data)
            cleaned, n_positions = self._strip_unknowns(sequence)
            metadata = self._extract_metadata(raw_data)
            self._log_state("PARSE_SUCCESS", filepath)

            self._log_state("FINISHED", filepath)
            return {"sequence": cleaned, "n_positions": n_positions, "metadata": metadata}

        except Exception as e:
            self._log_state("FAILED", filepath, error=str(e))
            raise

    async def read_batches(self, filepath: str, batch_size: int) -> AsyncGenerator[object, None]:
        """
        Default batch reader preserving order. Subclasses can override for more efficient
        streaming, but this keeps ordering and logging consistent.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        try:
            self._log_state("STARTED", f"Reading {filepath} in batches of {batch_size}")
            raw_data = await self._read_file(filepath)
            self._log_state("READ_SUCCESS", filepath)

            sequence = self._extract_sequence(raw_data)
            cleaned, n_positions = self._strip_unknowns(sequence)
            metadata = self._extract_metadata(raw_data)
            self._log_state("PARSE_SUCCESS", filepath)

            for chunk in self._chunk_sequence(cleaned, batch_size):
                yield {"sequence": chunk, "n_positions": n_positions, "metadata": metadata}

            self._log_state("FINISHED", filepath)
        except Exception as e:
            self._log_state("FAILED", filepath, error=str(e))
            raise

    @abstractmethod
    async def _read_file(self, filepath: str) -> List[str]:
        pass

    @abstractmethod
    def _extract_sequence(self, raw_data: List[str]) -> str:
        pass

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        """Optional metadata extraction; defaults to empty metadata."""
        return {}

    def _to_numeric(self, sequence) -> np.ndarray:
        """Legacy helper; actual vectorization handled by SampleLoader."""
        if isinstance(sequence, np.ndarray):
            return sequence
        if isinstance(sequence, (list, tuple)):
            return np.array(sequence)
        return np.frombuffer(str(sequence).encode(), dtype=np.uint8)

    def _strip_unknowns(self, sequence):
        """
        Remove 'N'/'n' characters from sequence, recording their positions.
        For non-string sequences, returns input as-is with empty positions.
        """
        if isinstance(sequence, str):
            positions = [i for i, ch in enumerate(sequence) if ch in ('N', 'n')]
            cleaned = ''.join(ch for ch in sequence if ch not in ('N', 'n'))
            return cleaned, positions
        if isinstance(sequence, np.ndarray):
            # assume numeric arrays already; treat -1 as unknown
            positions = np.where(sequence == -1)[0].tolist()
            cleaned = sequence[sequence != -1]
            return cleaned, positions
        return sequence, []

    def _chunk_sequence(self, sequence, batch_size: int):
        """Yield ordered chunks from a string, list/tuple, or numpy array."""
        if isinstance(sequence, str):
            for i in range(0, len(sequence), batch_size):
                yield sequence[i:i + batch_size]
            return

        if isinstance(sequence, np.ndarray):
            for i in range(0, len(sequence), batch_size):
                yield sequence[i:i + batch_size]
            return

        if isinstance(sequence, (list, tuple)):
            for i in range(0, len(sequence), batch_size):
                yield sequence[i:i + batch_size]
            return

        # last resort: yield the whole object to avoid silent data loss
        yield sequence
    




# ===========================================================
# BASE Visualizer 
# ===========================================================

class BaseVisualizer(BaseComponent):
    """
    Minimal visualization base: shared, step-by-step logging helpers for all
    visualization subclasses (e.g., plot renderers, CSV writers).
    """

    def log_prepare(self, message: str = ""):
        self._log_state("PREPARE", message)

    def log_encode(self, message: str = ""):
        self._log_state("ENCODE", message)

    def log_render(self, message: str = ""):
        self._log_state("RENDER", message)

    def log_write(self, message: str = ""):
        self._log_state("WRITE", message)

    def log_finish(self, message: str = ""):
        self._log_state("FINISHED", message)

    def log_error(self, message: str = "", error: str = None):
        self._log_state("FAILED", message, error=error)

    @abstractmethod
    async def visualize(self, *args, **kwargs):
        """Render or emit visualization output."""
        raise NotImplementedError
