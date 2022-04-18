from bfio import BioReader
import numpy
from typing import Tuple
from napari.utils.notifications import show_info
from napari.layers._multiscale_data import MultiScaleData


class BfioLayer:
    def __init__(self, br: BioReader, scale: int):

        self.br = br
        self.scale = 2**scale

    def __getitem__(self, keys):
        return self.br[tuple(reversed(keys))]

    @property
    def dtype(self):
        return self.br.dtype

    @property
    def shape(self):
        shape = tuple(
            reversed(
                tuple(round(d / self.scale) for d in self.br.shape[:2])
                + self.br.shape[2:]
            )
        )
        return shape

    @property
    def ndim(self):
        return len(self.br.shape)


class BfioReader(MultiScaleData):
    """Special class to read data into Napari."""

    def __init__(self, file: str):

        # Let BioReader try to guess the backend based on file extension
        try:
            self.br = BioReader(file)

        # If the backend is wrong, fall back to BioFormats if installed
        except ValueError:
            self.br = BioReader(file, backend="bioformats")

    def __call__(self, file):

        metadata = {
            "name": self.br.metadata.images[0].name,
            "rgb": False,
            "multiscale": True,
            "contrast_limits": (
                numpy.iinfo(self.br.dtype).min,
                numpy.iinfo(self.br.dtype).max,
            ),
        }

        return [
            (self, metadata, "image"),
        ]

    def __len__(self):

        longest = max(self.br.X, self.br.Y)
        return (longest - 1) // 1024 + 1

    def __getitem__(self, key: int):

        if key < len(self):
            return BfioLayer(self.br, key)

        raise IndexError

    @property
    def dtype(self) -> numpy.dtype:
        """Return dtype of the first scale.."""
        return self.br.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of multiscale is just the biggest shape."""
        return self.br.shape

    @property
    def shapes(self) -> Tuple[Tuple[int, ...], ...]:
        """Tuple shapes for all scales."""
        return tuple(BfioLayer(self.br, i).shape for i in range(len(self)))


def get_reader(path: str):

    # try:
    reader = BfioReader(path)
    BioReader.logger.info("Reading with the BioReader.")
    show_info("Using BfioReader")
    # except Exception as e:
    #     BioReader.logger.info(e)
    #     reader = None

    return reader
