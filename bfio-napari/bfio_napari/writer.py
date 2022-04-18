@napari_hook_implementation(specname="napari_write_image")
def get_writer(path: str, data: numpy.ndarray, meta: dict):

    if isinstance(data, NapariReader):
        bw = BioWriter(path, metadata=data.br.metadata)

        bw[:] = data.br[:]
    else:
        if meta["rgb"]:
            BioWriter.logger.info("The BioWriter cannot write color images.")
            return None

        bw = BioWriter(path)
        bw.shape = data.shape
        bw.dtype = data.dtype

        data = numpy.transpose(data, tuple(reversed(range(data.ndim))))

        while data.ndim < 5:
            data = data[..., numpy.newaxis]

        bw[:] = data

    return path
