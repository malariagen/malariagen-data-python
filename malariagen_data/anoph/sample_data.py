from .base import AnophelesBase


class AnophelesGenomeSequenceData(AnophelesBase):
    def __init__(self, **kwargs):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

        # Initialize cache attributes.
        # TODO
