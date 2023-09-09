class BaseLayer:
    def __init__(self):
        # Used to distinguish between trainable from non-trainable layers.
        self.trainable = False
        self.testing_phase = False
