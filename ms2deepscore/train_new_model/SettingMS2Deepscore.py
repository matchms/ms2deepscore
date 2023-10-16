class SettingsMS2Deepscore:
    def __init__(self, settings=None):
        self.epochs = 150,
        self.base_dims = (500, 500),
        self.embedding_dim = 200,
        self.average_pairs_per_bin = 20,
        self.max_pairs_per_bin = 100
        self.additional_metadata = ()

        if settings:
            for key, value in settings.items():
                if key in self.__dict__:
                    self.__dict__[key] = value
                else:
                    raise ValueError(f"Unknown setting: {key}")
