from quickstats import DescriptiveEnum

class LoggerSaveMode(DescriptiveEnum):
    BATCH = ("batch", "Save once per batch")
    EPOCH = ("epoch", "Save once per epoch")
    TRAIN = ("train", "Save once per training")