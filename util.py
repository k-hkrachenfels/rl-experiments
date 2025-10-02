from enum import EnumMeta

class IndexableEnumMeta(EnumMeta):
    """
    a metaclass to extend enums so that they can be indexed
    """
    def __init__(cls, name, bases, namespace):
        # initialize
        super().__init__(name, bases, namespace)
        # save members in a list
        cls.members = list(cls.__members__.values())

    # allows indexing on the class
    def __getitem__(cls, index):
        return cls.members[index]

    # allows to get number of elements
    def __len__(cls):
        return len(cls.members)