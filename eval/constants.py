import argparse


class Dataset(argparse.Namespace):
    def __init__(self, name, ids, pattern, shape, padding, padding_pos):
        self.name = name
        self.ids = ids
        self.pattern = pattern
        self.shape = shape
        self.padding = padding
        self.padding_pos = padding_pos
        self.samples = dict()
        self.paths = set()


lesav_ids = [
    12, 15, 33, 34, 37, 42, 49, 53, 90, 111, 119, 224,
    240, 275, 279, 304, 318,
    367, 370, 486, 529, 586
]
lesav_dataset = Dataset(
    pattern='{id_}.png',
    ids=[str(i) for i in lesav_ids],
    name='LES-AV',
    shape=(1444, 1620),
    padding=((6, 7), (0, 0), (0, 0)),
    padding_pos=1,
)

rite_test_dataset = Dataset(
    pattern='{id_}.png',
    ids=[f"{i:02d}" for i in range(1, 21)],
    name='RITE-test',
    shape=(584, 565),
    padding=((4, 4), (5, 6), (0, 0)),
    padding_pos=2,
)


hrf_ids = [
    '01_dr', '01_g', '01_h', '02_dr', '02_g', '02_h', '03_dr', '03_g', '03_h',
    '04_dr', '04_g', '04_h', '05_dr', '05_g', '05_h',
]
hrf_dataset = Dataset(
    pattern='{id_}.png',
    ids=hrf_ids,
    name='HRF',
    shape=(2336, 3504),
    padding=None,
    padding_pos=None,
)


dataset_factory = {
    'RITE-test': rite_test_dataset,
    'LESAV': lesav_dataset,
    'HRF-test': hrf_dataset,
}
