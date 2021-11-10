import einops as E
import numpy as np
import os

import datasets


_CITATION = ""
_DESCRIPTION = """Object-Centric Representation Benchmark """
_HOMEPAGE = ""
_LICENSE = ""

_URLs = {
    'vor': "datasets/vor.hdf5",
    'vmds': "datasets/vmds/",
    'tex_vmds': "datasets/tex_vmds/",
    'spmot': "datasets/spmot",
}


class OCRB(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="vor", version=VERSION, description="Video Object Room", data_dir="data/datasets/vor/"),
        datasets.BuilderConfig(name="vmds", version=VERSION, description="Video Multi-dSprites", data_dir="data/datasets/vmds"),
        datasets.BuilderConfig(name="tex_vmds", version=VERSION, description="Textured Video multi-dsprite", data_dir="data/datasets/tex_vmds"),
        datasets.BuilderConfig(name="spmot", version=VERSION, description="Sprite Multi-Object Tracking", data_dir="data/datasets/spmot"),
    ]

    DEFAULT_CONFIG_NAME = "vor"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "video": datasets.Array4D(shape=(10,64,64,3), dtype="uint8"),
            }),
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"{self.config.name}_train.npy"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"{self.config.name}_test.npy"),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"{self.config.name}_val.npy"),
                    "split": "val",
                },
            ),
        ]

    def _generate_examples(
        self, filepath, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.
        imgs = E.rearrange(np.load(filepath), 'b t c h w -> b t h w c')

        for idx in range(len(imgs)):
            yield idx, {"video": imgs[idx]}
