# coding=utf-8
# Copyright 2021 jemix.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python
""" Task: Named Entity Recognition """

import datasets
import logging

logger = logging.getLogger(__name__)


_CITATION = """\
@inproceedings{id_citation,
    title = "",
    author = "",
    booktitle = "",
    year = "",
    url = "",
    pages = "00--00",
}
"""

_DESCRIPTION = """\
description about data 

For more details see https://github.com/jemiaymen/TC/ner/
"""


_URL = "data/"
_TRAINING_FILE = "ner_train.txt"
_DEV_FILE = "ner_valid.txt"
_TEST_FILE = "ner_test.txt"


class NerConfig(datasets.BuilderConfig):
    """BuilderConfig for Ner"""

    def __init__(self, **kwargs):
        """BuilderConfig Ner

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NerConfig, self).__init__(**kwargs)


class NerPos(datasets.GeneratorBasedBuilder):
    """Ner dataset."""

    BUILDER_CONFIGS = [
        NerConfig(name="ner", version=datasets.Version(
            "1.0.0"), description="Ner dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "token": datasets.Value("string"),
                    "tag": datasets.features.ClassLabel(
                        names=[
                            'PERSON',  # People, including fictional.
                            # Nationalities or religious or political groups.
                            'NORP',
                            # Buildings, airports, highways, bridges, etc.
                            'FAC',
                            # Companies, agencies, institutions, etc.
                            'ORG',
                            'GPE',  # Countries, cities, states.
                            # Non-GPE locations, mountain ranges, bodies of water.
                            'LOC',
                            # Objects, vehicles, foods, etc. (Not services.)
                            'PRODUCT',
                            # Named hurricanes, battles, wars, sports events, etc.
                            'EVENT',
                            'WORK_OF_ART',  # Titles of books, songs, etc.
                            'LAW',  # Named documents made into laws.
                            'LANGUAGE',  # Any named language.
                            # Absolute or relative dates or periods.
                            'DATE',
                            'TIME',  # Times smaller than a day.
                            'PERCENT',  # Percentage, including ”%“.
                            'MONEY',  # Monetary values, including unit.
                            # Measurements, as of weight or distance.
                            'QUANTITY',
                            'ORDINAL',  # “first”, “second”, etc.
                            # Numerals that do not fall under another type.
                            'CARDINAL'
                        ]
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/jemiaymen/TC/ner",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                                    "filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                                    "filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={
                                    "filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            for line in f:
                splits = line.split("\t")
                if len(splits) < 2:
                    continue

                guid += 1
                yield guid, {
                    "id": str(guid),
                    "token": splits[0],
                    "tag": splits[1].rstrip(),
                }
        logger.info("Generating examples finish !")
