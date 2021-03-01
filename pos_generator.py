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
""" Task: Part-of-speech """

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

For more details see https://github.com/jemiaymen/TC/pos/
"""


_URL = "data/"
_TRAINING_FILE = "pos_train.txt"
_DEV_FILE = "pos_valid.txt"
_TEST_FILE = "pos_test.txt"


class PosConfig(datasets.BuilderConfig):
    """BuilderConfig for Pos"""

    def __init__(self, **kwargs):
        """BuilderConfig Pos

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PosConfig, self).__init__(**kwargs)


class NerPos(datasets.GeneratorBasedBuilder):
    """Ner dataset."""

    BUILDER_CONFIGS = [
        PosConfig(name="pos", version=datasets.Version(
            "1.0.0"), description="Pos dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Value("string"),
                    "pos_tags":
                        datasets.features.ClassLabel(
                            names=[
                                'NOUN',  # noun, singular or mass
                                'IN',  # Preposition or subordinating conjunction
                                'PUNC',  # punctuation
                                'JJ',  # adjective
                                'NNP',  # Proper noun, singular
                                'CC',  # Coordinating conjunction
                                'VBP',  # Verb, non-3rd person singular present
                                'VBD',  # Verb, past tense
                                'NNS',  # noun, plural
                                'RP',  # particle
                                'CD',  # Cardinal number
                                'WP',  # Wh-pronoun
                                'DT',  # determiner
                                'NOFUNC',  # withou function
                                'PRP',  # Personal pronoun
                                'RB',  # adverb
                                'VBN',  # verb, past participle
                                'UH',  # interjection
                                'WRB',  # Wh-adverb
                                'NNPS',  # Proper noun, plural
                                'VERB'  # verb, base form
                            ]

                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/jemiaymen/TC/pos/",
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
