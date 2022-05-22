"""data_loader.py - provides functionality to load data"""

import csv
import logging
import os

from lib.report import ComplexityReport


LOG = logging.getLogger(__name__)


def load_two_column_csv_file(path, encoding="utf-8", delimiter=","):
    """Generates file content while doing format verification.

    Args:
        path (str): Full path to a csv-file.
        encoding (Optional[str]): Encoding of the csv-file.

    Yields:
        tuple: Content of the next row in the file.

    """
    with open(path, "r", encoding=encoding, newline='') as file_in:
        csv.field_size_limit(2*10**5)
        reader = csv.reader(file_in, delimiter=delimiter)

        for n, line in enumerate(reader, 1):

            col_n = len(line)

            if not col_n:
                continue
            if col_n < 2:
                raise ValueError(
                    f"Line {n} is missing a column. Expected 2, got {col_n}."
                )
            if col_n > 2:
                raise ValueError(
                    f"Line {n} has too many columns. Expected 2, got {col_n}."
                )

            yield line


def iterate_training_datasets(directory, mode="train"):
    """Iterates over all training data in the passed directory.

    Each nested directory must follow the following structure:

    DATASET_NAME/
        |
        |_ README.md
        |
        |_ eval/
        |      |
        |      |_ DATASET_NAME__TEST.csv
        |      |
        |      |_ DATASET_NAME__DEV.csv (optional)
        |
        |_ training/
               |
               |_ DATASET_NAME__FULL.csv

    Args:
        str: Path to a directory that includes
            several data sets as nested directories.

    Yields:
        str: Path to the next csv-file with training data.

    """
    for nested_dir in os.listdir(directory):
        if nested_dir.startswith(".") or nested_dir.startswith("__"):
            continue

        if mode == "train":
            full_path = os.path.join(
                directory, nested_dir, "training", f"{nested_dir}__FULL.csv"
            )
        else:
            full_path = os.path.join(
                directory, nested_dir, "eval", f"{nested_dir}__TEST.csv"
            )
        yield nested_dir, full_path


def write_complexity_report(source, output, delimiter="\t", **kwargs):
    """Create a complexity report for a collection of data sets.

    Each nested directory must follow the following structure:

    DATASET_NAME/
        |
        |_ README.md
        |
        |_ eval/
        |      |
        |      |_ DATASET_NAME__TEST.csv
        |      |
        |      |_ DATASET_NAME__DEV.csv (optional)
        |
        |_ training/
               |
               |_ DATASET_NAME__FULL.csv

    Args:
        source (str): Path to a directory that includes
            several data sets as nested directories.
        output (str): Directory that a csv-file, called
            'complexity_report.tsv' will be saved to.
            If a file of the same name already exists, it
            will be overriden. Defaults to current directory.
        delimiter (str): Character that separates entries in
            the nested two column csv-files.

    """
    with open(
        os.path.join(output, "complexity_report.tsv") ,
        'w', encoding="utf-8", newline=''
    ) as file_out:

        writer = csv.writer(file_out, delimiter=delimiter)
        writer.writerow(["Dataset", "Class", *ComplexityReport.fields])

        for name, file in iterate_training_datasets(source):
            LOG.info(f"----> Dataset: {name} <----")

            texts, labels = zip(*load_two_column_csv_file(file, **kwargs))

            report = ComplexityReport.from_raw_data(texts, labels, name=name)

            for class_name, content in report.info.iterrows():
                writer.writerow([
                    name,
                    class_name,
                    *content.tolist()
                ])
            file_out.flush()
