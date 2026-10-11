"""Partitioned datasets, composite record ids, and S3 object-vs-prefix reads."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from foresightrlcore.common import io as common_io
from foresightrlcore.evaluate.datasets.tabular_loader import TabularDatasetLoader


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


class TestPartitionedDataset:
    """Anything Spark or pandas wrote is a directory, not a file."""

    def test_part_files_are_read_as_one_table(self, tmp_path):
        _write_jsonl(tmp_path / "part-00000.jsonl", [{"question": "a", "answer": "1"}])
        _write_jsonl(tmp_path / "part-00001.jsonl", [{"question": "b", "answer": "2"}])

        records = TabularDatasetLoader("jsonl").records({"path": str(tmp_path)})

        assert [r.inputs["question"] for r in records] == ["a", "b"]

    def test_parts_are_read_in_sorted_order_so_row_indices_are_stable(self, tmp_path):
        _write_jsonl(tmp_path / "part-00001.jsonl", [{"question": "second"}])
        _write_jsonl(tmp_path / "part-00000.jsonl", [{"question": "first"}])

        records = TabularDatasetLoader("jsonl").records({"path": str(tmp_path)})

        assert [r.inputs["question"] for r in records] == ["first", "second"]

    def test_spark_marker_files_are_ignored(self, tmp_path):
        _write_jsonl(tmp_path / "part-00000.jsonl", [{"question": "a"}])
        (tmp_path / "_SUCCESS").write_text("", encoding="utf-8")
        (tmp_path / ".hidden.crc").write_text("junk", encoding="utf-8")

        records = TabularDatasetLoader("jsonl").records({"path": str(tmp_path)})

        assert len(records) == 1

    def test_an_empty_directory_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no data files under"):
            TabularDatasetLoader("jsonl").records({"path": str(tmp_path)})

    def test_a_single_file_still_works(self, tmp_path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [{"question": "a", "answer": "1"}])

        assert len(TabularDatasetLoader("jsonl").records({"path": str(path)})) == 1


class TestCompositeRecordIds:
    def test_id_columns_are_joined(self, tmp_path):
        path = tmp_path / "d.jsonl"
        _write_jsonl(path, [{"question": "q", "co": "7", "txn": "42"}])

        records = TabularDatasetLoader("jsonl").records(
            {"path": str(path), "id_columns": ["co", "txn"]}
        )

        assert records[0].id == "7|42"

    def test_id_columns_win_over_a_single_id_column(self, tmp_path):
        path = tmp_path / "d.jsonl"
        _write_jsonl(path, [{"question": "q", "id": "ignored", "co": "7", "txn": "42"}])

        records = TabularDatasetLoader("jsonl").records(
            {"path": str(path), "id_columns": ["co", "txn"], "id_column": "id"}
        )

        assert records[0].id == "7|42"

    def test_a_misspelled_id_column_fails_closed(self, tmp_path):
        # Silently keying on "" would collapse every row to one id and put
        # resume back on row order.
        path = tmp_path / "d.jsonl"
        _write_jsonl(path, [{"question": "q", "co": "7", "txn": "42"}])

        with pytest.raises(ValueError, match=r"\['tx'\]") as err:
            TabularDatasetLoader("jsonl").records(
                {"path": str(path), "id_columns": ["co", "tx"]}
            )
        assert "id_columns" in str(err.value)

    def test_without_id_columns_the_row_index_is_still_used(self, tmp_path):
        path = tmp_path / "d.jsonl"
        _write_jsonl(path, [{"question": "a"}, {"question": "b"}])

        records = TabularDatasetLoader("jsonl").records({"path": str(path)})

        assert [r.id for r in records] == ["0", "1"]


class TestResolveDatasetPath:
    """An S3 prefix has to be synced as a directory, not fetched as an object."""

    def test_a_trailing_slash_is_read_as_a_prefix_without_asking_s3(self, tmp_path):
        # An explicit slash is taken at face value, so no HeadObject is issued.
        with (
            mock.patch.object(common_io, "s3_download_prefix") as sync,
            mock.patch("foresightrlcore.common.io.s3_object_exists") as probe,
        ):
            common_io.resolve_dataset_path(
                "s3://b/data/eval_set/", download_dir=tmp_path
            )
        assert sync.call_args[0][0] == "s3://b/data/eval_set/"
        assert not probe.called

    def test_a_key_that_is_not_an_object_is_read_as_a_prefix(self, tmp_path):
        with (
            mock.patch.object(common_io, "s3_download_prefix") as sync,
            mock.patch(
                "foresightrlcore.common.io.s3_object_exists", return_value=False
            ),
        ):
            common_io.resolve_dataset_path(
                "s3://b/data/eval_set", download_dir=tmp_path
            )
        assert sync.called

    def test_an_extensionless_object_is_downloaded_as_one_file(self, tmp_path):
        # The real case: the QBAA eval set is a parquet object whose key has no
        # extension, so the name cannot be what decides.
        with (
            mock.patch.object(common_io, "s3_download") as download,
            mock.patch("foresightrlcore.common.io.s3_object_exists", return_value=True),
        ):
            common_io.resolve_dataset_path(
                "s3://b/data/eval_set", download_dir=tmp_path
            )
        assert download.call_args[0][0] == "s3://b/data/eval_set"

    def test_a_named_object_is_still_downloaded_as_one_file(self, tmp_path):
        with (
            mock.patch.object(common_io, "s3_download") as download,
            mock.patch("foresightrlcore.common.io.s3_object_exists", return_value=True),
        ):
            common_io.resolve_dataset_path(
                "s3://b/data/eval.csv", download_dir=tmp_path
            )
        assert download.call_args[0][0] == "s3://b/data/eval.csv"

    def test_the_probe_reads_bucket_and_key_apart(self):
        with mock.patch("foresightrlcore.common.io.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0)
            assert common_io.s3_object_exists("s3://bucket/a/b/c.parquet")
        assert run.call_args[0][0][-4:] == [
            "--bucket",
            "bucket",
            "--key",
            "a/b/c.parquet",
        ]

    def test_a_bucket_with_no_key_is_never_an_object(self):
        assert not common_io.s3_object_exists("s3://bucket")

    def test_a_404_is_absence(self):
        with mock.patch("foresightrlcore.common.io.subprocess.run") as run:
            run.return_value = mock.Mock(
                returncode=255,
                stderr=(
                    "An error occurred (404) when calling the HeadObject "
                    "operation: Not Found"
                ),
                stdout="",
            )
            assert not common_io.s3_object_exists("s3://bucket/missing.csv")

    @pytest.mark.parametrize(
        "detail",
        [
            "An error occurred (ExpiredToken) when calling the HeadObject "
            "operation: The provided token has expired.",
            "An error occurred (403) when calling the HeadObject operation: "
            "Forbidden",
        ],
    )
    def test_any_other_failure_is_raised_rather_than_read_as_absence(self, detail):
        """Reporting these as absence sends the caller down the prefix branch.

        That is how an expired token turns into "no objects found under
        .../irs_bench.csv" for a file that is plainly there.
        """
        with mock.patch("foresightrlcore.common.io.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=255, stderr=detail, stdout="")
            with pytest.raises(RuntimeError, match="whether"):
                common_io.s3_object_exists("s3://bucket/there.csv")

    def test_a_bare_400_names_credentials_as_the_likely_cause(self):
        # What an expired token actually looks like: HeadObject has no body,
        # so the CLI cannot say ExpiredToken and reports only the status.
        detail = (
            "An error occurred (400) when calling the HeadObject operation: "
            "Bad Request"
        )
        with mock.patch("foresightrlcore.common.io.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=254, stderr=detail, stdout="")
            with pytest.raises(RuntimeError, match="credentials"):
                common_io.s3_object_exists("s3://bucket/there.csv")

    def test_prefixes_sharing_a_last_segment_get_separate_directories(self, tmp_path):
        # .../train/data/ and .../eval/data/ must not be mirrored into one
        # folder, or one benchmark reads the other's rows.
        with (
            mock.patch.object(
                common_io, "s3_download_prefix", side_effect=lambda uri, local: local
            ),
            mock.patch.object(common_io, "s3_object_exists", return_value=False),
        ):
            train = common_io.resolve_dataset_path(
                "s3://b/train/data/", download_dir=tmp_path
            )
            evaluation = common_io.resolve_dataset_path(
                "s3://b/eval/data/", download_dir=tmp_path
            )
            again = common_io.resolve_dataset_path(
                "s3://b/eval/data", download_dir=tmp_path
            )
        assert train != evaluation
        assert evaluation == again

    def test_a_prefix_is_mirrored_so_stale_part_files_are_removed(self, tmp_path):
        with mock.patch("foresightrlcore.common.io.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            (tmp_path / "part-0.parquet").write_text("x")
            common_io.s3_download_prefix("s3://b/data/", tmp_path)
        assert run.call_args[0][0][:4] == ["aws", "s3", "sync", "--delete"]

    def test_a_missing_local_path_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            common_io.resolve_dataset_path(
                str(tmp_path / "nope"), download_dir=tmp_path
            )
