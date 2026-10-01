import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import Mock, patch

import dataset_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def test_workflow_order_and_upload_options(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            calls = Mock()
            with patch.object(pipeline, "generate_dataset", calls.generate), patch.object(pipeline, "validate_dataset", calls.validate), patch.object(pipeline, "convert_dataset", calls.convert), patch.object(pipeline, "export_parquet", calls.export), patch.object(pipeline, "upload_parquet", calls.upload):
                pipeline.main(["owner/repo", "--output-dir", directory, "--public", "--execute"])
            self.assertEqual([c[0] for c in calls.mock_calls], ["generate", "validate", "convert", "export", "upload"])
            calls.validate.assert_called_once_with(root, execute=True)
            calls.upload.assert_called_once_with(root, "owner/repo", public=True)

    def test_upload_is_limited_to_parquet_and_card(self):
        import sys
        api = Mock()
        operation = lambda **kwargs: types.SimpleNamespace(**kwargs)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parquet").mkdir()
            for split in pipeline.SPLITS:
                (root / "parquet" / f"{split}.parquet").write_bytes(b"fixture")
            with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(HfApi=lambda: api, CommitOperationAdd=operation)}), contextlib.redirect_stdout(io.StringIO()):
                pipeline.upload_parquet(root, "owner/repo")
            api.create_repo.assert_called_once_with(repo_id="owner/repo", repo_type="dataset", private=True, exist_ok=True)
            kwargs = api.create_commit.call_args.kwargs
            self.assertEqual([op.path_in_repo for op in kwargs["operations"]], ["data/train.parquet", "data/validation.parquet", "data/test.parquet", "README.md"])
            self.assertIsInstance(kwargs["operations"][-1].path_or_fileobj, bytes)

    def test_missing_exports_prevent_remote_calls(self):
        import sys
        api = Mock()
        with TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(HfApi=lambda: api, CommitOperationAdd=Mock())}):
            with self.assertRaises(FileNotFoundError):
                pipeline.upload_parquet(Path(directory), "owner/repo")
        self.assertEqual(api.mock_calls, [])

    def test_local_run_never_uploads(self):
        with TemporaryDirectory() as directory, patch.object(pipeline, "generate_dataset"), patch.object(pipeline, "validate_dataset"), patch.object(pipeline, "convert_dataset"), patch.object(pipeline, "export_parquet"), patch.object(pipeline, "upload_parquet") as upload:
            pipeline.main(["--output-dir", directory])
            upload.assert_not_called()
