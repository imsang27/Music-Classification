from importlib.util import module_from_spec,spec_from_file_location
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from music_classification.app_paths import default_settings_path

ROOT=Path(__file__).resolve().parents[1]
SPEC=spec_from_file_location('public_setup',ROOT/'scripts/setup_tagging_model.py')
setup=module_from_spec(SPEC);SPEC.loader.exec_module(setup)


class PublicSetupTests(unittest.TestCase):
    def test_public_manifests_match_runtime_pins(self):
        records=setup.metadata()
        self.assertEqual(len(records['assets.json']),7)
        self.assertEqual(records['source-manifest.json']['revision'],'28847ea50cd31ac4b8b6a7dacc051ad7d1c7606a')

    def test_paths_cannot_escape_installation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):setup.child(tmp,'../outside')
            self.assertEqual(setup.child(tmp,'folder\\asset.json'),Path(tmp).resolve()/'folder/asset.json')

    def test_verified_cache_and_settings_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cached=root/'cache';cached.write_bytes(b'fixture')
            dest=root/'dest';setup.materialize(dest,setup.digest(cached),'unused',cache=cached,offline=True)
            self.assertEqual(dest.read_bytes(),b'fixture')
            with self.assertRaises(ValueError):setup.materialize(dest,'wrong','unused',offline=True)
            settings=root/'config/model_paths.local.json'
            setup.write_settings(settings,root/'model',Path(__file__))
            setup.write_settings(settings,root/'model',Path(__file__))
            with self.assertRaises(ValueError):setup.write_settings(settings,root/'different',Path(__file__))

    def test_settings_prefer_local_config_but_support_prior_bundle_and_override(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict('os.environ',{},clear=True):
            root=Path(tmp);modern=root/'config/model_paths.local.json'
            self.assertEqual(default_settings_path(root),modern)
            legacy=root/'outputs/research-release-r1/model_paths.local.json';legacy.parent.mkdir(parents=True);legacy.write_text('{}')
            self.assertEqual(default_settings_path(root),legacy)
            modern.parent.mkdir();modern.write_text('{}')
            self.assertEqual(default_settings_path(root),modern)
            with patch.dict('os.environ',{'MUSIC_CLASSIFICATION_SETTINGS':str(root/'custom.json')}):
                self.assertEqual(default_settings_path(root),root/'custom.json')


if __name__=='__main__':unittest.main()
