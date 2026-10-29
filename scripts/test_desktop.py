"""CPU-only source/GUI smoke suite; no model or YouTube downloads."""
from pathlib import Path
import os
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    temp=ROOT/'work/test-temp';temp.mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=str(temp)
    tempfile.tempdir=None
    suite=unittest.TestSuite()
    for name in ('test_youtube_input.py','test_candidate_presentation.py','test_familiar_candidates.py',
                 'test_hierarchical_tagging.py','test_obsidian_candidate_profile.py','test_obsidian_candidate_window.py',
                 'test_public_setup.py', 'test_local_tagging_server.py', 'test_web_audio_input.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'test'),pattern=name))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return not result.wasSuccessful()


if __name__=='__main__':raise SystemExit(main())
