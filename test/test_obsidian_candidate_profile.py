import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from music_classification.obsidian_candidates import (
    ObsidianCandidateClassifier,PROFILE_SHA256,profile_spec,ranked_axes)
from music_classification.muq_runtime import digest
import obsidian_classifier_app as app


def scores():
    return {r['id']:-.9+i*.01 for i,r in enumerate(r for rows in profile_spec()['axes'].values() for r in rows)}


class FakeSession:
    mode='ok'
    def __init__(self,*args,**kwargs):self.statistics={'worker_starts':1,'model_loads':1};self.closed=False
    def close(self):self.closed=True
    def run(self,argv,*,cwd,**kwargs):
        path=Path(argv[argv.index('--audio')+1]);values=scores()
        if self.mode=='bad_scores':values.pop(next(iter(values)))
        original=digest(path)
        if self.mode=='changed':path.write_bytes(b'changed')
        payload={'status':'ok','scores':values,'analysis':{'audio_sha256':original},
                 'model':{'profile_sha256':PROFILE_SHA256,'name':'OpenMuQ/MuQ-MuLan-large'}}
        (Path(cwd)/'result.json').write_text(json.dumps(payload),encoding='utf-8')
        return type('Completed',(),{'returncode':0})()


class CandidateProfileTests(unittest.TestCase):
    def test_profile_is_exactly_the_69_approved_candidate_ids_not_deferred(self):
        root=Path(__file__).resolve().parents[1]
        if not (root/'data/label_inventory/obsidian_music_v2/candidate-vocabulary.json').is_file():
            self.skipTest('Optional private vocabulary audit is not distributed')
        vocabulary=json.loads((root/'data/label_inventory/obsidian_music_v2/candidate-vocabulary.json').read_text(encoding='utf-8'))
        spec=profile_spec()
        for axis,n in [('genre',25),('emotion',22),('mood',22)]:
            self.assertEqual(len(spec['axes'][axis]),n)
            self.assertEqual([(r['id'],r['name']) for r in spec['axes'][axis]],[(r['id'],r['label']) for r in vocabulary[axis]])
        self.assertFalse(spec['validation']['accuracy_validated'])

    def test_ranks_negative_scores_without_probability_conversion(self):
        result=ranked_axes(scores())
        for value in result.values():
            self.assertLess(value['primary']['score'],0)
            self.assertEqual(value['primary']['score'],max(value['similarity_scores'].values()))
            self.assertEqual(len(value['candidates']),3)

    def test_malformed_scores_and_bool_options_are_rejected(self):
        for change in ('missing','nan','bool','out_of_range','extra'):
            values=scores();first=next(iter(values))
            if change=='missing':values.pop(first)
            elif change=='extra':values['invented']=.5
            else:values[first]={'nan':float('nan'),'bool':True,'out_of_range':2}[change]
            with self.subTest(change=change),self.assertRaises(ValueError):ranked_axes(values)
        with self.assertRaises(ValueError):ranked_axes(scores(),top_k=True)

    def test_batch_preserves_order_and_errors_then_continues(self):
        with tempfile.TemporaryDirectory() as tmp,patch('music_classification.obsidian_candidates.WorkerSession',FakeSession):
            p=Path(tmp);(p/'runtime-manifest.json').write_text('{}');good=p/'곡.wav';good.write_bytes(b'fake audio')
            classifier=ObsidianCandidateClassifier(installation=p,python_executable=Path(__file__),work_directory=p/'jobs',cache_directory=p/'cache')
            FakeSession.mode='ok';batch=classifier.classify_many([good,p/'missing.wav',good])
            self.assertEqual([r['status'] for r in batch['results']],['ok','invalid_audio','ok'])
            self.assertEqual(batch['results'][0]['axes'],batch['results'][2]['axes'])
            self.assertFalse(list((p/'jobs').glob('request-*')))

    def test_changed_input_or_wrong_score_space_never_returns_success(self):
        for mode in ('changed','bad_scores'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp,patch('music_classification.obsidian_candidates.WorkerSession',FakeSession):
                p=Path(tmp);(p/'runtime-manifest.json').write_text('{}');audio=p/'a.wav';audio.write_bytes(b'fake audio')
                classifier=ObsidianCandidateClassifier(installation=p,python_executable=Path(__file__),work_directory=p/'jobs',cache_directory=p/'cache')
                FakeSession.mode=mode;result=classifier.classify(audio)
                self.assertFalse(result['success']);self.assertTrue(all(a['status']!='ok' for a in result['axes'].values()))
        FakeSession.mode='ok'

    def test_json_save_preserves_negative_scores_and_protects_input_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);audio=p/'audio.json';settings=p/'settings.json'
            audio.write_text('original');settings.write_text('settings')
            doc={'batch':{'input_paths':[str(audio)],'results':[]},'negative':-.75}
            for path in (audio,settings):
                with self.assertRaises(ValueError):app.save_result(doc,path,settings,overwrite=True)
            output=p/'result.json';app.save_result(doc,output,settings)
            self.assertEqual(json.loads(output.read_text()),doc)
            with self.assertRaises(FileExistsError):app.save_result(doc,output,settings)
            self.assertEqual(audio.read_text(),'original');self.assertEqual(settings.read_text(),'settings')


if __name__=='__main__':unittest.main()
