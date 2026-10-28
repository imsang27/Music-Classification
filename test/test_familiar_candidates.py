import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from music_classification.familiar_candidates import (
    FamiliarCandidateClassifier,PROFILE_SHA256,profile_spec,ranked_axes)
from music_classification.obsidian_candidates import profile_spec as old_spec
from tag_audio import run


def scores():
    return {r['id']:-.9+i*.01 for i,r in enumerate(r for rows in profile_spec()['axes'].values() for r in rows)}


class Session:
    mode='ok'
    def __init__(self,*args,**kwargs):self.statistics={'worker_starts':1,'model_loads':1}
    def close(self):pass
    def run(self,argv,*,cwd,**kwargs):
        audio=Path(argv[argv.index('--audio')+1])
        sha=hashlib.sha256(audio.read_bytes()).hexdigest()
        values=scores()
        if self.mode=='incomplete':values.pop(next(iter(values)))
        model={'profile_sha256':PROFILE_SHA256,'name':'OpenMuQ/MuQ-MuLan-large'}
        if self.mode=='wrong_profile':model['profile_sha256']='wrong'
        if self.mode=='changed':audio.write_bytes(b'changed')
        payload={'status':'ok','scores':values,'model':model,'analysis':{'audio_sha256':sha}}
        (Path(cwd)/'result.json').write_text(json.dumps(payload),encoding='utf-8')
        return type('Completed',(),{'returncode':0})()


class FamiliarProfileTests(unittest.TestCase):
    def test_closed_vocabulary_is9_emotions11_moods_and_genres_unchanged(self):
        spec=profile_spec()
        self.assertEqual(spec['axes']['genre'],old_spec()['axes']['genre'])
        self.assertEqual([r['name'] for r in spec['axes']['emotion']],['신남','슬픔','분노','불안','설렘','그리움','외로움','희망','사랑'])
        self.assertEqual(spec['version'],'familiar_audio_candidates_v2')
        baseline=json.loads((Path(__file__).resolve().parents[1]/'music_classification/resources/familiar_candidate_prompts_v1.json').read_text(encoding='utf-8'))
        excited=spec['axes']['emotion'][0]
        self.assertNotEqual(excited['id'],baseline['axes']['emotion'][0]['id'])
        self.assertNotEqual(excited['prompt_en'],baseline['axes']['emotion'][0]['prompt_en'])
        self.assertEqual(spec['axes']['emotion'][1:],baseline['axes']['emotion'][1:])
        self.assertEqual(next(r for r in spec['axes']['mood'] if r['name']=='발랄'),next(r for r in baseline['axes']['mood'] if r['name']=='발랄'))
        self.assertEqual(len(spec['axes']['mood']),11)
        self.assertFalse(spec['validation']['accuracy_validated'])
        self.assertTrue(all(r['definition_ko'] and r['prompt_en'] for a in ('emotion','mood') for r in spec['axes'][a]))

    def test_negative_scores_preserved_and_only_known_names_returned(self):
        axes=ranked_axes(scores())
        for axis,value in axes.items():
            self.assertEqual(value['primary']['score'],max(value['similarity_scores'].values()))
            self.assertLess(value['primary']['score'],0)
            self.assertEqual(len(value['candidates']),3)
            self.assertTrue({r['name'] for r in value['candidates']} <= {r['name'] for r in profile_spec()['axes'][axis]})

    def test_bad_score_spaces_are_rejected(self):
        for mode in ('missing','extra','nan','bool','out_of_range'):
            values=scores();first=next(iter(values))
            if mode=='missing':values.pop(first)
            elif mode=='extra':values['invented']=0.
            else:values[first]={'nan':float('nan'),'bool':True,'out_of_range':2}[mode]
            with self.subTest(mode=mode),self.assertRaises(ValueError):ranked_axes(values)
        with self.assertRaises(ValueError):ranked_axes(scores(),True)

    def test_order_error_recovery_and_wrong_worker_results(self):
        for mode in ('ok','wrong_profile','incomplete','changed'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp,patch('music_classification.familiar_candidates.WorkerSession',Session):
                root=Path(tmp);(root/'runtime-manifest.json').write_text('{}')
                audio=root/'곡.wav';audio.write_bytes(b'fixture')
                classifier=FamiliarCandidateClassifier(installation=root,python_executable=Path(__file__),work_directory=root/'jobs',cache_directory=root/'cache')
                Session.mode=mode
                result=classifier.classify_many([audio,root/'missing.wav',audio])
                if mode=='ok':
                    self.assertEqual([r['status'] for r in result['results']],['ok','invalid_audio','ok'])
                elif mode=='changed':
                    self.assertFalse(result['results'][0]['success'])
                    self.assertEqual(result['results'][1]['status'],'invalid_audio')
                    # The next request sees the now-stable new bytes as its input.
                    self.assertTrue(result['results'][2]['success'])
                else:self.assertTrue(all(not r['success'] for r in result['results']))
                self.assertFalse(list((root/'jobs').glob('request-*')))
        Session.mode='ok'

    def test_familiar_cli_routes_to_new_classifier_and_keeps_legacy_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audio=root/'a.wav';audio.write_bytes(b'fixture')
            sha=hashlib.sha256(audio.read_bytes()).hexdigest()
            raw={'profile_sha256':PROFILE_SHA256,'batch':{'input_paths':[str(audio)],'results':[
                {'success':True,'status':'ok','axes':ranked_axes(scores()),'analysis':{'audio_sha256':sha}}]}}
            original=copy.deepcopy(raw)
            with patch('music_classification.familiar_candidates.classify_paths',return_value=raw) as classifier:
                target,code=run([audio],root/'out',profile='familiar')
            self.assertEqual(code,0);classifier.assert_called_once()
            result=json.loads((target/'tags.json').read_text(encoding='utf-8'))
            self.assertEqual(result['schema'],'music_familiar_tags_v1')
            self.assertEqual(result['vocabulary'],'familiar_audio_candidates_v2')
            self.assertEqual(raw,original)
            self.assertEqual(result['results'][0]['labels']['emotion'],[raw['batch']['results'][0]['axes']['emotion']['primary']['name']])


if __name__=='__main__':unittest.main()
