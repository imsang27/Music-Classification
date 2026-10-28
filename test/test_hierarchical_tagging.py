import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from music_classification.hierarchical_tagging import apply_hierarchy, taxonomy_spec
from music_classification.obsidian_candidates import PROFILE_SHA256, profile_spec, ranked_axes
from tag_audio import assigned_tags, markdown, run


class HierarchicalTaggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'曲.wav'
        self.path.write_bytes(b'synthetic fixture, not audio inference')
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        profile = profile_spec()
        scores = {r['id']:-.8+i*.01 for i,r in enumerate(r for rows in profile['axes'].values() for r in rows)}
        self.raw = {'profile_sha256':PROFILE_SHA256,'batch':{'input_paths':[str(self.path)],
            'results':[{'success':True,'status':'ok','axes':ranked_axes(scores),
                        'analysis':{'audio_sha256':self.sha}}]}}
        self.tags = assigned_tags(self.raw)
        self.spec,self.tax_sha = taxonomy_spec()

    def test_all44_names_and_ids_preserved_under15_parents(self):
        self.assertEqual(sum(len(t['parents']) for t in self.spec['axes'].values()),15)
        for axis in ('emotion','mood'):
            self.assertEqual(len(self.spec['axes'][axis]['labels']),22)
            self.assertTrue(all(r['definition'] and r['distinction'] for r in self.spec['axes'][axis]['labels']))

    def test_no_model_means_single_fallback_not_three_simultaneous_tags(self):
        original = copy.deepcopy(self.tags)
        value = apply_hierarchy(self.tags,self.raw)
        self.assertEqual(self.tags,original)
        self.assertEqual(value['results'][0]['labels'],self.tags['results'][0]['labels'])
        self.assertFalse(value['supervised_models_available'])
        for axis in ('emotion','mood'):
            self.assertEqual(value['results'][0]['hierarchy'][axis]['assignment'],'nearest_similarity_fallback')

    def test_supported_decisions_cap_three_and_keep_genre_unchanged(self):
        ids = [r['id'] for r in self.spec['axes']['emotion']['labels'][:5]]
        registry = {'fixture':{'accepted':True,'label_ids':ids,'thresholds':dict.fromkeys(ids,.5),'hierarchy_sha256':self.tax_sha}}
        decisions = {self.sha:{'emotion':[{'label_id':label,'score':score,'threshold':.5,
            'audio_sha256':self.sha,'model_id':'fixture'} for label,score in zip(ids,[.9,.8,.7,.6,.1])]}}
        value = apply_hierarchy(self.tags,self.raw,supervised_decisions=decisions,accepted_models=registry)
        row = value['results'][0]
        self.assertEqual(len(row['labels']['emotion']),3)
        self.assertEqual(row['details']['emotion']['selected_ids'],ids[:3])
        self.assertEqual(row['labels']['genre'],self.tags['results'][0]['labels']['genre'])
        self.assertEqual(len(row['labels']['mood']),1)
        self.assertFalse(row['hierarchy']['mood']['multilabel_supported'])
        self.assertIn(' · ',markdown(value))

    def test_unaccepted_wrong_audio_taxonomy_or_threshold_never_assigns(self):
        label = self.spec['axes']['emotion']['labels'][0]['id']
        model = {'accepted':True,'label_ids':[label],'thresholds':{label:.6},'hierarchy_sha256':self.tax_sha}
        decision = {'label_id':label,'score':.9,'threshold':.6,'model_id':'fixture','audio_sha256':self.sha}
        for mode in ('unaccepted','audio','taxonomy','threshold','nan','bool','unknown'):
            registry = {'fixture':copy.deepcopy(model)}
            row = copy.deepcopy(decision)
            if mode=='unaccepted':registry['fixture']['accepted']=False
            elif mode=='audio':row['audio_sha256']='wrong'
            elif mode=='taxonomy':registry['fixture']['hierarchy_sha256']='wrong'
            elif mode=='threshold':row['threshold']=.3
            elif mode=='nan':row['score']=float('nan')
            elif mode=='bool':row['score']=True
            elif mode=='unknown':row['label_id']='invented'
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                apply_hierarchy(self.tags,self.raw,supervised_decisions={self.sha:{'emotion':[row]}},accepted_models=registry)

    def test_failed_audio_stays_unlabelled_and_success_order_preserved(self):
        self.raw['batch']['input_paths'].insert(0,'missing.wav')
        self.raw['batch']['results'].insert(0,{'success':False,'status':'invalid_audio'})
        result = apply_hierarchy(assigned_tags(self.raw),self.raw)
        self.assertIsNone(result['results'][0]['labels'])
        self.assertNotIn('hierarchy',result['results'][0])
        self.assertEqual(result['results'][1]['audio_path'],str(self.path))

    def test_ties_keep_original_primary_and_raw_negative_scores(self):
        scores = {r['id']:-.5 for rows in profile_spec()['axes'].values() for r in rows}
        self.raw['batch']['results'][0]['axes'] = ranked_axes(scores)
        tags = assigned_tags(self.raw)
        result = apply_hierarchy(tags,self.raw)
        self.assertEqual(result['results'][0]['labels'],tags['results'][0]['labels'])
        self.assertEqual(result['results'][0]['hierarchy']['emotion']['selected'][0]['score'],-.5)

    def test_cli_profiles_export_separately_and_do_not_overwrite(self):
        for profile,schema in [('single','music_auto_tags_v1'),('hierarchical','music_hierarchical_tags_v1')]:
            target = Path(self.temp.name)/profile
            with patch('tag_audio.classify_paths',return_value=self.raw):
                _,code = run([self.path],target,profile=profile)
            self.assertEqual(code,0)
            value = json.loads((target/'tags.json').read_text(encoding='utf-8'))
            self.assertEqual(value['schema'],schema)
            with self.assertRaises(FileExistsError):run([self.path],target,profile=profile)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(),self.sha)

    def test_prepared_learning_gate_has_no_fabricated_binary_truth_or_weights(self):
        root = Path(__file__).resolve().parents[1]
        folder = root/'data/training/emotion_mood_hierarchy_v1'
        if not (folder/'training-readiness.json').is_file():
            self.skipTest('Optional local research-readiness audit is not distributed')
        state = json.loads((folder/'training-readiness.json').read_text(encoding='utf-8'))
        self.assertEqual(state['eligible_labels'],0)
        self.assertFalse(state['training_performed'])
        self.assertFalse(state['thresholds_fitted'])
        self.assertFalse(state['weights_created'])
        labels = json.loads((folder/'label-availability.json').read_text(encoding='utf-8'))['labels']
        self.assertEqual(len(labels),44)
        self.assertTrue(all(not r['training_eligible'] for r in labels))


if __name__ == '__main__':
    unittest.main()
