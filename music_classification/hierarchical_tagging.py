"""Opt-in hierarchy and evidence-gated multilabel presentation.

Raw text similarities are never thresholds for simultaneous tags. Until an
accepted supervised decision is available, preserve the original nearest tag.
"""
from copy import deepcopy
from hashlib import sha256
from importlib.resources import files
import json
import math

from .obsidian_candidates import PROFILE_SHA256, profile_spec


def taxonomy_spec():
    raw = files('music_classification').joinpath('resources/hierarchical_taxonomy_v1.json').read_bytes()
    spec = json.loads(raw)
    if spec['source_profile_sha256'] != PROFILE_SHA256 or spec['max_tags_per_axis'] != 3:
        raise ValueError('Hierarchy profile does not match the fixed model profile')
    original = profile_spec()
    for axis in ('emotion','mood'):
        tree = spec['axes'][axis]
        parents = {r['id'] for r in tree['parents']}
        rows = tree['labels']
        if len(parents)!=len(tree['parents']) or len(rows)!=22 or len({r['id'] for r in rows})!=22:
            raise ValueError('Invalid hierarchy IDs')
        if {r['id']:r['name'] for r in rows}!={r['id']:r['name'] for r in original['axes'][axis]}:
            raise ValueError('Canonical label IDs and names must be preserved')
        if any(r['parent_id'] not in parents for r in rows):
            raise ValueError('Unknown parent')
    return spec, sha256(raw).hexdigest()


def _finite(value):
    return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def select_labels(axis_result, tree, decisions, accepted_models, audio_sha256):
    """Select <=3 supported positives; unobserved labels remain unknown.

decisions is an optional upstream supervised result, NOT a model training API.
An accepted model registry is mandatory; the shipped registry is empty.
"""
    rows = {r['id']:r for r in tree['labels']}
    scores = axis_result['similarity_scores']
    if set(scores)!=set(rows) or any(not _finite(v) or not -1.000001<=v<=1.000001 for v in scores.values()):
        raise ValueError('Exact finite native score space required')
    parent_scores = {}
    for row in rows.values():
        parent = row['parent_id']
        parent_scores[parent] = max(parent_scores.get(parent,-math.inf),scores[row['id']])
    parent_order = sorted(parent_scores,key=parent_scores.get,reverse=True)
    selected = []
    seen = set()
    for decision in decisions:
        label = decision['label_id']
        if label not in rows or label in seen:
            raise ValueError('Unknown or duplicate supervised decision')
        seen.add(label)
        model = accepted_models.get(decision['model_id'])
        if not model or model.get('accepted') is not True or label not in model['label_ids']:
            raise ValueError('No accepted supervised model for this label')
        if decision.get('audio_sha256') != audio_sha256:
            raise ValueError('Supervised decision belongs to different audio')
        score, threshold = decision['score'], decision['threshold']
        if (not _finite(score) or not _finite(threshold) or not 0<=score<=1 or not 0<threshold<=1
                or threshold!=model['thresholds'].get(label)):
            raise ValueError('Invalid or changed decision threshold')
        if score>=threshold:
            selected.append((label,score,decision['model_id']))
    selected.sort(key=lambda r:r[1],reverse=True)
    if selected:
        selected = selected[:3]
        origin = 'supported_supervised_decisions'
    else:
        # Parent routing is a transparent organization of existing similarities,
        # not evidence that a broad class or multiple children are correct.
        best = axis_result['primary']['label']
        if best not in rows or scores[best]!=max(scores.values()):
            raise ValueError('Invalid primary label')
        parent = rows[best]['parent_id']
        selected = [(best,scores[best],None)]
        origin = 'nearest_similarity_fallback'
    parent_by_id = {r['id']:r['name'] for r in tree['parents']}
    return {
        'labels':[rows[label]['name'] for label,_,_ in selected],
        'selected':[{'id':label,'name':rows[label]['name'],'parent_id':rows[label]['parent_id'],
                     'parent':parent_by_id[rows[label]['parent_id']], 'score':score,
                     'model_id':model_id} for label,score,model_id in selected],
        'parents':list(dict.fromkeys(parent_by_id[rows[label]['parent_id']] for label,_,_ in selected)),
        'assignment':origin,'multilabel_supported':any(m.get('accepted') is True and set(m['label_ids'])&set(rows) for m in accepted_models.values()),
        'parent_candidates':[{'id':p,'name':parent_by_id[p],'raw_max_child_similarity':parent_scores[p]} for p in parent_order],
        'parent_score_type':'derived_max_child_similarity_not_probability',
    }


def apply_hierarchy(tags, raw, *, supervised_decisions=None, accepted_models=None):
    """Versioned export; never mutate the legacy raw document or tag document."""
    spec, taxonomy_sha = taxonomy_spec()
    if raw.get('profile_sha256')!=PROFILE_SHA256 or tags.get('profile_sha256')!=PROFILE_SHA256:
        raise ValueError('Wrong base profile')
    if len(tags['results'])!=len(raw['batch']['results']):
        raise ValueError('Result count mismatch')
    supervised_decisions = supervised_decisions or {}
    accepted_models = accepted_models or {}
    if any(model.get('hierarchy_sha256')!=taxonomy_sha for model in accepted_models.values()):
        raise ValueError('Supervised registry belongs to a different taxonomy')
    output = deepcopy(tags)
    output.update(schema='music_hierarchical_tags_v1',assignment='supported_multilabel_or_nearest_single_fallback',
                  hierarchy_version=spec['version'],hierarchy_sha256=taxonomy_sha,
                  max_tags_per_axis=3,supervised_models_available=any(m.get('accepted') is True for m in accepted_models.values()))
    for row, original in zip(output['results'],raw['batch']['results']):
        if row['labels'] is None:
            continue
        if original['analysis']['audio_sha256']!=row['audio_sha256']:
            raise ValueError('Audio hash differs from source inference')
        decisions = supervised_decisions.get(row['audio_sha256'],{})
        row['hierarchy'] = {}
        row['primary_labels'] = {}
        for axis in ('emotion','mood'):
            result = select_labels(original['axes'][axis],spec['axes'][axis],decisions.get(axis,[]),accepted_models,row['audio_sha256'])
            row['labels'][axis] = result['labels']
            row['primary_labels'][axis] = result['labels'][0]
            row['hierarchy'][axis] = result
            row['details'][axis] = {
                'selected_ids':[value['id'] for value in result['selected']],
                'assignment':result['assignment'],
                'selected_scores':[value['score'] for value in result['selected']],
                'score_type':'supervised_decision_score' if result['assignment']=='supported_supervised_decisions' else 'raw_similarity',
                'similarity_primary':deepcopy(original['axes'][axis]['primary']),
                'similarity_candidates':deepcopy(original['axes'][axis]['candidates']),
            }
        row['primary_labels']['genre'] = row['labels']['genre'][0]
        row['warnings'] = ['Parent categories are editorial organization, not validated acoustic classes.',
                           'Without accepted supervised decisions only one nearest label is assigned; alternatives are not simultaneous tags.']
    return output
