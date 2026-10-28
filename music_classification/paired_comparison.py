"""Fixed-threshold model comparison with paired, class-stratified resampling."""

import math
import random
from music_classification.benchmark import binary_classification_metrics


def compare_models(truth, baseline, candidate, *, samples=2000, seed=20261025):
    if any(not math.isfinite(x) or not 0 <= x <= 1 for scores in (baseline,candidate) for x in scores):
        raise ValueError('Finite unit scores required')
    before=binary_classification_metrics(truth,baseline)
    after=binary_classification_metrics(truth,candidate)
    if before['balanced_accuracy'] is None or samples<100:
        raise ValueError('Both classes and sufficient resamples required')
    groups=[[i for i,y in enumerate(truth) if y==label] for label in (0,1)]
    rng=random.Random(seed)
    values=[]
    for _ in range(samples):
        delta=0.
        for label,group in enumerate(groups):
            drawn=rng.choices(group,k=len(group))
            delta+=sum(int((candidate[i]>=.5)==bool(label))-int((baseline[i]>=.5)==bool(label)) for i in drawn)/len(group)/2
        values.append(delta)
    values.sort()
    def percentile(q):
        at=q*(len(values)-1);lo=int(at)
        return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(at-lo)
    return {'effnet_specialist':before,'music2emo':after,
            'balanced_accuracy_delta':after['balanced_accuracy']-before['balanced_accuracy'],
            'paired_bootstrap_delta_95_percentile':[percentile(.025),percentile(.975)],
            'resamples':samples,'seed':seed}
