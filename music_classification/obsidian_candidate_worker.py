"""Private serial worker; same bounded request/atomic response protocol."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from music_classification.batch_worker import atomic_json


def serve(args):
    from music_classification.obsidian_candidate_engine import ObsidianCandidateEngine
    import json
    engine=ObsidianCandidateEngine(args.base,args.job,args.cache);requests=args.requests_root.resolve();processed=0
    for line in sys.stdin:
        command=json.loads(line);output=Path(command['output']).resolve()
        if output.name!='result.json' or output.parent.parent!=requests or not output.parent.name.startswith('request-'):
            raise ValueError('Unsafe response path')
        try:payload=engine.predict(Path(command['audio']))
        except Exception as exc:payload={'status':'error','error':f'{type(exc).__name__}: {exc}'}
        processed+=1;stats={'model_loaded':engine.loaded,'processed':processed}
        if engine.loaded:
            import torch
            if torch.cuda.is_available():stats['cuda_memory']={'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
        atomic_json(output.with_name('worker-stats.json'),stats);atomic_json(output,payload)
        if payload['status']=='error':break


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['obsidian_candidates'],required=True)
    for option in ('base','job','cache','requests-root'):p.add_argument('--'+option,type=Path,required=True)
    serve(p.parse_args())
