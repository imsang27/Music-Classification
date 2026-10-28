"""Same verified weights and full-input windows; separate familiar text list."""
from copy import deepcopy
import math
from .inference_engines import BaseEngine


class FamiliarCandidateEngine(BaseEngine):
    def _load(self):
        import numpy as np
        import torch
        from .portable_runtime import load_muq
        from .familiar_candidates import profile_spec,PROFILE_SHA256
        torch.set_num_threads(4);torch.manual_seed(20261025);np.random.seed(20261025)
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        profile=profile_spec();self.rows=[r for rows in profile['axes'].values() for r in rows]
        self.model,self.info=load_muq(self.base,self.job/'runtime',self.device)
        with torch.inference_mode():
            self.text=torch.cat([self.model(texts=[r['prompt_en'] for r in self.rows[i:i+12]]) for i in range(0,len(self.rows),12)])
        if self.text.shape!=(45,512) or not torch.isfinite(self.text).all():raise ValueError('Invalid familiar embeddings')
        self.info.update(profile_sha256=PROFILE_SHA256,profile_version=profile['version'],
                         display_policy_sha256=profile['display_policy_sha256'])

    def _infer(self,path,decoded,original_sr,original_hash):
        import numpy as np
        import torch
        import librosa
        from .familiar_candidates import ranked_axes
        audio=decoded if decoded.ndim==1 else decoded.mean(axis=1)
        size=(len(audio)*24000+original_sr-1)//original_sr
        if original_sr!=24000:
            audio=librosa.resample(audio,orig_sr=original_sr,target_sr=24000)
            audio=librosa.util.fix_length(audio,size=size)
        if len(audio)!=size or not np.isfinite(audio).all():raise ValueError('Invalid resampled audio')
        count=math.ceil(size/240000);q,remainder=divmod(size,count);start=0;bounds=[];scores=[]
        with torch.inference_mode():
            for index in range(count):
                end=start+q+(index<remainder);wave=audio[start:end]
                padded=np.tile(wave,math.ceil(240000/len(wave)))[:240000]
                latent=self.model(wavs=torch.as_tensor(padded,device=self.device).unsqueeze(0),parallel_processing=False)
                scores.append(self.model.calc_similarity(latent,self.text)[0].cpu().numpy())
                bounds.append([start,end]);start=end
        mean=np.asarray(scores,dtype=np.float64).mean(axis=0)
        result={row['id']:float(mean[i]) for i,row in enumerate(self.rows)}
        ranked_axes(result)
        return {'status':'ok','scores':result,'model':deepcopy(self.info),
            'analysis':{'audio_sha256':original_hash,'duration_seconds':len(decoded)/original_sr,
                'input_sample_rate':original_sr,'model_sample_rate':24000,'resampled_frames':size,
                'source_sample_ranges_24k':bounds,'omitted_tail_seconds':0.,
                'input_scope':'complete_supplied_audio','aggregation':'float64_equal_window_mean_native_similarity',
                'strategy':'balanced_windows_at_most10s_fixed45_prompts','padding':'repeat segment prefix to10s'}}
