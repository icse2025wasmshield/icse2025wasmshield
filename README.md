## Important Note
This repository contains the code for the ICSE2025WASMSHIELD paper submission 03/08/2024. We intend to share a more complete artifact for the 'artifact evaluation track' including the generated data. The most recent version of the artifact is available in the following branch: [updated_artifact_link](https://github.com/icse2025wasmshield/icse2025wasmshield/tree/artifact).

## Structure



## Pre-Training

- **IMAGE GENERATION** : an exemple of image preprocessing of various pre-transformed WASM binary files is available in *image_generation_example.ipynb*.
- **PRE-TRAINING LOOP** : an example of the pre-training loop of ResBin is available in *training_loop_example.ipynb*.

## Pre-Trained ResBin models
These models are pretrained with a batch_size of 64, an image_size of 64, a contrastive loss temperature of 0.15:
- **ResBin18SA** ( *Emprique60_2_ResBin18_b64_i64_pil_v128_rmax_t015* )
- **ResBin8SA** ( *Emprique60_2_ResBin8_b64_i64_pil_v128_rmax_t015* )
- **ResBin18** ( *Emprique60_2_ResBin18_woSA_b64_i64_pil_v128_rmax_t015* )
- **ResBin8** ( *Emprique60_2_ResBin8_woSA_b64_i64_pil_v128_rmax_t015* )

The models are available under the *saving_models* folder and can be loaded as follows:

```
from wasmshield.models.resbin import (
    build_resbin_8,
    build_resbin_8_sa, 
    build_resbin_18, 
    build_resbin_18_sa,
    ResBinHandler
)
resbin_8_handler = ResBinHandler(build_resbin_8())
```

## Results
The results of each research question is in a different notebook:
- **RQ1** ( *rq1_embedding_generation_time.ipynb* ) : contains the results of embedding generation time.
- **RQ2** ( *rq2_transforms_resistance.ipynb* ) : contains the results of the similarity inference in the contexte of transformed WASM binaires.
- **RQ3.1** ( *rq3_1_malware_real_detection.ipynb* ) : contains the results of real transformed malware detection.
- **RQ3.2** ( *rq3_2_malware_wobfuscator_detection.ipynb* ) : contains the results of wobfuscator malware detection.
- **RQ4** ( *rq4_semantic_classification.ipynb* ) : contains the results of the semantic classification.

