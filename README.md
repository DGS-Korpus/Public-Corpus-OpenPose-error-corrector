__Latest release:__ [![DOI](https://www.fdr.uni-hamburg.de/badge/DOI/10.25592/uhhfdm.8239.svg)](https://doi.org/10.25592/uhhfdm.8239)

__This release:__ See [release description](https://github.com/DGS-Korpus/Public-Corpus-OpenPose-error-corrector/releases/tag/v1.0.0).

# Public Corpus OpenPose error corrector

The [Public DGS Corpus](http://ling.meine-dgs.de) provides [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) data for all of its transcripts. For the total perspective of each transcript, which shows both informants and the moderator, the OpenPose output was postprocessed to make several improvements. This repository contains the code used for these postprocessing steps.

For more information, see the project note [_OpenPose in the Public DGS Corpus_](https://www.sign-lang.uni-hamburg.de/dgs-korpus/arbeitspapiere/AP06-2019-01.html).
To convert the standard one-frame-per-file output of OpenPose into the single-file wrapper format used here, see the [DGS-Korpus OpenPose wrapper](https://github.com/DGS-Korpus/DGS-Korpus-OpenPose-wrapper) script.

## Postprocessing steps
### Fix person fragmentation
Fragmentation occurs when one actual person is recognised as several supposed people, e.g. the head and left arm are recognised as one person and the right arm and legs as another. Such fragments are merged together if certain conditions are met:
  1. The fragments are reasonably close to each other;
  2. The keypoints of the fragments are either disjoint sets or they only overlap in the collarbone or hip region.

### Remove ghosts
Sometimes OpenPose recognises a person where there is none. We identify and remove such ghosts. This process relies in part on our knowledge of the studio setup of the Public DGS Corpus.

### Track people across frames
While OpenPose identifies people in a frame, it does not provide any logical connections between frames. Furthermore, OpenPose is not consistent in the order in which it lists people. This means the first person listed in frame 1 may be the third person in the people list of frame 2.

We track people across frames based on their keypoint similarity. This information is used to sort the people lists consistently. But to actually have a list position always refer to the same person across frames, we also need to ensure that the people list of each frame in fact lists the same number of people. While excess people (unresolvable fragments, ghosts) can be removed, we also need to account for people not being recognised in some frames. In such cases we insert a person entry that consists only of absent keypoints (i.e. whose coordinate/confidence triple is `(0,0,0)`).

### Limit published data to informants
The Public DGS Corpus releases pose information only for informants. The pose information of the moderator is removed.  To achieve this, we identify participants and moderator by their usual location in most frames. Thanks to the controlled environment of the videos, we know that Participant B is always seated on the left, Participant A on the right and the moderator in the middle. However, in some cases the moderator moves behind one of the participants while leaving or entering the room. Fortunately, our cross-frame person tracking allows us to identify such cases.


## Applicability to other OpenPose data
Some parts of the script rely on information specific to the Public DGS Corpus [recording environment](https://www.sign-lang.uni-hamburg.de/dgs-korpus/files/inhalt_pdf/Hanke_et_al_2010_Studio.pdf), such as the number and usual position of people. However most steps, such as the de-fragmentation and most parts of the cross-frame tracking logic, are environment-agnostic.


## Requirements
- Python 3.7
- Numpy

Tested on Python 3.7.7 with Numpy 1.18.1



## Usage
```sh
correct_openpose_errors.py [-h] [--publishmoderator] INPUT OUTPUT
```

__Positional arguments:__
 * `INPUT`: JSON file structured in the _Public DGS Corpus OpenPose wrapper format_ (see [wrapper script](https://github.com/DGS-Korpus/DGS-Korpus-OpenPose-wrapper)).
 * `OUTPUT`: Filename for the corrected JSON file.

__Optional arguments:__
 * `-h`, `--help`: show this help message and exit.
 * `-m`, `--publishmoderator`: Also include moderator data in the output.
