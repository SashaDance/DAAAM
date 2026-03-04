## OC-NaVQA annotations

In ./oc-navqa_data.csv, we provide the corrected annotations for the NaVQA dataset (see [ReMEmbR](https://github.com/NVIDIA-AI-IOT/remembr)).

The annotations incorporate three major differences:

- They expand the horizon of the questions: the context window is always from the beginning of the sequence until the `current time` .
- They correct position annotations using the ground truth annotations of the UT Campus Object Dataset [CODa](https://amrl.cs.utexas.edu/coda/) such that position annotations are where the object _is_ rather than from which pose it was observed.
- They resolve ambiguities in the framing of the questions.

Please refer to [ReMEmbR](https://github.com/NVIDIA-AI-IOT/remembr) to see how the data can be used. It can simply be replaced with the [data.csv](https://github.com/NVIDIA-AI-IOT/remembr/blob/main/remembr/data/navqa/data.csv) of the remembr NaVQA dataset.

To process the data into the remembr `question_jsons`, refer to the script in our fork of remembr: https://github.com/nicogorlo/remembr/blob/main/remembr/scripts/question_scripts/form_question_jsons_fullseq.py .

When using this data, for curtesy please also cite [ReMEmbR](https://arxiv.org/abs/2409.13682), as OC-NaVQA is a derivative of their data with some changes.