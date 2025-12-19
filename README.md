# D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI

Suhwan Choi*, Jaeyoon Jung*, Haebin Seong*, Minchan Kim, Minyeong Kim, Yongjun Cho, Yoonshik Kim, Yubeen Park, Youngjae Yu‡, Yunsung Lee‡

[![project-page](https://img.shields.io/badge/Project%20Page-blue?style=flat-square)](https://worv-ai.github.io/d2e/) [![arXiv](https://img.shields.io/badge/arXiv-2410.01273-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2510.05684)

<img width="5334" height="2306" alt="image" src="https://github.com/user-attachments/assets/5a6dd6a1-201f-4da6-8530-340b5edfa2b1" />

## News

- [2025/12/18] We release the FHD/QHD versions of the dataset on Hugging Face [open-world-agents/D2E-Original](https://huggingface.co/datasets/open-world-agents/D2E-Original) for training world models and video generation models. We also fix issues in the 480p dataset [open-world-agents/D2E-480p](https://huggingface.co/datasets/open-world-agents/D2E-480p).

- [2025/12/01] We release 480p version of dataset at huggingface. [open-world-agents/D2E-480p](https://huggingface.co/datasets/open-world-agents/D2E-480p): **267 hours** of synchronized video, audio, and input events from **29** PC games across diverse genres (FPS, open-world, sandbox, and more), for training vision-action models and game agents.

- [2025/10/21] We release part of our source codes. Code is comming soon! `ocap` and `owa` toolkit is being open-sourced already, have a look at these first.
  - https://github.com/open-world-agents/ocap: ocap (Omnimodal CAPture) captures all essential desktop signals in synchronized format. Records screen video, audio, keyboard/mouse input, and window events.
  - https://github.com/open-world-agents/open-world-agents: A versatile and efficient monorepo that embraces and grows multiple projects, containing all the essential building blocks for agent development.
  - https://worv-ai.github.io/d2e/: D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI. Code will coming soon!

## Citation

If you find this work useful, please cite our paper:

```
@article{choi2025d2e,
  title={D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI},
  author={Choi, Suhwan and Jung, Jaeyoon and Seong, Haebin and Kim, Minchan and Kim, Minyeong and Cho, Yongjun and Kim, Yoonshik and Park, Yubeen and Yu, Youngjae and Lee, Yunsung},
  journal={arXiv preprint arXiv:2510.05684},
  year={2025}
}
```
