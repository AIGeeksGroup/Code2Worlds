# Code2Worlds: Empowering Coding LLMs for 4D World Generation

This is the official repository for the paper:
> **Code2Worlds: Empowering Coding LLMs for 4D World Generation**
>
> Yi Zhang\*, Yunshuang Wang\*, Zeyu Zhang\*<sup>†</sup>, and Hao Tang<sup>‡</sup>
>
> School of Computer Science, Peking University
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>‡</sup>Corresponding author
>
> ### [Paper](https://arxiv.org/abs/2602.11757) | [Website](https://aigeeksgroup.github.io/Code2Worlds) | [HF](https://huggingface.co/datasets/AIGeeksGroup/Code4D)

> [!NOTE]
> 💪 This project demonstrates the capability of coding LLMs in generating dynamic 4D worlds through code-based approaches.

## ✏️ Citation
If you find our code or paper helpful, please consider starring ⭐ us and citing:
```bibtex
@article{zhang2026code2worlds,
  title={Code2Worlds: Empowering Coding LLMs for 4D World Generation},
  author={Zhang, Yi and Wang, Yunshuang and Zhang, Zeyu and Tang, Hao},
  journal={arXiv preprint arXiv:2602.11757},
  year={2026}
}
```
---

## 🏃 Intro Code2Worlds
Achieving spatial intelligence requires moving beyond visual plausibility to build world simulators grounded in physical laws. While coding LLMs have advanced static 3D scene generation, extending this paradigm to 4D dynamics remains a critical frontier. This task presents two fundamental challenges: multi-scale context entanglement, where monolithic generation fails to balance local object structures with global environmental layouts; and a semantic-physical execution gap, where open-loop code generation leads to physical hallucinations lacking dynamic fidelity. We introduce **Code2Worlds**, a framework that formulates 4D generation as language-to-simulation code generation. First, we propose a dual-stream architecture that disentangles retrieval-augmented object generation from hierarchical environmental orchestration. Second, to ensure dynamic fidelity, we establish a physics-aware closed-loop mechanism in which a Post-Process Agent scripts dynamics, coupled with a VLM-Motion Critic that performs self-reflection to iteratively refine simulation code.Evaluations on the Code4D benchmark show Code2Worlds outperforms baselines with a 41% SGS gain and 49% higher Richness, while uniquely generating physics-aware dynamics absent in prior static methods.

![image](./assets/overview.png)

<!-- ## 📰 News
<b>2025/02/13:</b> 🎉 Project repository created. -->

## TODO List

- [ ] Upload our paper to arXiv and build project pages.
- [ ] Upload the code.
- [ ] Add a demo.

<!-- 
## ⚡ Quick Start
### Environment Setup


```bash
pip install -r requirements.txt
```


## 🧪 Run
Run evaluation/inference with a trained checkpoint:
```bash

```

## 👀 Visualization
[Add visualization instructions here] -->


---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=username/Code2Worlds&type=Date)](https://www.star-history.com/#username/Code2Worlds&Date)


<!-- ## 😘 Acknowledgement
We thank the authors of [relevant projects] for their open-source code. -->