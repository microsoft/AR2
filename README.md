# AR2 Project
This repo provides the code of AR2. In the paper, we propose a new *A*dversarial *R*etriever-*R*anker (AR2) framework, which constructs a unified minimax game for training the retriever and ranker models interactively.

This repo is still developing, feel free to report bugs and we will fix them ~

Starting with AR2, we developed a series of Text-Retrival methods. 

## News

- **CodeRetriever: Unimodal and Bimodal Contrastive Learning**, Xiaonan Li, Yeyun Gong, Yelong Shen, Xipeng Qiu, Hang Zhang, Bolun Yao, Weizhen Qi, Daxin Jiang, Weizhu Chen, Nan Duan ***arXiv***, [Code](https://github.com/microsoft/AR2/tree/main/CodeRetriever) [Paper](https://arxiv.org/abs/2201.10866) 
- **Distill-VQ: Learning Retrieval Oriented Vector Quantization By Distilling Knowledge from Dense Embeddings**, Shitao Xiao, Zheng Liu, Weihao Han, Jianjin Zhang, Defu Lian, Yeyun Gong, Qi Chen, Fan Yang, Hao Sun, Yingxia Shao, Denvy Deng, Qi Zhang, Xing Xie, ***SIGIR 2022***, [Code](https://github.com/staoxiao/LibVQ) [Paper](https://arxiv.org/abs/2204.00185)
- ** Adversarial Retriever-Ranker for Dense Text Retrieval**, Hang Zhang, Yeyun Gong, Yelong Shen, Jiancheng Lv, Nan Duan, Weizhu Chen, ***ICLR 2022***, [Code](https://github.com/microsoft/AR2/tree/main/AR2) [Paper](https://arxiv.org/abs/2110.03611)  



## How to Cite
If you extend or use this work, please cite the [paper](https://arxiv.org/pdf/2001.04063) where it was introduced:
```
@inproceedings{zhang2022ar2,
  title={Adversarial Retriever-Ranker model for Dense Retrieval},
  author={Hang Zhang, Yeyun Gong, Yelong Shen, Jiancheng Lv, Nan Duan, Weizhu Chen},
  booktitle={ICLR},
  year={2022}
}
@inproceedings{xiao2022distillvq,
  title={Distill-VQ: Learning Retrieval Oriented Vector Quantization By Distilling Knowledge from Dense Embeddings},
  author={Shitao Xiao, Zheng Liu, Weihao Han, Jianjin Zhang, Defu Lian, Yeyun Gong, Qi Chen, Fan Yang, Hao Sun, Yingxia Shao, Denvy Deng, Qi Zhang, Xing Xie},
  booktitle={SIGIR},
  year={2022}
}
@inproceedings{li2022coderetriever,
  title={CodeRetriever: Unimodal and Bimodal Contrastive Learning},
  author={Xiaonan Li, Yeyun Gong, Yelong Shen, Xipeng Qiu, Hang Zhang, Bolun Yao, Weizhen Qi, Daxin Jiang, Weizhu Chen, Nan Duan},
  booktitle={arXiv},
  year={2022}
}
```
