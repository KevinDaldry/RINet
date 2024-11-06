Open Source for the codes of RINet in [*A Regionally Indicated Visual Grounding Network for Remote Sensing Images*](https://ieeexplore.ieee.org/document/10741531).

<div align="center">
  <img src="https://github.com/KevinDaldry/RINet/blob/main/WorkFlow.png">
</div>

Considering the possible inconvenience brought to the creaters of RSVG and DIOR-RSVG datasets, please contact the authors of *Visual Grounding in Remote Sensing Images* and *RSVG: Exploring Data and Models for Visual
Grounding on Remote Sensing Data* for the permission to use their datasets and codes accordingly. 

Besides, as our proposed network are trained and evaluated partly based on the codes provided by the authors of *Visual Grounding in Remote Sensing Images*, we want to announce our gratitude to them. In the meantime, it means that our proposed RINet can be easily plugged into their training and validating process, which means you can simply import our RINet into their project, but please remember to adjust the hyper-parameters.  

Also, since DIOR-RSVG is proposed with a Transformer-based model, which means there is no officially available anchors for two-stage methods to use. As a result, we share the anchors selected and used by our RINet below.

For RSVG, they are: ['19,44, 19,20, 35,34, 37,72, 57,48, 73,123, 76,75, 137,103, 202,222'].
For DIOR-RSVG, they are: ['18,20, 34,56, 43,27, 76,68, 119,124, 109,253, 200,167, 255,300, 472,453'].

For more information and communication, please contact me at 202212490401@nusit.edu.cn or raise an issue here, I will reply to you when I received the message.

If you find this work interesting, you can refer its citation here:

    @ARTICLE{10741531,
    author={Hang, Renlong and Xu, Siqi and Liu, Qingshan},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={A Regionally Indicated Visual Grounding Network for Remote Sensing Images}, 
    year={2024},
    volume={},
    number={},
    pages={1-1},
    keywords={Feature extraction;Visualization;Remote sensing;Proposals;Transformers;Grounding;Sensors;Logic gates;Fuses;Detectors;Visual Grounding;Remote Sensing Images;Complex Language Expressions;Small-Scale Objects},
    doi={10.1109/TGRS.2024.3490847}}
