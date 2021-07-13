# Automatic Speech Recognition
 - ASR 공부를 위한 저장소

#### Conventional ASR Model(CTC Method)

![Screenshot from 2021-07-12 20-40-43](https://user-images.githubusercontent.com/76771847/125281752-91a07400-e351-11eb-8b30-3c40b7d3ca94.png)

 - End-to-End 가 아닌 Language, Acoustic, Pronunciation 따로 학습

#### End-To-End Deep Learning(LAS model)

![Las](https://user-images.githubusercontent.com/76771847/125281954-cf9d9800-e351-11eb-899d-6dcd2f01418b.png)

 - Encoder 가 Acoustic model 유사
   
 - Decoder 가 Language model과 유사(따료 어기서는 LM, PM 구분 X)

## Paper

https://arxiv.org/abs/1412.5567: deep speech 1(ASR 분야 end-to-end 거의 처음 도입)

https://arxiv.org/abs/1512.02595: deep speech 2

https://arxiv.org/abs/1506.07503: Attention-Based Models for Speech Recognition

https://arxiv.org/abs/1508.01211: Listen, Attend and Spell(기존에는 CTC방식이 주료 였는데 혁신적인 방법도입)

https://arxiv.org/abs/2004.09367: ClovaCall(코드 참고)

https://arxiv.org/abs/1904.08779: SpecAugment



## Code
https://github.com/clovaai/ClovaCall

## Reference
https://ratsgo.github.io/speechbook/docs/neuralam: ASR Tutorial

https://github.com/sooftware/Speech-Recognition-Tutorial: ASR tutorial 설명

https://github.com/sooftware/KoSpeech: 한국의 open source code 2개중 하나
(clovacall, kospeech)

https://hyongdoc.tistory.com/401: librosa(stft) 설명

https://kaen2891.tistory.com/39: librosa(stft) 설명

https://hello-stella.tistory.com/13: stft 설명


