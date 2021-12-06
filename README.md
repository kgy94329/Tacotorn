# Tacotorn
구글의 TTS모델인 Tacotron입니다.  
이 프로젝트는 TTS모델과 Tactron의 구조를 파악하고 직접 학습시켜보면서 데이터의 전처리부터 아웃풋까지의 과정을 확인합니다.  

# 구성
* model  
  * tacotron.py : 타코트론 모델입니다. End-to-End모델로 텍스트처리에서 음성 생성까지 모든 처리를 담당합니다.   

* text : 데이터셋의 텍스트파일을 전처리하는 모듈을 모아둔 디렉터리입니다.  
  * numbers.py :  
    * 텍스트파일의 숫자들을 문자열로 변경해주는 모듈입니다.  
  * symbols.py :  
    * 특수문자와 알파벳 처리를 담당하는 모듈입니다.
  * text_cleaner.py :  
    * 텍스트 파일 전처리 과정의 컨트롤러 역할을 합니다. 지정된 언어에 맞게 데이터를 전처리합니다.  

* util : 타코트론 기동에 필요한 나머지 모듈을 모아둔 디렉터리입니다.  
  * hparams.py :  
    * 타코트론 기동시 필요한 파라미터들의 값을 지정하는 모듈입니다. 각 처리별 모듈에서 직접 파라미터를 조정할 필요 없이 이 모듈에서 한번에 조정이 가능합니다.  
  * plot_alignment.py :  
    * 타코트론 학습 후 출력되는 그래프를 생성하는 모듈입니다.  

PreProcess.py :  
  *타코트론 모델을 학습시키기 위한 데이터셋을 전처리하는 모듈입니다. 이 모듈에서 클리닝된 데이터셋으로 tacotron.py모듈에서 학습합니다. 

train.py :  
  * 타코트론 학습의 엔트리 포인트가 되는 모듈입니다. PreProxess.py모듈에서 데이터셋을 클렌징한 후 이 모듈을 실행하면 학습이 시작됩니다. 학습시에는 500steps마다 weight값들이 저장되고, 이 저장된 값들은 save point역할을 합니다.  

test.py :  
  * 학습이 완료된 타코트론에 텍스트를 입력하여 음성 파일을 생성하는 모듈입니다. 

# 데이터셋  
* 영어 음성 데이터셋  
  * LJSpeech-1.1  
* 한국어 음성 데이터셋  
  * KSS
* 일본어 음성 데이터셋  
  * JSS

# 참조  
https://github.com/Kyubyong/tacotron  
https://github.com/keithito/tacotron  
https://github.com/chldkato/Tacotron-Korean-Tensorflow2  
https://github.com/NVIDIA/tacotron2
