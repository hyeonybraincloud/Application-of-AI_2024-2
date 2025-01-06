# CCTV 영상을 활용한 잡상인 탐지 AI 모델 설계
## 1. Introduction
기존의 CCTV 이미지를 활용한 객체 탐지는 주로 응급 상황을 감지하는 데에 국한되었다. 그래서 응급 상황에 비해 위험도가 현저히 낮은 지하철 내 잡상인의 경우, 그 탐지 대상에 포함되는 경우가 현저히 적었다. 그러나 현대 사회가 발전함에 따라 자연스레 대중교통의 이용도는 더욱 높아져가고, 그에 따라 대중교통에서의 공공질서 및 안전 확보에 대한 목소리가 커지면서 더 이상 열차 승객들을 방해하는 잡상인들을 좌시할 수 없는 시점이 점점 다가오고 있다. 현재 인력을 활용한 직접 단속과 방송을 통한 간접 단속은 있지만, 그외의 방법이 미비하므로, 이러한 문제에 대해 공학적으로 해결하고자 한다.

## 2. 전이 학습(Transfer Learning)
본 실험은 객체 탐지에서 우수한 성능을 보이는 YOLOv8n 모델에 잡상인 데이터셋을 전이 학습시킨 것을 기준으로 삼았다. 데이터셋은 과학기술정보통신부와 한국지능정보사회진흥원 AI Hub 플랫폼에서 제공하고 있는 ‘CCTV 추적 영상’(2024) 데이터(https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=160) 를 활용했다. 또한, 데이터셋은 train : validation : test = 6 : 2 : 2의 비율로 구성하였다. 이에 따라 기존 YOLOv8n 모델에 대해 데이터셋 6,033 장을 epochs을 10으로 하여 학습시켰다.

![image](https://github.com/user-attachments/assets/876f2c28-8649-4960-81e9-005f0b4b5703)

높은 성능을 보여 overfitting을 우려했으나, 다음 그래프에 의해 그렇지 않다는 것을 확인하였다. 
![image](https://github.com/user-attachments/assets/5d1f0f1d-dbca-4d7c-8c2b-1290ff0ef2dc)

여기서 실험을 마쳐도 좋으나, 궁극적으로 CCTV 등 저성능 컴퓨팅 환경까지 나아가는 것이 목표이기 때문에, 성능을 오차 범위 내에서 유지하되, CPU 및 GPU 사용량을 줄여 경량화를 할 필요가 있었다.

## 3. Ghost Convolution
최근 YOLO 모델에 Ghost Convolution 기능을 도입하여 경량화하는 방식에 대해 논문이 많이 나오고 있다. 소재에 따라 다르지만, 모델 크기, 파라미터 수, 계산량 등을 약 50 %에 근접하게 감소시키는데도 여전히 높은 정확도를 유지한다고 한다. 그렇다보니 해당 방식은 컴퓨팅 리소스가 제한적인 edge device 환경에서도 적합한 선택지인 것이다.

본 실험은 다음과 같이 구성하였다. YOLOv8n의 Backbone 구조에는 5개의 convolution layers가 포함되어 있다. 5개의 convolution layers 중 하나를 선택하여 ghost convolution 방식으로 변경한다. 이를 index 순대로 진행하여 어느 지점의 convolution layer을 ghost convolution으로 바꾸었을 때 가장 효과적인지를 판단한다.

![image](https://github.com/user-attachments/assets/3fb37b16-467a-483e-8c2c-786a11954cb3)

실험 대상 모델은 다음과 같이 정의한다. 예를 들어, i번째 standard convolution layer을 ghost convolution 방식으로 바꿨다면 Ghost[i] 모델이라 하겠다. 평가 방식은 다음과 같다.

Step 1: torch.rand로 16 개의 Sample(batch size 16), 3 개의 채널, 640×640 크기의 이미지를 무작위로 생성한다. 입력 데이터는 torch.rand로 생성된 0-1 사이의 값으로 구성되며, 이는 전이학습 때 사용된 YOLO 모델의 입력 형태와 일치한다.

Step 2: YOLO 모델의 입력값을 [0, 1] 범위로 정규화하여 학습 또는 추론에서의 데이터의 일관성을 유지한다.

Step 3: 프로파일링을 수행하여, CPU 및 GPU 리소스 사용량을 분석한다.

Step 4: 앞선 과정을 모델 다섯 가지(Ghost[0], Ghost[1], Ghost[2], Ghost[3], Ghost[4])에 대해 각각 다섯 번 수행 후 평균값을 도출한다. 실험 결과는 다음과 같다.

### 실험 결과 1
![image](https://github.com/user-attachments/assets/697ed9a3-bc41-4f94-bc44-7e751bf9255f)
Orig. 대비 CPU 및 GPU 사용량이 가장 많이 개선된 것은 Ghost[0]이다. Self CPU Time Total은 68.93%, Self CUDA Time Total은 23.96% 감소 되었다.

### 실험 결과 2
![image](https://github.com/user-attachments/assets/a55f26ae-9bf6-4175-8a95-d919f2e061aa)
Orig. 대비 CPU 및 GPU 사용량이 가장 많이 개선된 것은 Ghost[0]이다. Self CPU Time Total은 66.7%, Self CUDA Time Total은 20.96% 감소 되었다.

### 실험 결과 3
![image](https://github.com/user-attachments/assets/fe715f49-880b-4694-9dcb-70963b08eb0f)
Orig. 대비 CPU 및 GPU 사용량이 가장 많이 개선된 것은 Ghost[0]이다. Self CPU Time Total은 69.9%, Self CUDA Time Total은 23.67% 감소 되었다.

### 실험 결과 4
![image](https://github.com/user-attachments/assets/9d9685a3-36f9-49d4-96d4-688873df2202)
Orig. 대비 CPU 및 GPU 사용량이 가장 많이 개선된 것은 Ghost[0]이다. Self CPU Time Total은 66.36%, Self CUDA Time Total은 21.21% 감소 되었다.

### 실험 결과 5
![image](https://github.com/user-attachments/assets/0669800f-6b1c-44a1-990a-5182f935d8c2)
Orig. 대비 CPU 및 GPU 사용량이 가장 많이 개선된 것은 Ghost[0]이다. Self CPU Time Total은 66.19%, Self CUDA Time Total은 20.79% 감소 되었다.

다섯 번의 실험 결과를 통해 Orig. 대비 CPU 및 GPU 사용량이 가장 많이 감소한 것은 Ghost[0]라는 것을 알 수 있다. **Self CPU Time Total은 평균 67.068%**, **Self CUDA Time Total은 평균 22.118% 감소**하였다. Ghost Convolution layer로 기존의 Convolution layer를 2개 이상 대체하는 실험도 진행했으나 유의미한 결과는 나타나지 않았다.

따라서 0번째 Convolution layer을 Ghost convolution 방식으로 바꾼 YOLOv8n 모델에 대해 전이학습을 시킨 후, 성능을 검사하면 다음과 같다.

![image](https://github.com/user-attachments/assets/6f88e3ad-e193-480e-8b81-ba554005f032)

기존의 일부 layer가 Ghost Convolution으로 바뀌면서 Precision, Recall, mAP50, mAP50-95의 수치는 5% 이하로 감소했다. 정확도가 약간 낮아진 원인으로는 기존의 연산량을 줄임으로써 세밀한 특징이나 복잡한 환경을 탐지하기 어려웠을 가능성이 있고, Backbone의 첫 번째 Convolution layer에 적용되면서 저수준의 특징을 학습하는 과정에서 정보 손실이 발생했을 가능성이 있다. 그러나 여러 성능 지표가 5% 이하로 감소했어도 무시할만한 수치이고, 그에 대한 trade-off로 CPU 및 GPU의 사용량 감소를 얻었기 때문에 본 실험은 유의미하다.

## 4. Risk-Compensation

※ Transfer Learning과 Risk-Compensation은 다른 사람이 구현하였다. 본인은 Ghost Convolution 방식을 구현하였다.
