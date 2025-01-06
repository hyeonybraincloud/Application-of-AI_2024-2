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
인공신경망의 각 층에서 데이터를 처리하는 과정에서 발생하는 데이터 왜곡, 손실 등(이하 '리스크')은 모델의 학습 안정성을 저하시킨다. 따라서 탐지의 정확도를 높이고, 보다 안정적인 모델을 제작하기 위해, 리스크를 분석하고 조정하는 방법을 도입할 필요가 있다. 이에 대한 방안으로, 본 연구에서는 '리스크 항상성 이론'에 기반한 최적화 과정을 고안하였다. **리스크 항상성 이론**이란 인간이 특정 활동에서 느끼는 위험 수준을 일정하게 유지하려는 경향이 있음을 주장하는 이론이다. 즉, 현재 활동에서 느끼는 위험이 목표 수준보다 높거나 낮을 경우, 목표 수준으로 회귀하려는 경향을 보이므로, 순간을 기준으로 하였을 때, (리스크)x(기댓값)이 일정해야 한다는 것이다.

이를 객체 탐지 모델에 대해 적용하면, '데이터에 대한 분산'을 '리스크'로, '얻을 수 있는 정보의 크기'는 '기댓값'으로 정의하였다. 이에 따라 모델을 구성하는 layer의 (분산)x(정보량) 값이 일정하도록 설계하여 모델의 성능을 높이고자 한다. 리스크와 기댓값을 layer 별로 적용하기 위해 파스칼 삼각형을 활용했다. 파스칼 삼각형은 층별로 갈래 수와 깊이를 고려하기 때문에 데이터 간의 불균등에 따른 현실적인 리스크 평가가 가능하게 하고, 다음 식과 같이 한 번에 나오는 갈래의 수가 깊이와 제곱비례 관계일 때, 갈래 수가 분산에 선형적으로 기여해 리스크를 효율적으로 계산할 수 있도록 만든다.

$$\frac{\ln{(\lVert norm \rVert^2)}}{\ln b}=b^2
$$

$$f(b)\triangleq \vert \frac{\ln{(\lVert norm \rVert^2)}}{\ln b} - b^2 \vert
$$

따라서 기댓값을 다음과 같이 정의할 수 있다.

$$d=\log_b{Norm^2}
$$

$$Risk \times Expected Value = \frac{d \cdot (b-1)}{b^2} \times \log_2{(\lVert norm \rVert^2)}
$$

다음은 전이 학습한 YOLOv8n 모델의 각 layer 별 (분산)x(정보량)이며, layer마다 그 값이 다른 것을 알 수 있다. 본 실험에서는 그 값을 일정하게 하기 위해 함수를 통해 가중치를 조정하였다. 

![image](https://github.com/user-attachments/assets/f8e705b3-ac62-4307-8ad5-ae2f1b604242)

은닉층 중에서 학습에 따라 가중치가 변하는 layer을 대상으로 함수가 적용되어 (분산)X(정보량)을 일정하게 만다는 작업을 진행했다. 그 과정에서 모든 층의 (분산)X(정보량)의 평균을 계산한 다음, 평균으로부터 20 % 이상의 차이가 나는 layers는 파레토 법칙에 기해 적용 대상에서 배제하였다. 이후 잔여 layers의 (분산)X(정보량)의 평균을 다시 계산하고, 그 평균을 기준으로 각 layer의 (분산)X(정보량)을 일정하게 유지하도록 가중치를 변경시켰다.

이러한 방식으로 해당 모델은 각 layer에서 발생하는 리스크를 효율적으로 조정하여 데이터의 왜곡 및 손실을 최소화할 수 있다. 또한, Ghost Convolution에 따른 모델의 정확도를 보완할 수 있으며, 모델의 안정성을 향상시켜 다양항 컴퓨팅 환경에서도 안정적으로 동작하다록 할 수 있다. 

## 5. Flow Chart 
![image](https://github.com/user-attachments/assets/5c91e073-811c-4dd9-9d52-bf2e9d62e87d)

## 6. Result
### 6.1 실험 모델의 최종 성능 지표 비교
![image](https://github.com/user-attachments/assets/1941ebdb-6297-4520-a852-7071f4ba3b17)

### 6.2 Test 데이터셋을 통한 최종 성능 지표 비교
![image](https://github.com/user-attachments/assets/8d674ac7-6fe0-4e62-8744-1ebf4e2de4e1)

### 6.3 Application
![image](https://github.com/user-attachments/assets/c20dc5c3-4b1c-4687-aa33-db4821c93993)

※ 'Merchant'라는 글자가 작아서 안 보일 수 있음. 캐리어를 끌고 가는 사람이 Merchant로 올바르게 묶임.

## 7. Consideration
### 7.1 데이터의 다양성 부족
본 실험에서 쓰인 데이터셋은 실제 지하철이나 지하철 역사의 잡상인을 촬영한 것이 아니라, 데이터 제공 측에서 인위적으로 만든 것이다. 그래서 제공한 측에서 만든 데이터셋에 대해 적용을 하면 높은 성능을 보이지만, 실제 잡상인 사진에 적용하면 성능이 많이 떨어진다. 앞으로 이 프로젝트가 궁극적으로 나아가야 할 방향은, 실제 잡상인의 모습이 담긴 데이터셋까지 학습시켜서 그 범용성을 넓히는 것이다.

### 7.2 Risk Compensation 및 Ghost Convolution의 순서
Risk Compensation은 이미 전이학습이 마쳐진 모델을 더욱 완벽하게 하기 위함이다. 그래서 본질적으로 기존 YOLOv8n에 대해 Ghost Convolutionization을 하고, 전이 학습까지 마친 다음에 Risk-Compensation을 적용해야 한다. 그러나 처음 Risk-Compensation을 설계 및 실험하면서 그 대상을 Ghost Convolutionization하지 않은 모델로 하여 전체 실험의 완전성에 약간의 결여가 있어 아쉬움이 있다. 그래도 기존 5% 이하의 손실을 약 2.5% 이하의 손실로 줄였기 때문에, 본 실험에서 적용한 순서가 완전히 무의미한 것은 아니다.

※ Transfer Learning과 Risk-Compensation은 다른 사람이 구현하였다. 본인은 Ghost Convolution 방식을 구현하였다.
