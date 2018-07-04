# Characteristic of NLP's RL

우리는 이제까지 강화학습 중에서도 정책기반 학습 방식인 policy gradients에 대해서 간단히 다루어 보았습니다. 사실, policy gradients의 경우에도 소개한 방법 이외에도 발전된 방법들이 많이 있습니다. 예를 들어 Actor Critic의 경우에는 정책망($$\theta$$) 이외에도 가치 네트워크($$W$$)를 따로 두어, episode의 종료까지 기다리지 않고 online으로 학습이 가능합니다. 여기에서 더욱 발전하여 기존의 단점을 보완한 A3C와 같은 다양한 방법들이 존재 합니다.