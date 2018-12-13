# Reinforcement Learning on Natural Language Generation

## How to Apply

강화학습은 Markov Decision Process (MDP)상에서 정의되고 동작합니다. 따라서 여러 선택(action)들을 통해서 여러 상황(state)들을 옮겨다니며(transition) 에피소드가 구성되고, action과 state에 따라서 보상(reward)가 주어지며 누적(cumulated)되어, episode가 종료되면 누적보상(cumulative reward)를 얻을 수 있습니다.

따라서 이를 자연어처리에 적용하게 되면 텍스트 분류(text classification)와 같은 문제에 적용되기 보단 자연어생성(natural language generation)에 적용되게 됩니다. 이제까지 생성된 단어들의 시퀀스가 현재의 상황(state)이 될 것이며, 이제까지 생성된 단어를 기반으로 새롭게 선택하는 단어가 action이 될 것 입니다. 이렇게 문장의 첫 단어(BOS, beginning of sentence)부터 문장의 끝 단어(EOS, end of sentence)까지 선택하는 과정(한 문장을 생성하는 과정)이 하나의 에피소드(episode)가 됩니다. 우리는 훈련 corpus에 대해서 문장을 생성하는 방법을 배워(즉, episode를 반복하여) 기대누적보상(expected cumulative reward)을 최대화(maximize)할 수 있도록 번역 신경망(강화학습에서는 정책망) $\theta$를 훈련 하는 것 입니다.

다시한번 정리를 위해, 기계번역에 강화학습을 대입시켜 보면, 현재의 상황(state)은 주어진 source 문장과 이전까지 생성(번역)된 단어들의 시퀀스가 될 것이고, action은 현재 state에 기반하여 새로운 단어를 선택 하는 것이 될 것 입니다. 그리고 action을 선택하게 되면 다음 state는 deterministic하게 정해지게 됩니다. action을 선택한 후에 환경(environment)으로부터 즉각적인(immediate) 보상(reward)를 받지는 않으며, 모든 단어의 선택이 끝나고(EOS를 선택) 에피소드(episode)가 종료되면 BLEU 점수를 통해 보상을 받을 수 있습니다. 즉, 종료 시에 받는 보상값은 에피소드 누적보상값(cumulative reward)과 일치 합니다.

강화학습을 통해 모델을 훈련할 때, 훈련의 처음부터 강화학습만 적용하기에는 그 훈련방식이 비효율적이고 어려움이 크기 때문에, 보통 기존의 maximum likelihood estimation (MLE)를 통해 어느정도 학습이 된 신경망 $\theta$에 강화학습을 적용 합니다. <comment> 강화학습은 탐험(exploration)을 통해 더 나은 정책의 가능성을 찾고, exploitation을 통해 그 정책을 발전시켜 나갑니다. </comment>

## Characteristics

우리는 이제까지 강화학습 중에서도 정책기반 학습 방식인 policy gradients에 대해서 간단히 다루어 보았습니다. 사실, policy gradients의 경우에도 소개한 방법 이외에도 발전된 방법들이 많이 있습니다. 예를 들어 Actor Critic의 경우에는 정책망($\theta$) 이외에도 가치 네트워크($W$)를 따로 두어, episode의 종료까지 기다리지 않고 online으로 학습이 가능합니다. 여기에서 더욱 발전하여 기존의 단점을 보완한 A3C와 같은 다양한 방법들이 존재 합니다. <comment> [Reinforcement Learning: An Introduction[Sutton et al.2017]](http://www.incompleteideas.net/book/bookdraft2017nov5.pdf)을 참고하세요. </comment>

하지만, 자연어처리에서의 강화학습은 이런 다양한 방법들을 굳이 사용하기보다는 간단한 REINFORCE with baseline를 사용하더라도 큰 문제가 없습니다. 이것은 자연어처리 분야의 특성에서 기인합니다. 강화학습을 자연어처리에 적용할 때는 아래와 같은 특성들이 있습니다.

1. 매우 많은 action들이 존재 합니다. 보통 다음 단어를 선택하는 것이 action이 되기 때문에, 선택 가능한 action set의 크기는 어휘(vocabularty) 사전의 크기와 같다고 볼 수 있습니다. 따라서 action set의 사이즈는 보통 몇만개가 되기 마련입니다.
1. 매우 많은 state들이 존재 합니다. 단어를 선택하는 것이 action이었다면, 이제까지 선택된 단어들의 시퀀스는 state가 되기 때문에, 여러 time-step을 거쳐 수많은 action(단어)들이 선택되었다면, 가능한 state의 경우의 수는 매우 커질 것 입니다. <comment>  $|state|=|action|^n$ </comment>
1. 매우 많은 action들과 매우 많은 state들을 훈련 과정에서 모두 겪어보고 만나보는 것은 거의 불가능하다고 볼 수 있습니다. 또한, 추론(inference)과정에서 보지 못한(unseen) 샘플을 만나는 것은 매우 당연한 일이 될 것 입니다. 따라서 이러한 희소성(sparseness)문제는 큰 골칫거리가 될 수 있습니다. 하지만 우리는 신경망(neural network)과 딥러닝을 통해서 이 문제를 해결 할 수 있습니다. <comment> 이것은 Deep Reinforcement Learning이 큰 성공을 거두고 있는 이유이기도 합니다. </comment>
1. 강화학습을 자연어처리에 적용할 때에 쉬운 점도 있습니다. 대부분 하나의 문장을 생성하는 것이 하나의 에피소드(episode)가 되는데, 보통 문장의 길이는 길어봤자 80단어도 되기 힘들다는 것 입니다. 따라서 다른 분야의 강화학습보다 훨씬 쉬운 이점을 가지게 됩니다. 예를 들어 딥마인드(DeepMind)의 바둑(AlphaGo)이나 스타크래프트의 경우에는 하나의 에피소드가 끝나기까지 매우 긴 시간이 흐르게 됩니다. 따라서 에피소드 내에서 선택되었던 action들이 update되기 위해서는 매우 긴 에피소드가 끝나기를 기다려야 할 뿐만 아니라, 10분 전에 선택 했던 action이 이 게임(game)의 승패에 얼마나 큰 영향을 미쳤는지 알아내는 것은 매우 어려운 일이 될 것 입니다. 따라서, 자연어처리 분야가 다른 분야에 비해서 에피소드가 짧은 것은 매우 큰 이점으로 작용하여 정책망(policy network)을 훨씬 더 쉽게 훈련(training)시킬 수 있게 됩니다.
1. 대신에 문장 단위의 에피소드를 가진 강화학습에서는 보통 에피소드 중간에 reward를 얻기는 힘듭니다. 예를 들어 번역의 경우에는 각 time-step 마다 단어를 선택할 때 즉각(immediate) reward를 얻지 못하고, 번역이 모두 끝난 이후 완성된 문장과 정답(reference) 문장을 비교하여 BLEU 점수를 누적 reward로 사용하게 됩니다. 마찬가지로 에피소드가 매우 길다면 이것은 매우 큰 문제가 되었겠지만, 다행히도 문장 단위의 에피소드 상에서는 큰 문제가 되지 않습니다.

이제 우리는 위의 특성들을 활용하여 강화학습을 성공적으로 기계번역에 적용한 방법들과 사례들을 살펴보고 실습 해 보도록 하겠습니다.