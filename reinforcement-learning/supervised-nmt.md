# Supervised NMT

## Cross-entropy vs BLEU

$$
L= -\frac{1}{|Y|}\sum_{y \in Y}{P(y) \log P_\theta(y)}
$$

Cross entropy는 훌륭한 분류(classification) 문제에서 이미 훌륭한 손실함수(loss function)이지만 약간의 문제점을 가지고 있습니다. 자연어생성(NLG)을 위한 sequence-to-sequence의 훈련 과정에 적용하게 되면, 그 자체의 특성으로 인해서 우리가 평가하는 BLEU와의 괴리(discrepancy)가 생기게 됩니다. (자세한 내용은 이전 챕터 내용 참조 바랍니다.) 따라서 어찌보면 우리가 원하는 실제 기계번역의 목표(objective)와 다름으로 인해서 cross-entropy 자체에 오버피팅(over-fitting) 되는 효과가 생길 수 있습니다. 일반적으로 BLEU는 human evluation과 좋은 상관관계에 있다고 알려져 있지만, cross entropy는 이에 비해 낮은 상관관계를 가지기 때문입니다. 따라서 차라리 BLEU를 훈련 과정의 목적함수(objective function)로 사용하게 된다면 더 좋은 결과를 얻을 수 있을 것 입니다. 마찬가지로 다른 NLG 문제(요약 및 챗봇 등)에 대해서도 비슷한 접근을 생각 할 수 있습니다.

## Minimum Risk Training

위의 아이디어에서 출발한 논문[\[Shen at el.2015\]](https://arxiv.org/pdf/1512.02433.pdf)이 Minimum Risk Training이라는 방법을 제안하였습니다. 이때에는 Policy Gradient를 직접적으로 사용하진 않았지만, 거의 비슷한 수식이 유도 되었다는 점이 매우 인상적입니다.

$$
\begin{aligned}
\hat{\theta}_{MLE} &= argmin_\theta(\mathcal{L}(\theta)) \\
where~\mathcal{L}(\theta)&=-\sum_{s=1}^S\log{P(y^{(s)}|x^{(s)};\theta)}
\end{aligned}
$$

기존의 Maximum Likelihood Estimation (MLE)방식은 위와 같은 손실 함수(Loss function)를 사용하여 $$ |S| $$개의 입력과 출력에 대해서 손실(loss)값을 구하고, 이를 최소화 하는 $$ \theta $$를 찾는 것이 목표(objective)였습니다. 하지만 이 논문에서는 ***Risk***를 아래와 같이 정의하고, 이를 최소화 하는 학습 방식을 Minimum Risk Training (MRT)라고 하였습니다.

$$
\begin{aligned}
\mathcal{R}(\theta)&=\sum_{s=1}^S E_{y|x^{(s)};\theta}[\triangle(y,y^{(s)})] \\
&=\sum_{s=1}^S \sum_{y \in \mathcal{Y(x^{(s)})}}{P(y|x^{(s)};\theta) \triangle(y, y^{(s)})}
\end{aligned}
$$

위의 수식에서 $$ \mathcal{Y}(x^{(s)}) $$는 full search space로써, $$ s $$번째 입력 $$ x^{(s)} $$가 주어졌을 때, 가능한 정답의 집합을 의미합니다. 또한 $$ \triangle(y,y^{(s)}) $$는 입력과 파라미터($$ \theta $$)가 주어졌을 때, sampling한 $$ y $$와 실제 정답 $$ y^{(s)} $$의 차이(error)값을 나타냅니다. 즉, 위 수식에 따르면 risk $$ \mathcal{R} $$은 주어진 입력과 현재 파라미터 상에서 얻은 y를 통해 현재 모델(함수)을 구하고, 동시에 이를 사용하여 risk의 기대값을 구한다고 볼 수 있습니다.

$$
\hat{\theta}_{MRT}=argmin_\theta(\mathcal{R}(\theta))
$$

이렇게 정의된 risk를 최소화(minimize) 하도록 하는 것이 목표(objective)입니다. 사실 risk 대신에 reward로 생각하면, reward를 최대화(maximize) 하는 것이 목표가 됩니다. 결국은 risk를 최소화 할 때에는 gradient descent, reward를 최대화 할 때는 gradient ascent를 사용하게 되므로, 수식을 풀어보면 결국 완벽하게 같은 이야기라고 볼 수 있습니다. 따라서 실제 구현에 있어서는 $$ \triangle(y,y^{(s)}) $$ 사용을 위해서 BLEU 점수에 $$ -1 $$을 곱하여 사용 합니다.

$$
\begin{aligned}
\tilde{\mathcal{R}}(\theta)&=\sum_{s=1}^S{E_{y|x^{(s)};\theta,\alpha}[\triangle(y,y^{(s)})]} \\
&=\sum_{s=1}^S \sum_{y \in \mathcal{S}(x^{(s)})}{Q(y|x^{(s)};\theta,\alpha)\triangle(y,y^{(s)})}
\end{aligned}
$$
$$
\begin{aligned}
where~\mathcal{S}(x^{(s)})~is~a~sampled~subset~of~the~full~search~space~\mathcal{y}(x^{(s)}) \\
and~Q(y|x^{(s)};\theta,\alpha)~is~a~distribution~defined~on~the~subspace~S(x^{(s)}):
\end{aligned}
$$

$$
Q(y|x^{(s)};\theta,\alpha)=\frac{P(y|x^{(s)};\theta)^\alpha}{\sum_{y' \in S(x^{(s)})}P(y'|x^{(s)};\theta)^\alpha}
$$

하지만 주어진 입력에 대한 가능한 정답에 대한 전체 space를 탐색(search)할 수는 없기 때문에, Monte Carlo를 사용하여 서브스페이스(sub-space)를 샘플링(sampling) 하는 것을 택합니다. 그리고 위의 수식에서 $$ \theta $$에 대해서 미분을 수행합니다. 미분을 하여 얻은 수식은 아래와 같습니다.

$$
\begin{aligned}
\frac{\partial\tilde{R}(\theta)}{\partial\theta_i}&=\alpha\sum_{s=1}^{S}{\mathbb{E}_{y|x^{(s)};\theta,\alpha}[\frac{\partial P(y|x^{(s)};\theta)}{\partial\theta_i P(y|x^{(s)};\theta)}\times(\triangle(y,y^{(s)})-\mathbb{E}_{y'|x^{(s)};\theta,\alpha}[\triangle(y',y^{(s)})])]} \\
&=\alpha\sum_{s=1}^{S}{\mathbb{E}_{y|x^{(s)};\theta,\alpha}[\frac{\partial \log{P(y|x^{(s)};\theta)}}{\partial\theta_i}\times(\triangle(y,y^{(s)})-\mathbb{E}_{y'|x^{(s)};\theta,\alpha}[\triangle(y',y^{(s)})])]} \\
&\approx \alpha \sum_{s=1}^{S}{\frac{\partial \log{P(y|x^{(s)};\theta)}}{\partial\theta_i}\times(\triangle(y,y^{(s)})-\frac{1}{K}\sum_{k=1}^{K}{\triangle(y^{(k)},y^{(s)})})}
\end{aligned}
$$

$$
\theta_{i+1} \leftarrow \theta_i - \frac{\partial\tilde{R}(\theta)}{\partial\theta_i}
$$

이제 미분을 통해 얻은 MRT의 최종 수식을 해석 해 보겠습니다. 이해가 어렵다면 아래의 policy gradients 수식과 비교하며 따라가면 좀 더 이해가 수월할 수 있습니다.

- $$s$$번째 입력 $$x^{(s)}$$를 신경망 $$\theta$$에 넣어 얻은 로그확률 $$\log{P(y|x^{(s)};\theta)}$$을 미분하여 gradient를 얻습니다.
- 그리고 $$\theta$$로부터 샘플링(samping) 한 $$y$$와 실제 정답 $$y^{(s)}$$와의 차이(여기서는 주로 BLEU에 $$-1$$을 곱하여 사용)값에서 
- 또 다시 $$\theta$$로부터 샘플링하여 얻은 $$y'$$와 실제 정답 $$y^{(s)}$$와의 차이(마찬가지로 -BLEU)의 기대값을
- 빼 준 값을 risk로써 로그확률의 gradient에 곱해 줍니다.
- 이 과정을 전체 데이터셋(실제로는 mini-batch) $$S$$에 대해서 수행한 후 합(summation)을 구하고 learning rate $$\alpha$$를 곱 합니다.

최종적으로는 기대값 수식을 monte carlo sampling을 통해 제거할 수 있습니다.

아래는 policy gradients 수식 입니다.

$$
\begin{aligned}
\triangledown_\theta J(\theta)&=\mathbb{E}_{\pi_\theta}[\triangledown_\theta \log{\pi_\theta (a|s)} \times Q^{\pi_\theta}(s,a)] \\
\theta &\leftarrow \theta + \alpha \triangledown_\theta J(\theta)
\end{aligned}
$$

MRT는 risk에 대해 minimize 해야 하기 때문에 gradient descent를 해 주는 것을 제외하면 똑같은 수식이 나오는 것을 알 수 있습니다. 

![](/assets/rl-minimum-risk-training.png)

위와 같이 훈련한 MRT에 대한 성능을 실험한 결과 입니다. 기존의 MLE 방식에 비해서 BLEU가 1.5가량 상승한 것을 확인할 수 있습니다. 이처럼 MRT는 강화학습으로써의 접근을 전혀 하지 않고도, 수식적으로 policy gradients의 일종인 REINFORCE with baseline 수식을 이끌어내고 성능을 끌어올리는 방법을 제시한 점이 인상깊습니다.

### Implementation

우리는 아래의 방법을 통해 Minimum Risk Training을 PyTorch로 구현 할 겁니다. 

1. 먼저 BLEU를 통해 얻은 reward에 $$-1$$을 곱해주어 risk로 변환 합니다. 
1. 그리고 로그 확률에 risk를 곱해주고, 기존에 Negative Log Likelihodd Loss (NLLLoss)를 사용했으므로 NLLLoss 값에 $$-1$$을 곱해주어 sum of positive log probability를 구합니다. 
1. Summation 결과물에 대해서 $$\theta$$에 대해 미분을 수행하면, back-propagation을 통해서 신경망 $$\theta$$ 전체에 gradient가 구해집니다. 
1. 이 gradient를 사용하여 gradient descent를 통해 최적화(optimize) 하도록 할 겁니다.

$$
\nabla_\theta J(\theta) = \nabla_\theta\sum_{s=1}^{S}{\bigg( \log{P(y|x^{(s)};\theta)}\times\Big(\triangle(y,y^{(s)})-\frac{1}{K}\sum_{k=1}^{K}{\triangle(y^{(k)},y^{(s)})}\Big)\bigg)}
$$

$$
\theta \leftarrow \theta - \lambda\nabla_\theta J(\theta)
$$

$$
where~\triangle(\hat{y}, y)=-BLEU(\hat{y}, y)
$$

### Code

MRT(or RL)을 PyTorch를 사용하여 구현 해 보도록 하겠습니다. 자세한 전체 코드는 이전의 NMT PyTorch 실습 코드의 git repository에서 다운로드 할 수 있습니다.

- git repo url: https://github.com/kh-kim/simple-nmt

#### train.py

```python
def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('-model', required = True)
    p.add_argument('-train', required = True)
    p.add_argument('-valid', required = True)
    p.add_argument('-lang', required = True)
    p.add_argument('-gpu_id', type = int, default = -1)

    p.add_argument('-batch_size', type = int, default = 32)
    p.add_argument('-n_epochs', type = int, default = 18)
    p.add_argument('-print_every', type = int, default = 50)
    p.add_argument('-early_stop', type = int, default = -1)

    p.add_argument('-max_length', type = int, default = 80)
    p.add_argument('-dropout', type = float, default = .2)
    p.add_argument('-word_vec_dim', type = int, default = 512)
    p.add_argument('-hidden_size', type = int, default = 768)
    p.add_argument('-n_layers', type = int, default = 4)   
    
    p.add_argument('-max_grad_norm', type = float, default = 5.)
    p.add_argument('-adam', action = 'store_true', help = 'Use Adam instead of using SGD.')
    p.add_argument('-lr', type = float, default = 1.)
    p.add_argument('-min_lr', type = float, default = .000001)
    p.add_argument('-lr_decay_start_at', type = int, default = 10, help = 'Start learning rate decay from this epoch.')
    p.add_argument('-lr_slow_decay', action = 'store_true', help = 'Decay learning rate only if there is no improvement on last epoch.')
    p.add_argument('-lr_decay_rate', type = float, default = .5)

    p.add_argument('-rl_lr', type = float, default = .01)
    p.add_argument('-n_samples', type = int, default = 1)
    p.add_argument('-rl_n_epochs', type = int, default = 0)
    p.add_argument('-rl_ratio_per_epoch', type = float, default = 1.)

    config = p.parse_args()

    return config
```

#### simple_nmt/rl_trainer.py

```python
import time
import numpy as np
#from nltk.translate.bleu_score import sentence_bleu as score_func
from nltk.translate.gleu_score import sentence_gleu as score_func

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils

import utils
import data_loader
```

```python
def get_reward(y, y_hat):
    # |y| = (batch_size, length1)
    # |y_hat| = (batch_size, length2)

    scores = []

    for b in range(y.size(0)):
        ref = []
        hyp = []
        for t in range(y.size(1)):
            ref += [str(int(y[b, t]))]
            if y[b, t] == data_loader.EOS:
                break

        for t in range(y_hat.size(1)):
            hyp += [str(int(y_hat[b, t]))]
            if y_hat[b, t] == data_loader.EOS:
                break

        # for nltk.bleu & nltk.gleu
        scores += [score_func([ref], hyp) * 100.]
    scores = torch.FloatTensor(scores).to(y.device)
    # |scores| = (batch_size)

    return scores
```

```python
def get_gradient(y, y_hat, criterion, reward = 1):
    # |y| = (batch_size, length)
    # |y_hat| = (batch_size, length, output_size)
    # |reward| = (batch_size)
    batch_size = y.size(0)

    # Before we get the gradient, multiply -reward for each sample and each time-step.
    y_hat = y_hat * -reward.view(-1, 1, 1).expand(*y_hat.size())

    # Again, multiply -1 because criterion is NLLLoss.
    log_prob = -criterion(y_hat.contiguous().view(-1, y_hat.size(-1)), y.contiguous().view(-1))
    log_prob.div(batch_size).backward()

    return log_prob
```

```python
def train_epoch(model, criterion, train_iter, valid_iter, config, start_epoch = 1, others_to_save = None):
    current_lr = config.rl_lr

    highest_valid_bleu = -np.inf
    no_improve_cnt = 0

    # Print initial valid BLEU before we start RL.
    model.eval()
    total_reward, sample_cnt = 0, 0
    for batch_index, batch in enumerate(valid_iter):
        current_batch_word_cnt = torch.sum(batch.tgt[1])
        x = batch.src
        y = batch.tgt[0][:, 1:]
        batch_size = y.size(0)
        # |x| = (batch_size, length)
        # |y| = (batch_size, length)

        # feed-forward
        y_hat, indice = model.search(x, is_greedy = True, max_length = config.max_length)
        # |y_hat| = (batch_size, length, output_size)
        # |indice| = (batch_size, length)

        reward = get_reward(y, indice)

        total_reward += float(reward.sum())
        sample_cnt += batch_size
        if sample_cnt >= len(valid_iter.dataset.examples):
            break
    avg_bleu = total_reward / sample_cnt
    print("initial valid BLEU: %.4f" % avg_bleu)
    model.train()

    # Start RL
    for epoch in range(start_epoch, config.rl_n_epochs + 1):
        #optimizer = optim.Adam(model.parameters(), lr = current_lr)
        optimizer = optim.SGD(model.parameters(), lr = current_lr)
        print("current learning rate: %f" % current_lr)
        print(optimizer)

        sample_cnt = 0
        total_loss, total_bleu, total_sample_count, total_word_count, total_parameter_norm, total_grad_norm = 0, 0, 0, 0, 0, 0
        start_time = time.time()
        train_bleu = np.inf

        for batch_index, batch in enumerate(train_iter):
            optimizer.zero_grad()

            current_batch_word_cnt = torch.sum(batch.tgt[1])
            x = batch.src
            y = batch.tgt[0][:, 1:]
            batch_size = y.size(0)
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            # feed-forward
            y_hat, indice = model.search(x, is_greedy = False, max_length = config.max_length)
            q_actor = get_reward(y, indice)
            # |y_hat| = (batch_size, length, output_size)
            # |indice| = (batch_size, length)
            # |q_actor| = (batch_size)

            baseline = []
            with torch.no_grad():
                for i in range(config.n_samples):
                    _, sampled_indice = model.search(x, is_greedy = False, max_length = config.max_length)
                    baseline += [get_reward(y, sampled_indice)]
                baseline = torch.stack(baseline).sum(dim = 0).div(config.n_samples)
                # |baseline| = (n_samples, batch_size) --> (batch_size)

            # calcuate gradients with back-propagation
            tmp_reward = q_actor - baseline
            # |tmp_reward| = (batch_size)
            get_gradient(indice, y_hat, criterion, reward = tmp_reward)

            # simple math to show stats
            total_loss += float(tmp_reward.sum())
            total_bleu += float(q_actor.sum())
            total_sample_count += batch_size
            total_word_count += int(current_batch_word_cnt)
            total_parameter_norm += float(utils.get_parameter_norm(model.parameters()))
            total_grad_norm += float(utils.get_grad_norm(model.parameters()))

            if (batch_index + 1) % config.print_every == 0:
                avg_loss = total_loss / total_sample_count
                avg_bleu = total_bleu / total_sample_count
                avg_parameter_norm = total_parameter_norm / config.print_every
                avg_grad_norm = total_grad_norm / config.print_every
                elapsed_time = time.time() - start_time

                print("epoch: %d batch: %d/%d\t|param|: %.2f\t|g_param|: %.2f\trwd: %.4f\tBLEU: %.4f\t%5d words/s %3d secs" % (epoch, 
                                                                                                            batch_index + 1, 
                                                                                                            int(len(train_iter.dataset.examples) // config.batch_size), 
                                                                                                            avg_parameter_norm, 
                                                                                                            avg_grad_norm, 
                                                                                                            avg_loss,
                                                                                                            avg_bleu,
                                                                                                            total_word_count // elapsed_time,
                                                                                                            elapsed_time
                                                                                                            ))

                total_loss, total_bleu, total_sample_count, total_word_count, total_parameter_norm, total_grad_norm = 0, 0, 0, 0, 0, 0
                start_time = time.time()

                train_bleu = avg_bleu

            # Another important line in this method.
            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            # Take a step of gradient descent.
            optimizer.step()

            sample_cnt += batch_size
            if sample_cnt >= len(train_iter.dataset.examples) * config.rl_ratio_per_epoch:
                break

        sample_cnt = 0
        total_reward = 0

        with torch.no_grad():
            model.eval()

            for batch_index, batch in enumerate(valid_iter):
                current_batch_word_cnt = torch.sum(batch.tgt[1])
                x = batch.src
                y = batch.tgt[0][:, 1:]
                batch_size = y.size(0)
                # |x| = (batch_size, length)
                # |y| = (batch_size, length)

                # feed-forward
                y_hat, indice = model.search(x, is_greedy = True, max_length = config.max_length)
                # |y_hat| = (batch_size, length, output_size)
                # |indice| = (batch_size, length)

                reward = get_reward(y, indice)

                total_reward += float(reward.sum())
                sample_cnt += batch_size
                if sample_cnt >= len(valid_iter.dataset.examples):
                    break

            avg_bleu = total_reward / sample_cnt
            print("valid BLEU: %.4f" % avg_bleu)

            if highest_valid_bleu < avg_bleu:
                highest_valid_bleu = avg_bleu
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1

            model.train()

        model_fn = config.model.split(".")
        model_fn = model_fn[:-1] + ["%02d" % (config.n_epochs + epoch), "%.2f-%.4f" % (train_bleu, avg_bleu)] + [model_fn[-1]]

        # PyTorch provides efficient method for save and load model, which uses python pickle.
        to_save = {"model": model.state_dict(),
                    "config": config,
                    "epoch": config.n_epochs + epoch + 1,
                    "current_lr": current_lr
                    }
        if others_to_save is not None:
            for k, v in others_to_save.items():
                to_save[k] = v
        torch.save(to_save, '.'.join(model_fn))

        if config.early_stop > 0 and no_improve_cnt > config.early_stop:
            break
```