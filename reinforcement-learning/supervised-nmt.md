# Supervised NMT

## Cross-entropy vs BLEU

$$
L= -\frac{1}{|Y|}\sum_{y \in Y}{P(y) \log P_\theta(y)}
$$

Cross entropy는 훌륭한 classification을 위해서 이미 훌륭한 loss function이지만 약간의 문제점을 가지고 있습니다. Seq2seq의 훈련 과정에 적용하게 되면, 그 자체의 특성으로 인해서 우리가 평가하는 BLEU와의 괴리가 생기게 됩니다. (자세한 내용은 이전 챕터 내용 참조) 따라서 어찌보면 우리가 원하는 번역 task의 objective와 다름으로 인해서 cross-entropy 자체에 over-fitting되는 효과가 생길 수 있습니다. 일반적으로 BLEU는 implicit evluation과 좋은 상관관계에 있다고 알려져 있지만, cross entropy는 이에 비해 낮은 상관관계를 가지기 때문입니다. 따라서 차라리 BLEU를 objective function으로 사용하게 된다면 더 좋은 결과를 얻을 수 있습니다. 마찬가지로 다른 NLP task의 문제에 대해서도 비슷한 접근을 생각 할 수 있습니다.

## Minimum Risk Training

위의 아이디어에서 출발한 논문[\[Shen at el.2015\]](https://arxiv.org/pdf/1512.02433.pdf)이 Minimum Risk Training이라는 방법을 제안하였습니다. 이때에는 Policy Gradient를 직접적으로 사용하진 않았지만, 거의 비슷한 수식이 유도 되었다는 점이 매우 인상적입니다.

$$
\begin{aligned}
\hat{\theta}_{MLE} &= argmax_\theta(\mathcal{L}(\theta)) \\
where~\mathcal{L}(\theta)&=\sum_{s=1}^S\log{P(y^{(s)}|x^{(s)};\theta)}
\end{aligned}
$$

기존의 Maximum Likelihood Estimation (MLE)방식은 위와 같은 손실 함수(Loss function)를 사용하여 $$ |S| $$개의 입력과 출력에 대해서 loss 값을 구하고, 이를 최대화 하는 $$ \theta $$를 찾는 것이 목표(objective)였습니다. 하지만 이 논문에서는 ***Risk***를 아래와 같이 정의하고, 이를 최소화 하는 학습 방식을 Minimum Risk Training (MRT)라고 하였습니다.

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

이렇게 정의된 risk를 최소화 하도록 하는 것이 objective입니다. 사실 risk 대신에 reward로 생각하면, reward를 최대화 하는 것이 objective가 됩니다. 결국은 risk를 최소화 할 때에는 gradient descent, reward를 최대화 할 때는 gradient ascent를 사용하게 되므로, 결국 완벽하게 같은 이야기라고 볼 수 있습니다. 따라서 실제 구현에 있어서는 $$ \triangle(y,y^{(s)}) $$ 사용을 위해서 BLEU 점수에 $$ -1 $$을 곱하여 사용 합니다.

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

하지만 주어진 입력에 대한 가능한 정답에 대한 전체 space를 search할 수는 없기 때문에, Monte Carlo를 사용하여 sampling하는 것을 택합니다. 그리고 위의 수식에서 $$ \theta $$에 대해서 미분을 수행합니다.

아래는 위와 같이 훈련한 MRT에 대한 성능을 실험한 결과 입니다. 기존의 MLE 방식에 비해서 BLEU가 상승한 것을 확인할 수 있습니다.

![](/assets/rl-minimum-risk-training.png)

아래에서 설명할 policy gradient 방식과의 **차이점은 sampling을 통해 기대값을 approximation 할 때에 $$ Q $$라는 값을 sampling한 값의 확률들에 대해서 normalize한 형태로 만들어 주었다**는 것 입니다.

### Implementation

### Code

MRT(or RL)을 PyTorch를 사용하여 구현 해 보도록 하겠습니다. 자세한 전체 코드는 이전의 NMT PyTorch 실습 코드의 git repository에서 다운로드 할 수 있습니다.

- git repo url: https://github.com/kh-kim/simple-nmt

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