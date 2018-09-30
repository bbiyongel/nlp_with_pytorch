# Segmentation using Subword (Byte Pair Encoding, BPE)

Byte Pair Encoding (BPE) 알고리즘을 통한 subword segmentation은 [[Sennrich at el.2015]](https://arxiv.org/pdf/1508.07909.pdf)에서 처음 제안되었고, 현재는 주류 전처리 방법으로 자리잡아 사용되고 있습니다.

Subword segmenation은 기본적으로 단어는 여러 sub-word들의 조합으로 이루어진다는 가정 하에 적용 되는 알고리즘입니다. 실제로 영어나 한국어의 경우에도 latin어와 한자를 기반으로 형성 된 언어이기 때문에, 많은 단어들이 sub-word들로 구성되어 있습니다. 따라서 적절한 subword detection을 통해서 subword 단위로 쪼개어 주면 어휘수를 줄일 수 있고 sparsity를 효과적으로 감소시킬 수 있습니다.

|언어|단어|조합|
|-|-|-|
|영어|Concentrate|con(=together) + centr(=center) + ate(=make)|
|한국어|집중(集中)|集(모을 집) + 中(가운데 중)|

Sparsity 감소 이외에도, 가장 대표적인 subword segmentation으로 얻는 효과는 unknown(UNK) token에 대한 효율적인 대처 입니다. Natrual Language Generation (NLG)를 포함한 대부분의 딥러닝 NLP 알고리즘들은 문장을 입력으로 받을 때 단순히 word sequence로써 받아들이게 됩니다. 따라서 UNK가 나타나게 되면 이후의 language model의 확률은 매우 망가져버리게 됩니다. 따라서 적절한 문장의 encoding또는 generation이 어렵습니다. -- 특히 문장 generation의 경우에는 이전 단어를 기반으로 다음 단어를 예측하기 때문에 더더욱 어렵습니다.

하지만 subword tokenization을 통해서 신조어나 typo(오타) 같은 UNK에 대해서 subword나 character 단위로 쪼개줌으로써 known token들의 조합으로 만들어버릴 수 있습니다. 이로써, UNK 자체를 없앰으로서 효율적으로 UNK에 대처할 수 있고, NLP 결과물의 품질을 향상시킬 수 있습니다.

## Example

아래와 같이 BPE를 적용하면 원래의 띄어쓰기 공백 이외에도 BPE의 적용으로 인한 공백이 추가됩니다. 따라서 원래의 띄어쓰기와 BPE로 인한 띄어쓰기를 구분해야 할 필요성이 있습니다. 그래서 특수문자(기존의 _가 아닌 ▁)를 사용하여 기존의 띄어쓰기(또는 단어의 시작)를 나타냅니다. 이를 통해 후에 설명할 detokenization에서 문장을 BPE이전의 형태로 원상복구 할 수 있습니다.

한글 Mecab에 의해 tokenization 된 원문
___
현재 TED 웹 사이트 에 는 1 , 000 개 가 넘 는 TED 강연 들 이 있 습니다 .
여기 계신 여러분 의 대다수 는 정말 대단 한 일 이 라고 생각 하 시 겠 죠 -- 전 다릅니다 .
전 그렇게 생각 하 지 않 아요 .
저 는 여기 한 가지 문제점 이 있 다고 생각 합니다 .
왜냐하면 강연 이 1 , 000 개 라는 것 은 , 공유 할 만 한 아이디어 들 이 1 , 000 개 이상 이 라는 뜻 이 되 기 때문 이 죠 .
도대체 무슨 수로 1 , 000 개 나 되 는 아이디어 를 널리 알릴 건가요 ?
1 , 000 개 의 TED 영상 전부 를 보 면서 그 모든 아이디어 들 을 머리 속 에 집 어 넣 으려고 해도 , 250 시간 이상 의 시간 이 필요 할 겁니다 .
250 시간 이상 의 시간 이 필요 할 겁니다 .
간단 한 계산 을 해 봤 는데요 .
정말 그렇게 하 는 경우 1 인 당 경제 적 손실 은 15 , 000 달러 정도 가 됩니다 .
___

한글 BPE 적용
___
▁현재 ▁TED ▁웹 ▁사이트 ▁에 ▁는 ▁1 ▁, ▁000 ▁개 ▁가 ▁넘 ▁는 ▁TED ▁강 연 ▁들 ▁이 ▁있 ▁습니다 ▁.
▁여기 ▁계 신 ▁여러분 ▁의 ▁대 다 수 ▁는 ▁정말 ▁대단 ▁한 ▁일 ▁이 ▁라고 ▁생각 ▁하 ▁시 ▁겠 ▁죠 ▁-- ▁전 ▁다 릅니다 ▁.
▁전 ▁그렇게 ▁생각 ▁하 ▁지 ▁않 ▁아요 ▁.
▁저 ▁는 ▁여기 ▁한 ▁가지 ▁문 제 점 ▁이 ▁있 ▁다고 ▁생각 ▁합니다 ▁.
▁왜냐하면 ▁강 연 ▁이 ▁1 ▁, ▁000 ▁개 ▁라는 ▁것 ▁은 ▁, ▁공유 ▁할 ▁만 ▁한 ▁아이디어 ▁들 ▁이 ▁1 ▁, ▁000 ▁개 ▁이상 ▁이 ▁라는 ▁뜻 ▁이 ▁되 ▁기 ▁때문 ▁이 ▁죠 ▁.
▁도 대체 ▁무슨 ▁수 로 ▁1 ▁, ▁000 ▁개 ▁나 ▁되 ▁는 ▁아이디어 ▁를 ▁널 리 ▁알 릴 ▁건 가요 ▁?
▁1 ▁, ▁000 ▁개 ▁의 ▁TED ▁영상 ▁전부 ▁를 ▁보 ▁면서 ▁그 ▁모든 ▁아이디어 ▁들 ▁을 ▁머리 ▁속 ▁에 ▁집 ▁어 ▁넣 ▁으 려고 ▁해도 ▁, ▁2 50 ▁시간 ▁이상 ▁의 ▁시간 ▁이 ▁필요 ▁할 ▁겁니다 ▁.
▁2 50 ▁시간 ▁이상 ▁의 ▁시간 ▁이 ▁필요 ▁할 ▁겁니다 ▁.
▁간단 ▁한 ▁계산 ▁을 ▁해 ▁봤 ▁는데요 ▁.
▁정말 ▁그렇게 ▁하 ▁는 ▁경우 ▁1 ▁인 ▁당 ▁경제 ▁적 ▁손 실 ▁은 ▁15 ▁, ▁000 ▁달러 ▁정도 ▁가 ▁됩니다 ▁.
___

영어 NLTK에 의해 tokenization이 된 원문
___
There 's currently over a thousand TED Talks on the TED website .
And I guess many of you here think that this is quite fantastic , except for me , I don 't agree with this .
I think we have a situation here .
Because if you think about it , 1,000 TED Talks , that 's over 1,000 ideas worth spreading .
How on earth are you going to spread a thousand ideas ?
Even if you just try to get all of those ideas into your head by watching all those thousand TED videos , it would actually currently take you over 250 hours to do so .
And I did a little calculation of this .
The damage to the economy for each one who does this is around $ 15,000 .
So having seen this danger to the economy , I thought , we need to find a solution to this problem .
Here 's my approach to it all .
___

영어 BPE 적용
___
▁There ▁'s ▁currently ▁over ▁a ▁thous and ▁TED ▁T al ks ▁on ▁the ▁TED ▁we b site ▁.
▁And ▁I ▁guess ▁many ▁of ▁you ▁here ▁think ▁that ▁this ▁is ▁quite ▁f ant as tic ▁, ▁ex cept ▁for ▁me ▁, ▁I ▁don ▁'t ▁agree ▁with ▁this ▁.
▁I ▁think ▁we ▁have ▁a ▁situation ▁here ▁.
▁Because ▁if ▁you ▁think ▁about ▁it ▁, ▁1 ,000 ▁TED ▁T al ks ▁, ▁that ▁'s ▁over ▁1 ,000 ▁ideas ▁worth ▁sp reading ▁.
▁How ▁on ▁earth ▁are ▁you ▁going ▁to ▁spread ▁a ▁thous and ▁ideas ▁?
▁Even ▁if ▁you ▁just ▁try ▁to ▁get ▁all ▁of ▁those ▁ideas ▁into ▁your ▁head ▁by ▁watching ▁all ▁those ▁thous and ▁TED ▁vide os ▁, ▁it ▁would ▁actually ▁currently ▁take ▁you ▁over ▁2 50 ▁hours ▁to ▁do ▁so ▁.
▁And ▁I ▁did ▁a ▁little ▁cal cu lation ▁of ▁this ▁.
▁The ▁damage ▁to ▁the ▁economy ▁for ▁each ▁one ▁who ▁does ▁this ▁is ▁around ▁$ ▁15 ,000 ▁.
▁So ▁having ▁seen ▁this ▁dang er ▁to ▁the ▁economy ▁, ▁I ▁thought ▁, ▁we ▁need ▁to ▁find ▁a ▁solution ▁to ▁this ▁problem ▁.
▁Here ▁'s ▁my ▁approach ▁to ▁it ▁all ▁.
___

|원문|subword segmentation|
|-|-|
|대다수|대 + 다 + 수|
|문제점|문 + 제 + 점|
|건가요|건 + 가요|
|website|we + b + site|
|except|ex + cept|
|250|2 + 50|
|15,000|15 + ,000|

위와 같이 subword segmentation에 의해서 추가적으로 쪼개진 단어들은 적절한 subword의 형태로 나누어진 것을 볼 수 있습니다. 특히, 숫자 같은 경우에는 자주 쓰이는 숫자 50이나 ,000의 경우 성공적으로 하나의 subword로 지정된 것을 볼 수 있습니다.