# Segmentation using Subword (Byte Pare Encoding, BPE)

[[Sennrich at el.2015]](https://arxiv.org/pdf/1508.07909.pdf)

## Example

한글 원문
```
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
```

한글 BPE 적용
```
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
```

영어 원문
```
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
```

영어 BPE 적용
```
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
```