# Segmentation using Subword (Byte Pare Encoding, BPE)

[[Sennrich at el.2015]](https://arxiv.org/pdf/1508.07909.pdf)

한글 원문
>큰 거리 는 연중 내내 번잡 하 다
태양 은 지구 의 지름 의 109 배 이 다
( 고어 ) 여론 .
백문 이 불여일견 이 란 말 도 있 잖아 .
남자 들 은 자재 를 기계 에 넣 고 있 다 .
나 는 주인 의 초대 로 파티 에 갔었 다 .
그 음악 은 가사 의 의미 에 꼭 맞 는다 .
인체 의 조직 .
한국 에서 는 가족 이 사회 의 중심 이 다
남자 가 가격표 의 금액 을 보 고 있 다 .

한글 BPE 적용
>▁큰 ▁거리 ▁는 ▁연중 ▁내내 ▁번 잡 ▁하 ▁다
▁태양 ▁은 ▁지구 ▁의 ▁지름 ▁의 ▁10 9 ▁배 ▁이 ▁다
▁( ▁고어 ▁) ▁여론 ▁.
▁백 문 ▁이 ▁불 여 일 견 ▁이 ▁란 ▁말 ▁도 ▁있 ▁잖아 ▁.
▁남자 ▁들 ▁은 ▁자재 ▁를 ▁기계 ▁에 ▁넣 ▁고 ▁있 ▁다 ▁.
▁나 ▁는 ▁주인 ▁의 ▁초대 ▁로 ▁파티 ▁에 ▁갔었 ▁다 ▁.
▁그 ▁음악 ▁은 ▁가사 ▁의 ▁의미 ▁에 ▁꼭 ▁맞 ▁는다 ▁.
▁인체 ▁의 ▁조직 ▁.
▁한국 ▁에서 ▁는 ▁가족 ▁이 ▁사회 ▁의 ▁중심 ▁이 ▁다
▁남자 ▁가 ▁가격 표 ▁의 ▁금액 ▁을 ▁보 ▁고 ▁있 ▁다 ▁.

영어 원문
>There 's currently over a thousand TED Talks on the TED website .
And I guess many of you here think that this is quite fantastic , except for me , I don 't agree with this .
I think we have a situation here .
Because if you think about it , 1,000 TED Talks , that 's over 1,000 ideas worth spreading .
How on earth are you going to spread a thousand ideas ?
Even if you just try to get all of those ideas into your head by watching all those thousand TED videos , it would actually currently take you over 250 hours to do so .
And I did a little calculation of this .
The damage to the economy for each one who does this is around $ 15,000 .
So having seen this danger to the economy , I thought , we need to find a solution to this problem .
Here 's my approach to it all .

영어 BPE 적용
>▁There ▁'s ▁currently ▁over ▁a ▁thousand ▁TED ▁Talks ▁on ▁the ▁TED ▁website ▁.
▁And ▁I ▁guess ▁many ▁of ▁you ▁here ▁think ▁that ▁this ▁is ▁quite ▁fantastic ▁, ▁except ▁for ▁me ▁, ▁I ▁don ▁'t ▁agree ▁with ▁this ▁.
▁I ▁think ▁we ▁have ▁a ▁situation ▁here ▁.
▁Because ▁if ▁you ▁think ▁about ▁it ▁, ▁1,000 ▁TED ▁Talks ▁, ▁that ▁'s ▁over ▁1,000 ▁ideas ▁worth ▁spreading ▁.
▁How ▁on ▁earth ▁are ▁you ▁going ▁to ▁spread ▁a ▁thousand ▁ideas ▁?
▁Even ▁if ▁you ▁just ▁try ▁to ▁get ▁all ▁of ▁those ▁ideas ▁into ▁your ▁head ▁by ▁watching ▁all ▁those ▁thousand ▁TED ▁videos ▁, ▁it ▁would ▁actually ▁currently ▁take ▁you ▁over ▁250 ▁hours ▁to ▁do ▁so ▁.
▁And ▁I ▁did ▁a ▁little ▁calculation ▁of ▁this ▁.
▁The ▁damage ▁to ▁the ▁economy ▁for ▁each ▁one ▁who ▁does ▁this ▁is ▁around ▁$ ▁15,000 ▁.
▁So ▁having ▁seen ▁this ▁danger ▁to ▁the ▁economy ▁, ▁I ▁thought ▁, ▁we ▁need ▁to ▁find ▁a ▁solution ▁to ▁this ▁problem ▁.
▁Here ▁'s ▁my ▁approach ▁to ▁it ▁all ▁.