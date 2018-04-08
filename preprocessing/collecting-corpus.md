# Collecting Corpus

Corpus를 구하는 방법은 여러가지가 있습니다. Open된 데이터를 사용할 수도 있고, 구매를 할 수도 있습니다. Open된 데이터는 주로 각종 Task(e.g. Sentiment Analysis and Machine Translation)들의 대회 또는 논문을 위한 데이터 입니다. 여기서는 crawling을 통한 수집을 주로 다루도록 하겠습니다. 다양한 웹사이트에서 crawling을 수행 할 수 있는만큼, 다양한 domain의 corpus를 모을 수 있습니다. 만약 특정 domain만을 위한 NLP task가 아니라면, 특정 domain에 편향(biased)되지 않도록 최대한 다양한 domain에서 corpus를 수집하는 것이 중요합니다.

하지만 무작정 웹사이트로부터 corpus를 crawling하는 것은 법적인 문제가 될 수 있습니다. 저작권 뿐만 아니라, 불필요한 traffic을 웹서버에 가중시킴으로써, 문제가 생길 수 있습니다. 따라서 올바른 방법으로 적절한 웹사이트에서 상업적인 목적이 아닌 경우에 제한된 crawling을 할 것을 권장합니다. 해당 웹사이트의 Crawling에 대한 허용 여부는 그 사이트의 robots.txt를 보면 확인 할 수 있습니다. 예를 들어 TED의 robot.txt는 다음과 같이 확인 할 수 있습니다.

```bash
$ wget https://www.ted.com/robots.txt
$ cat robots.txt
User-agent: *
Disallow: /latest
Disallow: /latest-talk
Disallow: /latest-playlist
Disallow: /people
Disallow: /profiles
Disallow: /conversations

User-agent: Baiduspider
Disallow: /search
Disallow: /latest
Disallow: /latest-talk
Disallow: /latest-playlist
Disallow: /people
Disallow: /profiles
```

모든 User-agent에 대해서 일부 경우에 대해서 disallow 인 것을 확인 할 수 있습니다. robots.txt에 대한 좀 더 자세한 내용은 http://www.robotstxt.org/ 에서 확인 할 수 있습니다.

Crawling 할 때에는 selenium이라는 package를 사용하여 web-browser driver를 직접 control하여 crawling을 수행하곤 합니다. 이때, phantomJS나 chrome과 같은 headless browser를 사용하면, 실제 사용자가 화면에서 보던 것과 같은 page를 다운로드 받을 수 있습니다. (예를 들어 wget과 같이 그냥 다운로드 할 경우, 브라우저와 다른 모양의 페이지가 다운로드 되는 경우도 많습니다.) 이후, beautiful-soup이라는 package를 사용하여 HTML 코드를 쉽고 간단하게 parsing할 수 있습니다.

## Monolingual Corpora

사실 가장 손 쉽게 구할 수 있는 종류의 corpus 입니다. 경우에 따라서 Wikipedia나 각종 Wiki에서는 dump 데이터를 제공하기도 합니다. 따라서 해당 데이터를 다운로드 및 수집하는 것은 손쉽게 대량의 corpus를 얻을 수 있는 방법 중에 하나 입니다. 아래는 domain에 따른 대표적인 corpus의 수집 방식 입니다. 또한 Kaggle에서도 많은 종류의 dataset이 대량으로 upload되어 있으니 필요에 따라 다운로드 받아 사용하면 매우 유용합니다. Crawling을 수행 할 때에는 해당 사이트의 robots.txt 등을 확인하여 적법한 절차를 통해 crawling을 수행하길 권장 합니다.

|문체|domain|수집처|정제 난이도|
|-|-|-|-|
|대화체|일반|채팅 로그|높음|
|대화체|일반|블로그|높음|
|문어체|시사|뉴스 기사|낮음|
|문어체|과학, 교양, 역사 등|Wikipedia|중간|
|문어체|과학, 교양, 역사, 서브컬쳐 등|나무위키|중간|
|대화체|일반(각 분야별 게시판 존재)|[클리앙](https://www.clien.net/)|중간|
|문어체|일반, 시사 등|[PGR21](https://pgr21.com/)|중간|
|대화체|일반|드라마, 영화 자막|낮음|

자막은 저작권이 있는 경우가 많기 때문에 저작권 관련 정보를 잘 확인 하는 것도 중요합니다.

## Multilingual Corpora

Machine Translation을 위한 parallel corpus를 구하는 것은 monolingual corpus에 비해서 상당히 어렵습니다. Crawling을 수행 할 때에는 해당 사이트의 robots.txt 등을 확인하여 적법한 절차를 통해 crawling을 수행하길 권장 합니다. 자막은 저작권이 있는 경우가 많기 때문에 저작권 관련 정보를 잘 확인 하는 것도 중요합니다.

|문체|domain|수집처|정제 난이도|정렬 난이도|
|-|-|-|-|-|
|문어체|시사, 과학 등|[OPUS](http://opus.nlpl.eu/)|낮음|중간|
|대화체|시사, 교양, 과학 등|TED|낮음|중간|
|문어체|시사|[중앙일보영자신문](http://koreajoongangdaily.joins.com/news/list/List.aspx?gCat=060201)|낮음|중간|
|문어체|시사|[동아일보영자신문](http://english.donga.com/)|낮음|중간|
|문어체|일반|[Korean Parallel Data](https://sites.google.com/site/koreanparalleldata/)|낮음|낮음|
|대화체|일반|드라마, 영화 자막|낮음|중간|

자막을 parallel corpus로 사용할 경우 몇가지 문제점이 있습니다. 번역 품질의 저하로 인한 문제는 둘째치고, 'he'나 'she'같은 대명사가 사람 이름과 같은 고유명사로 표현 되어 있는 경우도 많습니다. 따라서, 이러한 문제점들을 두루 고려해야 할 필요성이 있습니다.

## Speech with Transcript

Speech 데이터와 그와 함께 annotated된 transcript 데이터를 구하는 것은 정말 어렵습니다. 주로 연구용으로 공개 된 데이터들을 이용하거나 자막을 활용하여 영상에서 추출 하는 방법도 생각 해 볼 수 있습니다.