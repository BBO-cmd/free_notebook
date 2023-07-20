##########################################################
#       이번 방학때 해야할 일 TODO
##########################################################

todo_list={}

#반드시 해야 하는 것
todo_list["academic_must"]=[
"Ambient AI: 7/12 개강, 공부 제대로 하기", 
"Ben Eater 연구플젝: 어떻게 하면 idea를 적용시킬 수 있을지 생각하고, 미리 contact하기",
"창의축전: 전체 계획 잘 짜서 주차별 진행사항 확인하고 weekly progress 확인하기. Milestones 3개정도 세우기",
"다음 학기 수업 계획짜고 필수이수 공부하기: 전공수업 뭐 들을 수 있는지" 
"영어: 완전 fluent 하게 만들기- 특히 listening, speaking"
]

#하고싶은 것
todo_list["academic_additional"]=[
"Deep Learning 논문 스터디 들어가기"
]


# 학업외
todo_list["life"]=[
"면허따기: 고수의 운전면허, 필기시험",
]


# Q. list인 변수명만 받아올 수는 없는지?
# sol) 아예 딕셔너리로, 쓰임새가 많은 변수형을 key에다가 저장해놓으면 변수명 받아오는 복잡한 일 안해도 됨
for list in todo_list:
    print(f"---------    {list}    -------------------")
    for i in enumerate(todo_list[list]):
        print(i,'\n')


##############################################
#    Thoughts, Plans... on a daily basis 
##############################################

#230708
# 1. 운동은 후회할 확률이 0에 수렴하는 매우 희소한 일 중 하나임. 확률적으로 따졌을때 이만큼 승산있는 일을 찾기는 힘들다. 그러니까 오늘도 일단 하고본다. 예외는 없다.
# 2. 딥러닝 논문 일단 혼자라도 시작하기: 이미 익숙한 yolo부터 보면 쉽게 볼 수 있을듯. 일단 지금 관심있는 CV부터 시작
# 3. 영어: 일단 "단어들" 다시 reach out해서 refresh memory할 필요가 있음 

###############################
#230709: Summarize sentences above using transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
'''
# Load pre-trained model and tokenizer
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Example sentences
sentences = [
    "This is the first sentence.",
    "Here's the second sentence.",
    "And finally, the third sentence."
]

# Tokenize and encode the sentences
inputs = tokenizer.batch_encode_plus(
    sentences,
    padding='longest',
    truncation=True,
    return_tensors='pt'
)

# Generate summaries
outputs = model.generate(
    inputs['input_ids'],
    num_beams=4,
    max_length=100,
    early_stopping=True
)

# Decode and print the summaries
summaries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
for summary in summaries:
    print(summary)
'''

#############################
#230710 continue to use transformers: Colab에서 실행

## BERT tokenizer가 하는 일: 
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("Vocabulary size:", tokenizer.vocab_size)
print("Token IDs example:", tokenizer.encode("Hello, how are you?"))
print("Decoded example:", tokenizer.decode([101, 7592, 1010, 2129, 2024, 2017, 1029]))

# Vocabulary size: 30522
# Token IDs example: [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]
# Decoded example: [CLS] hello, how are you? -tonekize-> ['hello', ',', 'how', 'are', 'you', '?']
# [101] represents the [CLS] token, which indicates the start of the sequence.

sum=0
t_list=tokenizer.encode("Hello, how are you?")
for i in t_list:
  sum+=i
print("\nSUM: ",sum) # 16004, != 30522(voca size of bert tok.)
## tokenizer의 vocabulary size는 total 몇개의 tokekn을 cover하는지 그 coverage이고, 문장과는 무관


# BERT, GPT... 등의 Language model은 기본적으로 단어 기준 tokenize-> encoding해서 numerical하게 만든 다음
# "TOKEN들간의 relationships and dependencies를", "Attention mechanism"을 통해 capture함
# token들간의 관계: ex) 인접하여/멀리 얼마나 자주 나왔는지(Dependencies), 대명사 등에 의해 재언급되는 경우 등 


# summarizer(bert-base-uncase)/ tokenizer(facebook/bart-large-cnn) 사용 시도
pip install transformers
import numpy as np

from transformers import pipeline, AutoTokenizer

##summarizer는 tokeizer 필요없이 그냥 plain text를 바로 summarizer에 넣으면 됨
def summarize_sentences(sentences):
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []

    for sentence in sentences:
        #tokens = tokenizer.encode(sentence, truncation=True, padding=True, max_length=512, return_tensors="pt")
        #tokens = tokens.tolist()
        summary = summarizer(sentence, max_length=10, min_length=5, do_sample=False)
        #deoced_tokens = tokenizer.decode(tokens)
        summaries.append(summary[0]['summary_text'])

    return summaries

# 내 문장들
sentences = [
# 1. 운동은 후회할 확률이 0에 수렴하는 매우 희소한 일 중 하나임. 확률적으로 따졌을때 이만큼 승산있는 일을 찾기는 힘들다. 그러니까 오늘도 일단 하고본다. 예외는 없다.
"Exercise is one of the rarest tasks of which the probability of regretting converges to 0. It's hard to find tasks that have less probability of regretting, so I'm gonna do this today, too. No exceptions."
# 2. 딥러닝 논문 일단 혼자라도 시작하기: 이미 익숙한 yolo부터 보면 쉽게 볼 수 있을듯. 일단 지금 관심있는 CV부터 시작
, "Getting started on reading papers about Deep Learning. It would be better to start with yolo, which I am already famailiar with. Let me cover CV first."
# 3. 영어: 일단 "단어들" 다시 reach out해서 refresh memory할 필요가 있음
, "Firstly, I have to refresh my memory by reaching out the vocabularies."
              ]

summary_results = summarize_sentences(sentences)

for i in range(len(sentences)):
    print("Original sentence:", sentences[i])
    print("Summary:", summary_results[i])
    print()


##결과: 
#Original sentence: Exercise is one of the hardest things of which the probability of regret converges to 0. It's hard to find another one that has higher probability so I'm gonna do this today, too. No exceptions.
#Summary: Exercise is one of the hardest 

#Original sentence: Getting started on reading papers about Deep Learning. It would be better to start with yolo, which is already famailiar. Let me cover from CV first.
#Summary: Getting started on reading papers about Deep

#Original sentence: Firstly, I have to refresh my memory by reaching out the vocabularies.
#Summary: "I have to refresh my memory

## 문제: 이런식으로 요약이 아니라 그냥 잘림ㅋㅎ


##################################
#230712 troubleshooting previous problems

##tokenize
origin = "The technology industry is constantly evolving and innovating. New advancements and breakthroughs are made every day. One of the most exciting recent developments is the rise of artificial intelligence and machine learning. These technologies have the potential to revolutionize various fields such as healthcare, finance, and transportation. They can automate tasks, improve decision-making processes, and unlock new insights from data. As the adoption of AI and ML continues to grow, it is important to stay informed about the latest trends and applications."
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens=tokenizer.encode(origin)
print("num of tokens: ", len(tokens) #before summarize: 103 tokens


##summarize
from transformers import pipeline

# 일케 하면 pipeline에서 summarization에 적합한 pre-trained model을 자동으로 선택함: t5-base 등
summarizer = pipeline("summarization")

#origin(str)자리에 origin(str)들의 list로 넣으면 저절로 summary list의 dictionary도 len(text_list)만큼 생김
origin=[ origin, origin, origin ]
print("original text x3:" , origin)

summary= summarizer(origin, max_length = 100, min_length=30)
print(summary)    #[ {'summary_text':'~~~~'}, {'summary_text':'~~~'},{'summary_text':} ]
#key가 'summary_text"인 dictionary가 len(원문_list)만큼 생김

#after summarization: 45 tokens(summaized된거 맞음)
summary_1st = summary[0]['summary_text']
len(tokenizer.encode(summary_1st))



##apply summarizer to my sentences
sentences = [
"Exercise is one of the rarest tasks of which the probability of regretting converges to 0. It's hard to find tasks that have less probability of regretting, so I'm gonna do this today, too. No exceptions."
, "Getting started on reading papers about Deep Learning. It would be better to start with yolo, which I am already famailiar with. Let me cover CV first."
, "Firstly, I have to refresh my memory by reaching out the vocabularies."
]

#origin txt가 너무 짧아서 summay가 되는지 안되는지 판단불가 -> len 100+ words 로 extend
#done by chatGPT
extended_sentences =["Engaging in physical exercise is undeniably one of the rarest and most rewarding tasks in life, with a probability of experiencing regret that effortlessly approaches zero. It is an arduous endeavor to come across activities that offer such an incredibly low likelihood of subsequent remorse, which is precisely why I am resolute in dedicating myself to this pursuit today, as I have done on countless occasions before. Regardless of circumstances or external factors, I am unwavering in my commitment to seize this opportunity and partake in the revitalizing power of exercise without making any exceptions or allowances for any excuse that might arise.",
  "aaaaaa",
  "bbbbbb" 
]


my_summary=summarizer(extended_sentences, max_length=50, min_length=0)
print(my_summary)
#somehow summarize 완료됨(일단 문장 하나로 확인)
# 결과: 'Engaging in physical exercise is undeniably one of the rarest and most rewarding tasks in life, with a probability of experiencing regret that effortlessly approaches zero . Regardless of circumstances or external factors, I am unwavering in my commitment to seize this'
# 핵심만 잘 summarize 됨
# one step further: 내 원래 txt와 extended ver., 그리고 summarized txt를 비교해볼 수 있음


##20230718 MLP model build

#Sequential() 사용하여 layer 쌓기
from keras.models import Sequential
from keras.layers import Dense


##모델 만드는 방법 1: Sequential
#1. load model
model=Sequential()
#2. input layer ( unit은 node 개수임 )
model.add(Dense(units = 64, activation = 'relu', input-dim=100))
#3. hidden layer
model.add(Dense(units = 32, activation = 'relu'))
#4. output layer
model.add(Dense(units=10, activation ='softmax'))
#5. model compile
model.compile(loss='categorical_crossentropy'. optimizer='adam', metrics=['accuracy'])


## 또는 (Flatten 부터) 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten()
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])
model.compile(loss='', optimizer='',metrics=[])


##230720 Model save and load

# 1. weight만 저장하기(model.save_weights): 
model.save_weights(param_paths) # param_paths는 parameter가 담긴 checkpoint 파일(.ckpt)이 저장되는 경로
# 같은 형태의 model을 새로 build한 후 저장된 weight 불러오기
model_2=Sequential([...])
model_2.compile(loss='', optimizer='', metrics=[])

model_2.load_weights(param.path) #param_path는 ckpt파일의 경로임


#2. model 자체를 저장하기(model.save(path to save) 또는 tf.keras.models.save_model(model, path to save) ) : model, weight, compile된것을 통째로 저장
model_path = ./your/directory # 모델 저장경로

model.save(model_path)
#또는 
tf.keras.models.save_model(model,model_path)

# 저장된 모델을 load함(models.load_model(path of the model))
loaded_model = tf.keras.models.load_model(model_path)


#3. h5로 저장할 수도 있음




## training 중간에 특정 함수를 넣고싶을 때: Callback 함수 활용(ex. 끝날때마다 ckpt저장 및 learning rate 조정, early stopping 등)
# 1. epoch마다 ckpt 파일 저장하기
ckpt_dir = os.path.join(save_root, 'ckpt저장할 새폴더명')
ckpt_prefix = os.path.join(checkpoint_dir, f'ckpt_{epoch}')

#2. learning rate shceduler
def decay(epoch):
    if epoch<3:
        return 1e-3
    elif 3 =< epoch and epoch<7:
        return 1e-4
    else:
        return 1e-5


import match  
def exp_decay(epoch):  
    if epoch<3:
        return 0.001
    else:
        return 0.001 * math.exp(0.1*(3-epoch))



#callbacks=[callback_f1, callback_f2,... ]넣어놓고 학습시에 model.fit(    ,   ,callbacks)해서 사용함
callbacks =[
    tf.keras.callbacks.ModelCheckpoint(filepath=, save_weights_only=True),
    tf.keras.callbacks.LearningTateScheduler(decay),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patence=3)
    PrintLR()
]


##TensorBoard
#1. callback에 tensorboard넣기
log_dir='경로'
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)



##convolution 
#1. Datashape: 28*28 에다가 channel을 추가해주어야 함(np.expand_dim)
def expand(input_img):
    return np.expand_dim(input_img, axis=-1)

#2. 간단 convolution layer로 model 구성
# Conv2D:
#MaxPool2D:

model.tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filter=64, kernel_size=(3,3), padding='same', activation ='relu', 
                           input_shape=x_train.shape[1:]), #input_shape에서 이미지 번호 제외
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filter=128, kernel_size=(3,3), padding='same', activation ='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Flatten(), #1D array로 펼쳐줌
    tf.keras.layers.Dense(10, activation='softmax')#fully connectd layer
])

#컴파일
loss_fn='cross...'
optimizer='adam'

