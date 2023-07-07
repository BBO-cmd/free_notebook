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
