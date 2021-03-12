compare={                   #eq mt(more than) lt(less than) nmt(no more than) nlt(no less than)
    'eq':lambda a,b:a==b,
    'mt':lambda a,b:a>b,
    'lt':lambda a,b:a<b,
    'nmt':lambda a,b:a<=b,
    'nlt':lambda a,b:a>=b
}

#------function of ops (Algorithm)--------
'''
ws: (word,word,word,....)  ('NER',NERTYPE) word 
w: word ('NER',NERTYPE) 
c: candidate  
p: function 
arg: ArgX or ArgY
a: (Word1,Word2)
'''
#List of attributes we need to support besides 'word' and 'tokens'
Selection = ['NER']

def count_sublist(lis,sublis):
    cnt = 0
    if len(sublis)>len(lis):
        return cnt
    len_sub = len(sublis)
    for st in range(len(lis)-len(sublis)+1):
        if lis[st:st+len_sub]==sublis:
            cnt+=1
    return cnt


#function for $And
def merge(x,y):
    if type(x)!= tuple:
        x = [x]
    if type(y)!= tuple:
        y = [y]
    x = list(x)
    y = list(y)
    return tuple(x+y)

#function for $Is
def IsFunc(ws,ps,c):
    if isinstance(ps,tuple):
        bool_list  = []
        for p in ps:
            if isinstance(ws, tuple):
                if ws[0] in Selection:
                    bool_list.append(p(ws)(c))
                else:
                    bool_list.append(all([p(w)(c) for w in ws]))
            else:
                bool_list.append(p(ws)(c))
        return all(bool_list)

    if isinstance(ws,tuple):
        if ws[0] in Selection:
            return ps(ws)(c)
        else:
            return all([ps(w)(c) for w in ws])
    else:
        return ps(ws)(c)

#function for @Left and @Right
def at_POSI(POSI,ws,arg,c,option=None):
    if isinstance(ws,tuple):
        if ws[0] in Selection:
            return ws[1] in c.get_other_posi(POSI,arg[-1])[ws[0]]
        else:
            bool_list = []
            for w in ws:
                bool_list.append(at_POSI_0(POSI,arg,w,c,option))
            return all(bool_list)
    else:
        return at_POSI_0(POSI,arg,ws,c,option)

#function for @Left0 and @Right0
def at_POSI_0(POSI,arg,w,c,option=None):
    if arg not in ['ArgX','ArgY']:
        w,arg = arg,w
        if POSI == 'Left':
            POSI = 'Right'
        elif POSI == 'Right':
            POSI = 'Left'

    if isinstance(w,tuple) and w[0] not in Selection:
        return all([at_POSI_0(POSI,arg,ww,c,option) for ww in w])

    if w=='ArgY':
        w = c.obj
    elif w=='ArgX':
        w = c.subj

    if option==None:
        option = {'attr': 'word', 'range': -1, 'numAppear':1,'cmp':'nlt','onlyCount':False}  #For now, if onlyCount==True, then attr=='tokens'
    if isinstance(w,tuple):
        return w[1] in c.get_other_posi(POSI, arg[-1])[w[0]]
    else:                                                                                   #For now,option['attr']==tokens if and only if 'right before' is used or onlyCount==True, otherwise ==word
        w = w.lower()
        w_split = w.split()
        while '' in w_split:
            w_split.remove('')
        if option == 'Direct':
            range_ = len(w_split)
            info = [token.lower() for token in c.get_other_posi(POSI, arg[-1])['tokens']]
            if POSI == 'Left':
                st = max(0, len(info) - range_)
                info = info[st:]
            elif POSI == 'Right':
                ed = min(len(info) - 1, range_-1)
                info = info[:ed + 1]
            else:
                raise ValueError
            # print(info)
            if info==w_split:
                return True
            else:
                return False
        if option['range']==-1:
            info = [token.lower() for token in c.get_other_posi(POSI, arg[-1])['tokens']]
        else:                                                                               #For now, if range!=-1 then attr == 'tokens'
            info = [token.lower() for token in c.get_other_posi(POSI, arg[-1])[option['attr']]]
            range_ = option['range']
            if POSI == 'Left':
                st = max(0, len(info) - range_)
                info = info[st:]
            elif POSI=='Right':
                ed = min(len(info) - 1, range_-1)
                info = info[:ed + 1]
            else:
                count_posi = c.get_other_posi(POSI, arg[-1])['POSI']
                st = max(0,count_posi-range_)
                ed = min(len(info),count_posi+1+range_)
                info = info[st:ed+1]
        if option['onlyCount']:
            return compare[option['cmp']](len(info),option['numAppear'])
        else:
            return compare[option['cmp']](count_sublist(info,w_split),option['numAppear'])


#function for @Between

def at_between(w,c,option=None,a=None):
    if w=='ArgY':
        w = c.obj
    elif w=='ArgX':
        w = c.subj
    if option==None:
        option = {'attr': 'word', 'numAppear':1,'cmp':'nlt','onlyCount':False}                #For now, if onlyCount==True, then attr=='tokens'
    if isinstance(w,tuple):
        return w[1] in c.get_mid()[w[0]]
    else:                                                                                   #For now,option['attr']==tokens if and only if  onlyCount==True, otherwise ==word
        w = w.lower()
        w_split = w.split()
        while '' in w_split:
            w_split.remove('')
        info = [token.lower() for token in c.get_mid()['tokens']]
        if option['onlyCount']:
            return compare[option['cmp']](len(info),option['numAppear'])
        else:
            # print(info, w,info.count(w),type(info.count(w)),option['numAppear'],option['eq'],compare[option['eq']](info.count(w),option['numAppear']))
            return compare[option['cmp']](count_sublist(info,w_split),option['numAppear'])


#function for counting
def at_lessthan(funcx,nouny,w,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'lt','onlyCount':onlyCount})(c)                #There are less than 3 words before OBJ
    else:
        return funcx(w, {'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(c)      #the word 'x' is less than 3 words before OBJ

def at_atmost(funcx,nouny,w,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'nmt','onlyCount':onlyCount})(c)                #There are at most 3 words before OBJ
    else:
        return funcx(w, {'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(c)      #the word 'x' is at most 3 words before OBJ

def at_atleast(funcx,nouny,w,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'nlt','onlyCount':onlyCount})(c)             #There are at least 3 words before OBJ
    else:
        return funcx(w,{'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(c)      #the word 'x' is no less than 3 words before OBJ

def at_morethan(funcx,nouny,w,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'mt','onlyCount':onlyCount})(c)                #There are more than 3 words before OBJ
    else:
        return funcx(w, {'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(c)    #the word 'x' is more than 3 words before OBJ


#function for @In0
def at_In0(arg,w,c):
    assert arg=='Sentence'
    if isinstance(w,tuple):
        return w in c.ner
    else:
        w = w.lower().split()
        info = [token.lower() for token in c.tokens]
        return count_sublist(info,w)>0

def at_WordCount(nounNum,nouny,F,c):
    if isinstance(nouny,tuple):
        return all([F(noun, option={'attr': 'word', 'range': -1, 'numAppear': 1, 'cmp': 'nlt', 'onlyCount': False})(c) for noun in nouny]) and F(nouny[0],option={'attr': 'tokens','range': -1,'numAppear':sum([len(noun.split()) for noun in nouny]),'cmp': 'eq','onlyCount': True})(c)
    else:
        return F(nouny,option={'attr':'word','range':-1,'numAppear':1,'cmp':'nlt','onlyCount':False})(c) and F(nouny,option={'attr':'tokens','range':-1,'numAppear':len(nouny.split()),'cmp':'eq','onlyCount':True})(c)