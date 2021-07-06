"""
    Functions that semantic phrases map to when constructing soft-matching functions

    Author: Ziqi
"""
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def gather_nd_mask(mask_mat,param):
    param = param.tolist()
    length = mask_mat.shape[2]
    for e in param:
        assert len(e)==2
    rax = []
    for e in param:
        if e[0]<0 or e[1]<0 or e[0]>=length or e[1]>=length:
            rax.append(torch.zeros([1,length],dtype=torch.float32).to(device))
        else:
            rax.append(mask_mat[e[0],e[1],:].view(1,-1))
    return torch.cat(rax,dim=0)

unmatch_count_score = 0.5
unmatch_count_dist = 7
unmatch_match_score = 0.5
unmatch_match_dist = 7

# compare_soft: torch.max input must be 2 tensor? have chance to be think (input,dim) rather thant (input,other)? in other func too?
# reshape -> view?
# torch.tensor   -> to.device

NER_LABEL_SPACE = {} # is dynamically filled to allow different types of NERS to work in explanations

Selection = ['NER']

compare_soft={                   #eq mt(more than) lt(less than) nmt(no more than) nlt(no less than)
    'eq':lambda a,b:torch.max(torch.eq(a,b).to(torch.float32),other=(torch.ge(a,b-unmatch_count_dist)*torch.le(a,b+unmatch_count_dist)).to(torch.float32)*unmatch_count_score),
    'mt':lambda a,b:torch.max(torch.gt(a,b).to(torch.float32),other=torch.gt(a,b-unmatch_count_dist).to(torch.float32)*unmatch_count_score),
    'lt':lambda a,b:torch.max(torch.lt(a,b).to(torch.float32),other=torch.lt(a,b+unmatch_count_dist).to(torch.float32)*unmatch_count_score),
    'nmt':lambda a,b:torch.max(torch.le(a,b).to(torch.float32),other=torch.le(a,b+unmatch_count_dist).to(torch.float32)*unmatch_count_score),
    'nlt':lambda a,b:torch.max(torch.ge(a,b).to(torch.float32),other=torch.ge(a,b-unmatch_count_dist).to(torch.float32)*unmatch_count_score),
}

#c: Tensor [seqlen seqlen 2 2] : sentence,ner,subj_posi,obj_posi,subj,obj
def get_mid(attr,mask_mat,c):
    subj_posi = c[:,-4:-3]
    obj_posi = c[:,-3:-2]
    seqlen = (c.size()[1]-4)//2
    tokens = c[:,0:seqlen]
    ner = c[:,seqlen:seqlen*2]
    st_posi = torch.min(subj_posi,obj_posi)
    ed_posi = subj_posi+obj_posi-st_posi
    mask = gather_nd_mask(mask_mat,torch.cat([st_posi+1,ed_posi-1],dim=1))
    ner = ner*mask
    tokens = tokens*mask
    if attr=='NER':
        res = ner
    else:
        res = tokens
    return res

def get_other_posi(POSI,attr,arg,mask_mat,c):
    assert POSI=='Left' or POSI=='Right'
    batch_size = c.size()[0]
    subj_posi = c[:, -4:-3]
    obj_posi = c[:, -3:-2]
    seqlen = (c.size()[1] - 4) // 2
    tokens = c[:, 0:seqlen]
    ner = c[:, seqlen:seqlen * 2]
    if arg=='ArgY':
        posi = obj_posi
    else:
        posi = subj_posi

    if arg=='Left':
        mask = gather_nd_mask(mask_mat,torch.cat([0*posi,posi-1],dim=1))
    else:
        mask = gather_nd_mask(mask_mat,torch.cat([posi+1,(0*posi+1)*(seqlen-1)],dim=1))

    new_ner = mask*ner
    new_tokens = mask*tokens
    if attr=='NER':
        res = new_ner
    else:
        res = new_tokens
    return res

def get_range(attr,arg,range_,mask_mat,c):
    subj_posi = c[:, -4:-3]
    obj_posi = c[:, -3:-2]
    seqlen = (c.size()[1] - 4) // 2
    tokens = c[:, 0:seqlen]
    ner = c[:, seqlen:seqlen * 2]
    if arg=='ArgY':
        posi = obj_posi
    else:
        posi = subj_posi
    if attr=='NER':
        res = ner
    else:
        res = tokens

    mask = gather_nd_mask(mask_mat,torch.cat([torch.max(torch.tensor(0).to(device),other=posi-range_),torch.min(torch.tensor(seqlen-1).to(device),posi+range_)],dim=1))
    res = res*mask
    return res

def In(w,seq):
    assert w in NER_LABEL_SPACE
    idx = NER_LABEL_SPACE[w]
    eq = torch.eq(seq,idx)
    sum_ = torch.sum(eq,dim=1).view([1,-1])
    return sum_.to(torch.bool)


def merge_soft(x,y):
    if type(x)!= tuple:
        x = [x]
    if type(y)!= tuple:
        y = [y]
    x = list(x)
    y = list(y)
    return tuple(x+y)

#function for $Is
def IsFunc_soft(ws,ps,label_mat,keyword_dict,mask_mat,c):
    if isinstance(ps,tuple):
        bool_list  = []
        for p in ps:
            if isinstance(ws, tuple):
                if ws[0] in Selection:
                    bool_list.append(p(ws)(label_mat,keyword_dict,mask_mat)(c))
                else:
                    bool_list.append(torch.max(torch.tensor(sum([p(w)(label_mat,keyword_dict,mask_mat)(c) for w in ws])-len(ws)+1).to(device),torch.tensor(0.0).to(device)))
            else:
                bool_list.append(p(ws)(label_mat,keyword_dict,mask_mat)(c))
        return torch.max(torch.tensor(sum(bool_list)-len(bool_list)+1).to(device),torch.tensor(0.0).to(device))

    if isinstance(ws,tuple):
        if ws[0] in Selection:
            return ps(ws)(label_mat,keyword_dict,mask_mat)(c)
        else:
            return torch.max(torch.tensor(sum([ps(w)(label_mat,keyword_dict,mask_mat)(c) for w in ws])-len(ws)+1).to(device),torch.tensor(0.0).to(device))
    else:
        return ps(ws)(label_mat,keyword_dict,mask_mat)(c)

#function for @Left and @Right
def at_POSI_soft(POSI,ws,arg,label_mat,keyword_dict,mask_mat,c,option=None):
    if isinstance(ws,tuple):
        if ws[0] in Selection:
            assert POSI=='Left' or POSI=='Right' or POSI=='Range'
            if POSI=='Left' or POSI=='Right':
                score_raw = In(ws[1],get_other_posi(POSI,ws[0],arg,mask_mat,c)).to(torch.float32)
                return score_raw
            else:
                score_raw =  In(ws[1], get_range(ws[0], arg, option['range'],mask_mat,c)).to(torch.float32)
                return score_raw
        else:
            bool_list = []
            for w in ws:
                bool_list.append(at_POSI_0_soft(POSI,arg,w,label_mat,keyword_dict,mask_mat,c,option))
            return torch.max(torch.tensor(sum(bool_list)-len(bool_list)+1).to(device),torch.tensor(0.0).to(device))
    else:
        return at_POSI_0_soft(POSI,arg,ws,label_mat,keyword_dict,mask_mat,c,option)

#function for @Left0 and @Right0
def at_POSI_0_soft(POSI,arg,w,label_mat,keyword_dict,mask_mat,c,option=None):
    if arg not in ['ArgX','ArgY']:
        w,arg = arg,w
        if POSI == 'Left':
            POSI = 'Right'
        elif POSI == 'Right':
            POSI = 'Left'

    if isinstance(w,tuple) and w[0] not in Selection:
        return torch.max(torch.tensor(sum([at_POSI_0_soft(POSI,arg,ww,label_mat,keyword_dict,mask_mat,c,option) for ww in w])-len(w)+1).to(device),torch.tensor(0.0).to(device))


    if option==None:
        option = {'attr': 'word', 'range': -1, 'numAppear':1,'cmp':'nlt','onlyCount':False}  #For now, if onlyCount==True, then attr=='tokens'
    if isinstance(w,tuple):
        assert POSI == 'Left' or POSI == 'Right' or POSI == 'Range'
        if POSI == 'Left' or POSI == 'Right':
            score_raw = In(ws[1], get_other_posi(POSI, ws[0], arg, mask_mat,c)).to(torch.float32)
            return score_raw
        else:
            score_raw = In(ws[1], get_range(ws[0], arg, option['range'], mask_mat,c)).to(torch.float32)
            return score_raw
    else:                                                                                   #For now,option['attr']==tokens if and only if 'right before' is used or onlyCount==True, otherwise ==word
        if option == 'Direct':
            w_split = w.split()
            while '' in w_split:
                w_split.remove('')
            range_ = len(w_split)

            subj_posi = c[:, -4:-3]
            obj_posi = c[:, -3:-2]
            seqlen = (c.size()[1] - 4) // 2

            if arg=='ArgY':
                arg_posi = obj_posi
            else:
                arg_posi = subj_posi

            if POSI == 'Left':
                st = torch.max(torch.tensor(0).to(device), arg_posi - range_)
                position = torch.cat([st,arg_posi-1],dim=1)
                unmatch_position = torch.cat([torch.max(torch.tensor(0).to(device),arg_posi - range_ - unmatch_match_dist), arg_posi - 1], dim=1)
            elif POSI == 'Right':
                st = arg_posi
                ed = torch.min(torch.tensor(seqlen - 1).to(device), arg_posi+range_)
                position = torch.cat([st+1,ed],dim=1)
                unmatch_position = torch.cat([st + 1, torch.min(torch.tensor(seqlen - 1).to(device),st+range_+unmatch_match_dist)], dim=1)
            else:
                raise ValueError
            if w in ['ArgX','ArgY']:
                if w=='ArgX':
                    tar = c[:,-4:-3]
                    src = c[:,-3:-2]
                else:
                    src = c[:,-4:-3]
                    tar = c[:,-3:-2]
                if POSI=='Left':
                    score_raw = torch.eq(src-tar,1).to(torch.float32).view([1,-1])
                    return score_raw
                else:
                    score_raw = torch.eq(tar-src, 1).to(torch.float32).view([1,-1])
                    return score_raw
            else:
                mask_raw = gather_nd_mask(mask_mat,position).to(torch.float32)
                mask_raw = mask_raw+(gather_nd_mask(mask_mat,unmatch_position).to(torch.float32)-mask_raw)*unmatch_match_score
                return torch.max(label_mat[:, :, keyword_dict[w]] * (mask_raw),dim=1)[0].view([1, -1])

        if option['range']==-1:
            assert POSI=='Left' or POSI=='Right'
            batch_size = c.size()[0]
            subj_posi = c[:, -4:-3]
            obj_posi = c[:, -3:-2]
            seqlen = (c.size()[1] - 4) // 2
            if arg == 'ArgY':
                arg_posi = obj_posi
            else:
                arg_posi = subj_posi

            if POSI=='Left':
                position = torch.cat([0*arg_posi, arg_posi - 1], dim=1)
                unmatch_position = position
            else:
                position = torch.cat([arg_posi + 1, (0*arg_posi+1) * (seqlen - 1)],dim=1)
                unmatch_position = position

        else:                                                                               #For now, if range!=-1 then attr == 'tokens'
            subj_posi = c[:, -4:-3]
            obj_posi = c[:, -3:-2]
            seqlen = (c.size()[1] - 4) // 2
            if arg == 'ArgY':
                arg_posi = obj_posi
            else:
                arg_posi = subj_posi

            range_ = option['range']

            if POSI == 'Left':
                st = torch.max(torch.tensor(0).to(device), arg_posi - range_)
                position = torch.cat([st, arg_posi-1],dim=1)
                unmatch_position = torch.cat([torch.max(torch.tensor(0).to(device), arg_posi - range_ - unmatch_match_dist), arg_posi - 1],dim=1)
            elif POSI=='Right':
                st = arg_posi
                ed = torch.min(torch.tensor(seqlen - 1).to(device), arg_posi+range_)
                position = torch.cat([arg_posi+1, ed],dim=1)
                unmatch_position = torch.cat([st + 1, torch.min(torch.tensor(seqlen - 1).to(device), st + range_ + unmatch_match_dist)],dim=1)
            else:
                st = torch.max(torch.tensor(0).to(device),arg_posi-range_)
                ed = torch.min(torch.tensor(seqlen-1).to(device),arg_posi+range_)
                position=torch.cat([st,ed],dim=1)
                unmatch_position = torch.cat([torch.max(torch.tensor(0).to(device), arg_posi - range_ - unmatch_match_dist), torch.min(torch.tensor(seqlen - 1).to(device), st + range_ + unmatch_match_dist)], dim=1)

        if option['onlyCount']:
            score_raw = compare_soft[option['cmp']](position[:,1]-position[:,0],option['numAppear']).to(torch.float32).view([1,-1])
            return score_raw
        else:
            assert option['cmp']=='nlt' and option['numAppear']==1
            if w in ['ArgX','ArgY']:
                if w=='ArgX':
                    tar = c[:,-4]
                else:
                    tar = c[:,-3]
                score_raw = ((tar>=position[:,0])*(tar<=position[:,1])).to(torch.float32).view([1,-1])
                return score_raw
            else:
                mask_raw = gather_nd_mask(mask_mat,position).to(torch.float32)
                mask_raw = mask_raw+(gather_nd_mask(mask_mat,unmatch_position).to(torch.float32)-mask_raw)*unmatch_match_score
                return torch.max(label_mat[:, :, keyword_dict[w]] * (mask_raw),dim=1)[0].view([1, -1])



#function for @Between

def at_between_soft(w,label_mat,keyword_dict,mask_mat,c,option=None):
    if option==None:
        option = {'attr': 'word', 'numAppear':1,'cmp':'nlt','onlyCount':False}                #For now, if onlyCount==True, then attr=='tokens'
    if isinstance(w,tuple):
        score_raw = In(w[1],get_mid(w[0],mask_mat,c)).to(torch.float32)
        return score_raw
    else:                                                                                   #For now,option['attr']==tokens if and only if  onlyCount==True, otherwise ==word
        if option['onlyCount']:
            score_raw =  compare_soft[option['cmp']](torch.abs(c[:,-4]-c[:,-3])-1,option['numAppear']).to(torch.float32).view([1,-1])
            return score_raw
        else:
            assert option['cmp'] == 'nlt' and option['numAppear'] == 1
            l_posi = torch.min(c[:,-4],c[:,-3])
            g_posi = c[:,-4]+c[:,-3]-l_posi
            if w in ['ArgX','ArgY']:
                if w=='ArgX':
                    tar = c[:,-4]
                else:
                    tar = c[:,-3]
                score_raw = ((tar>l_posi)*(tar<g_posi)).to(torch.float32).view([1,-1])
                return score_raw
            else:
                l_posi = l_posi.view([-1,1])+1
                g_posi = g_posi.view([-1, 1])-1
                position = torch.cat([l_posi, g_posi],dim=1)
                mask_raw = gather_nd_mask(mask_mat,position).to(torch.float32)
                seqlen = (c.size()[1] - 4) // 2
                return torch.max(label_mat[:, :, keyword_dict[w]] * (mask_raw),dim=1)[0].view([1, -1])



#function for counting
def at_lessthan_soft(funcx,nouny,w,label_mat,keyword_dict,mask_mat,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'lt','onlyCount':onlyCount})(label_mat,keyword_dict,mask_mat)(c)                #There are less than 3 words before OBJ
    else:
        return funcx(w, {'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(label_mat,keyword_dict,mask_mat)(c)      #the word 'x' is less than 3 words before OBJ

def at_atmost_soft(funcx,nouny,w,label_mat,keyword_dict,mask_mat,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'nmt','onlyCount':onlyCount})(label_mat,keyword_dict,mask_mat)(c)                #There are at most 3 words before OBJ
    else:
        return funcx(w, {'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(label_mat,keyword_dict,mask_mat)(c)      #the word 'x' is at most 3 words before OBJ

def at_atleast_soft(funcx,nouny,w,label_mat,keyword_dict,mask_mat,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'nlt','onlyCount':onlyCount})(label_mat,keyword_dict,mask_mat)(c)             #There are at least 3 words before OBJ
    else:
        return funcx(w,{'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(label_mat,keyword_dict,mask_mat)(c)      #the word 'x' is no less than 3 words before OBJ

def at_morethan_soft(funcx,nouny,w,label_mat,keyword_dict,mask_mat,c):
    if w=='There':
        onlyCount=True
    else:
        onlyCount = False
    if onlyCount:
        return funcx(w,{'attr':nouny['attr'],'range':-1,'numAppear':nouny['num'],'cmp':'mt','onlyCount':onlyCount})(label_mat,keyword_dict,mask_mat)(c)                #There are more than 3 words before OBJ
    else:
        return funcx(w, {'attr': nouny['attr'], 'range': nouny['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': onlyCount})(label_mat,keyword_dict,mask_mat)(c)    #the word 'x' is more than 3 words before OBJ


#function for @In0

def at_In0_soft(arg,w,label_mat,keyword_dict,mask_mat,c):
    assert arg=='Sentence'                                                                                                          #Right?
    if isinstance(w,tuple):
        seqlen = (c.size()[1] - 4) // 2
        score_raw = In(w,c[:,seqlen:seqlen*2]).to(torch.float32)
        return score_raw
    else:
        seqlen = (c.size()[1] - 4) // 2
        return torch.max(label_mat[:, :, keyword_dict[w]], dim=1)[0].view([1, -1])
        # return w in c.sentence

def at_WordCount_soft(nounNum,nouny,F,label_mat,keyword_dict,mask_mat,c):
    if isinstance(nouny,tuple):
        return torch.max(torch.max(torch.tensor(sum([F(noun, option={'attr': 'word', 'range': -1, 'numAppear': 1, 'cmp': 'nlt', 'onlyCount': False})(label_mat,keyword_dict,mask_mat)(c) for noun in nouny])-len(nouny)+1).to(device),torch.tensor(0.0).to(device))+F(nouny[0],option={'attr': 'tokens','range': -1,'numAppear':sum([len(noun.split()) for noun in nouny]),'cmp': 'eq','onlyCount': True})(label_mat,keyword_dict,mask_mat)(c)-1,torch.tensor(0.0).to(device))
    else:
        return torch.max(torch.tensor(F(nouny, option={'attr':'word','range':-1,'numAppear':1,'cmp':'nlt','onlyCount':False})(label_mat,keyword_dict,mask_mat)(c)+F(nouny, option={'attr':'tokens','range':-1,'numAppear':len(nouny.split()),'cmp':'eq','onlyCount':True})(label_mat,keyword_dict,mask_mat)(c)-1).to(device),torch.tensor(0.0).to(device))