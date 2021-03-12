import torch
import pdb

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

unmatch_count_score = 0.5
unmatch_count_dist = 7
unmatch_match_score = 0.5
unmatch_match_dist = 7

NER_LABEL_SPACE = {} # is dynamically filled to allow different types of NERS to work in explanations

def _mask_mat_select(mask_mat, positions):
    mask_mat = torch.cat([mask_mat[entry[0]][entry[1]].unsqueeze(1).permute(1, 0) for entry in positions], dim=0)
    return mask_mat

def _eq_function(a, b):
    right_hand_side = torch.logical_and(torch.ge(a,b-unmatch_count_dist), torch.le(a,b+unmatch_count_dist)).float()
    right_hand_side = right_hand_side * unmatch_count_score
    
    left_hand_side = torch.eq(a,b).float()

    return torch.maximum(left_hand_side, right_hand_side)

def _gt_function(a, b):
    right_hand_side = torch.gt(a,b-unmatch_count_dist).float()
    right_hand_side = right_hand_side * unmatch_count_score

    left_hand_side = torch.gt(a,b).float()
    return torch.maximum(left_hand_side, right_hand_side)

def _lt_function(a, b):
    right_hand_side = torch.lt(a,b+unmatch_count_dist).float()
    right_hand_side = right_hand_side * unmatch_count_score

    left_hand_side = torch.lt(a,b).float()
    return torch.maximum(left_hand_side, right_hand_side)

def _lte_function(a, b):
    right_hand_side = torch.le(a,b+unmatch_count_dist).float()
    right_hand_side = right_hand_side * unmatch_count_score

    left_hand_side = torch.le(a,b).float()
    return torch.maximum(left_hand_side, right_hand_side)

def _gte_function(a, b):
    right_hand_side = torch.ge(a,b-unmatch_count_dist).float()
    right_hand_side = right_hand_side * unmatch_count_score

    left_hand_side = torch.ge(a,b).float()
    return torch.maximum(left_hand_side, right_hand_side)


Selection = ['NER']

compare_soft={                   #eq mt(more than) lt(less than) nmt(no more than) nlt(no less than)
    'eq' : _eq_function,
    'mt' : _gt_function,
    'lt' : _lt_function,
    'nmt': _lte_function,
    'nlt': _gte_function,
}

#c: Tensor [seqlen seqlen 2 2] : sentence,ner,subj_posi,obj_posi,subj,obj
def get_mid(attr,mask_mat,c):
    subj_posi = c[:,-4:-3]
    obj_posi = c[:,-3:-2]
    seqlen = (c.shape[1]-4)//2
    tokens = c[:,0:seqlen]
    ner = c[:,seqlen:seqlen*2]
    st_posi = torch.minimum(subj_posi, obj_posi)
    ed_posi = subj_posi+obj_posi-st_posi
    positions = torch.cat([st_posi+1, ed_posi-1], dim=1)
    mask = _mask_mat_select(mask_mat, positions)
    ner = ner*mask
    tokens = tokens*mask
    res = ner if attr == "NER" else tokens
    return res

def get_other_posi(POSI,attr,arg,mask_mat,c):
    assert POSI=='Left' or POSI=='Right'
    batch_size = c.shape[0]
    subj_posi = c[:, -4:-3]
    obj_posi = c[:, -3:-2]
    seqlen = (c.shape[1] - 4) // 2
    tokens = c[:, 0:seqlen]
    ner = c[:, seqlen:seqlen * 2]
    posi = obj_posi if arg=='ArgY' else subj_posi

    positions = torch.cat([0*posi, posi-1], dim=1) if POSI=="Left" else torch.cat([posi+1, (0*posi)+seqlen-1], dim=1)

    mask = _mask_mat_select(mask_mat, positions)

    new_ner = mask*ner
    new_tokens = mask*tokens

    res = new_ner if attr == "NER" else new_tokens
    return res

def get_range(attr,arg,range_,mask_mat,c):
    subj_posi = c[:, -4:-3]
    obj_posi = c[:, -3:-2]
    seqlen = (c.shape[1] - 4) // 2
    tokens = c[:, 0:seqlen]
    ner = c[:, seqlen:seqlen * 2]
    posi = obj_posi if arg == 'ArgY' else subj_posi
    res = ner if attr == 'NER' else tokens
    positions = torch.cat([torch.clamp(posi-range_, min=0), torch.clamp(posi+range_, max=seqlen-1)], dim=1)
    mask = _mask_mat_select(mask_mat, positions)
    res = res*mask
    return res

def In(w,seq):
    assert w in NER_LABEL_SPACE
    idx = NER_LABEL_SPACE[w]
    eq = torch.eq(seq, idx)
    total = torch.reshape(torch.sum(eq, dim=1), [1,-1]).type(torch.BoolTensor)
    return total

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
                    bool_list.append(torch.clamp(sum([p(w)(label_mat,keyword_dict,mask_mat)(c) for w in ws])-len(ws)+1, min=0.0))
            else:
                bool_list.append(p(ws)(label_mat,keyword_dict,mask_mat)(c))
        return torch.clamp(sum(bool_list)-len(bool_list)+1, min=0.0)

    if isinstance(ws,tuple):
        if ws[0] in Selection:
            return ps(ws)(label_mat,keyword_dict,mask_mat)(c)
        else:
            return torch.clamp(sum([ps(w)(label_mat,keyword_dict,mask_mat)(c) for w in ws])-len(ws)+1, min=0.0)
    else:
        return ps(ws)(label_mat,keyword_dict,mask_mat)(c)

#function for @Left and @Right
def at_POSI_soft(POSI,ws,arg,label_mat,keyword_dict,mask_mat,c,option=None):
    if isinstance(ws,tuple):
        if ws[0] in Selection:
            assert POSI=='Left' or POSI=='Right' or POSI=='Range'
            if POSI=='Left' or POSI=='Right':
                score_raw = In(ws[1], get_other_posi(POSI,ws[0],arg,mask_mat,c)).float()
                return score_raw
            else:
                score_raw = In(ws[1], get_range(ws[0], arg, option['range'],mask_mat,c)).float()
                return score_raw
        else:
            bool_list = []
            for w in ws:
                bool_list.append(at_POSI_0_soft(POSI,arg,w,label_mat,keyword_dict,mask_mat,c,option))
            return torch.clamp(sum(bool_list)-len(bool_list)+1, min=0.0)
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
        return torch.clamp(sum([at_POSI_0_soft(POSI,arg,ww,label_mat,keyword_dict,mask_mat,c,option) for ww in w])-len(w)+1, min=0.0)

    if option==None:
        option = {'attr': 'word', 'range': -1, 'numAppear':1,'cmp':'nlt','onlyCount':False}  #For now, if onlyCount==True, then attr=='tokens'
    if isinstance(w,tuple):
        assert POSI == 'Left' or POSI == 'Right' or POSI == 'Range'
        if POSI == 'Left' or POSI == 'Right':
            score_raw = In(ws[1], get_other_posi(POSI, ws[0], arg, mask_mat,c)).float()
            return score_raw
        else:
            score_raw = In(ws[1], get_range(ws[0], arg, option['range'], mask_mat,c)).float()
            return score_raw
    else:                                                                                   #For now,option['attr']==tokens if and only if 'right before' is used or onlyCount==True, otherwise ==word
        if option == 'Direct':
            w_split = w.split()
            while '' in w_split:
                w_split.remove('')
            range_ = len(w_split)

            subj_posi = c[:, -4:-3]
            obj_posi = c[:, -3:-2]
            seqlen = (c.shape[1] - 4) // 2

            if arg=='ArgY':
                arg_posi = obj_posi
            else:
                arg_posi = subj_posi

            if POSI == 'Left':
                st = torch.clamp(arg_posi - range_, min=0.0)
                position = torch.cat([st, arg_posi-1], dim=1)
                unmatch_position = torch.cat([torch.clamp(arg_posi - range_ - unmatch_match_dist, min=0.0), arg_posi-1], dim=1)
            elif POSI == 'Right':
                st = arg_posi
                ed = torch.clamp(arg_posi+range_, max=seqlen-1)
                position = torch.cat([st+1, ed], dim=1)
                unmatch_position = torch.cat([st+1, torch.clamp(st + range_ + unmatch_match_dist, max=seqlen-1)], dim=1)
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
                    score_raw = torch.reshape(torch.eq(src-tar, 1).float(), [1,-1])
                    return score_raw
                else:
                    score_raw = torch.reshape(torch.eq(tar-src, 1).float(), [1,-1])
                    return score_raw
            else:
                mask_raw = _mask_mat_select(mask_mat, position).float()
                mask_raw = mask_raw + (_mask_mat_select(mask_mat, unmatch_position).float() - mask_raw) * unmatch_match_score
                return torch.reshape(torch.amax(label_mat[:, :, keyword_dict[w]] * (mask_raw), dim=1), [1, -1])

        if option['range']==-1:
            assert POSI=='Left' or POSI=='Right'
            batch_size = c.shape[0]
            subj_posi = c[:, -4:-3]
            obj_posi = c[:, -3:-2]
            seqlen = (c.shape[1] - 4) // 2
            if arg == 'ArgY':
                arg_posi = obj_posi
            else:
                arg_posi = subj_posi

            if POSI=='Left':
                position = torch.cat([0*arg_posi, arg_posi - 1], dim=1)
                unmatch_position = position
            else:
                position = torch.cat([arg_posi + 1, (0*arg_posi) + (seqlen - 1)], dim=1)
                unmatch_position = position

        else:                                                                               #For now, if range!=-1 then attr == 'tokens'
            subj_posi = c[:, -4:-3]
            obj_posi = c[:, -3:-2]
            seqlen = (c.shape[1] - 4) // 2
            if arg == 'ArgY':
                arg_posi = obj_posi
            else:
                arg_posi = subj_posi

            range_ = option['range']

            if POSI == 'Left':
                st = torch.clamp(arg_posi-range_, min=0)
                position = torch.cat([st, arg_posi-1], dim=1)
                unmatch_position = torch.cat([torch.clamp(arg_posi - range_ - unmatch_match_dist, min=0), arg_posi-1], dim=1)
            elif POSI=='Right':
                st = arg_posi
                ed = torch.clamp(arg_posi+range_, max=seqlen-1)
                position = torch.cat([arg_posi+1, ed], dim=1)
                unmatch_position = torch.cat([st + 1, torch.clamp(st + range_ + unmatch_match_dist, max=seqlen-1)], dim=1)
            else:
                st = torch.clamp(arg_posi-range_, min=0)
                ed = torch.clamp(arg_posi+range_, max=seqlen-1)
                position = torch.cat([st, ed], dim=1)
                unmatch_position = torch.cat([torch.clamp(arg_posi - range_ - unmatch_match_dist, min=0), torch.clamp(st + range_ + unmatch_match_dist, max=seqlen-1)], dim=1)

        if option['onlyCount']:
            score_raw = torch.reshape(compare_soft[option['cmp']](position[:,1]-position[:,0],option['numAppear']).float(), [1,-1])
            return score_raw
        else:
            assert option['cmp']=='nlt' and option['numAppear']==1
            if w in ['ArgX','ArgY']:
                if w=='ArgX':
                    tar = c[:,-4]
                else:
                    tar = c[:,-3]
                score_raw = torch.reshape(torch.logical_and(tar>=position[:,0],tar<=position[:,1]).float(), [1,-1])
                return score_raw
            else:
                mask_raw = _mask_mat_select(mask_mat, position).float()
                mask_raw = mask_raw + (_mask_mat_select(mask_mat, unmatch_position).float() - mask_raw) * unmatch_match_score
                return torch.reshape(torch.amax(label_mat[:, :, keyword_dict[w]] * (mask_raw), dim=1), [1, -1])

#function for @Between

def at_between_soft(w,label_mat,keyword_dict,mask_mat,c,option=None):
    if option==None:
        option = {'attr': 'word', 'numAppear':1,'cmp':'nlt','onlyCount':False}                #For now, if onlyCount==True, then attr=='tokens'
    if isinstance(w,tuple):
        score_raw = In(w[1],get_mid(w[0],mask_mat,c)).float()
        return score_raw
    else:                                                                                   #For now,option['attr']==tokens if and only if  onlyCount==True, otherwise ==word
        if option['onlyCount']:
            score_raw =  torch.reshape(compare_soft[option['cmp']](torch.abs(c[:,-4]-c[:,-3])-1, option['numAppear']).float(), [1,-1])
            return score_raw
        else:
            assert option['cmp'] == 'nlt' and option['numAppear'] == 1
            l_posi = torch.minimum(c[:,-4], c[:,-3])
            g_posi = c[:,-4] + c[:,-3] - l_posi
            if w in ['ArgX','ArgY']:
                if w=='ArgX':
                    tar = c[:,-4]
                else:
                    tar = c[:,-3]
                score_raw = torch.reshape(torch.logical_and(tar>l_posi, tar<g_posi).float(), [1,-1])
                return score_raw
            else:
                l_posi = torch.reshape(l_posi,[-1,1])+1
                g_posi = torch.reshape(g_posi, [-1, 1])-1
                position = torch.cat([l_posi, g_posi], dim=1)
                mask_raw = _mask_mat_select(mask_mat, position).float()
                seqlen = (c.shape[1] - 4) // 2
                return torch.reshape(torch.amax(label_mat[:, :, keyword_dict[w]] * (mask_raw), dim=1), [1, -1])

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
        seqlen = (c.shape[1] - 4) // 2
        score_raw = In(w,c[:,seqlen:seqlen*2]).float()
        return score_raw
    else:
        seqlen = (c.shape[1] - 4) // 2
        return torch.reshape(torch.amax(label_mat[:, :, keyword_dict[w]], dim=1), [1, -1])
        # return w in c.sentence

def at_WordCount_soft(nounNum,nouny,F,label_mat,keyword_dict,mask_mat,c):
    option_1 = {'attr': 'word', 'range': -1, 'numAppear': 1, 'cmp': 'nlt', 'onlyCount': False}
    if isinstance(nouny,tuple):
        option_2 = {'attr': 'tokens','range': -1,'numAppear':sum([len(noun.split()) for noun in nouny]),'cmp': 'eq','onlyCount': True}
        temp_score = torch.clamp(sum([F(noun, option=option_1)(label_mat,keyword_dict,mask_mat)(c) for noun in nouny])-len(nouny)+1, min=0.0)
        final_score = torch.clamp(temp_score + F(nouny[0],option=option_2)(label_mat,keyword_dict,mask_mat)(c)-1, min=0.0)
        return final_score
    else:
        option_2 = {'attr':'tokens','range':-1,'numAppear':len(nouny.split()),'cmp':'eq','onlyCount':True}
        temp_score = F(nouny, option=option_1)(label_mat,keyword_dict,mask_mat)(c)+F(nouny, option=option_2)(label_mat,keyword_dict,mask_mat)(c)-1
        final_score = torch.clamp(temp_score, min=0.0)
        return final_score