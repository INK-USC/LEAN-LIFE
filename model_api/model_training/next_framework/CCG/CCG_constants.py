"""
    Constant file holding important grammar information
"""
import pathlib
import sys
import torch
PATH_TO_PARENT = str(pathlib.Path(__file__).parent.absolute()) + "/"
sys.path.append(PATH_TO_PARENT)
from CCG import strict_grammar_functions as gram_f
from CCG import soft_grammar_functions as soft_gram_f

SPACE = "SPACE"
PUNCT = "PUNCT"
PHRASE = "PHRASE"

RAW_GRAMMAR =''':- S,NP,N,PP
        VP :: S\\NP
        Det :: NP/N
        Adj :: N/N
        arg => NP {None}
        $And => var\\.,var/.,var {\\x y.'@And'(x,y)}
        $Or => var\\.,var/.,var {\\x y.'@Or'(x,y)}
        $Not => (S\\NP)\\(S\\NP) {None}
        $Not => (S\\NP)/(S\\NP) {None}
        $All => NP/N {None}
        $All => NP {None}
        $All => NP/NP {None}
        $Any => NP/N {None}
        $None => N {None}
        $Is => (S\\NP)/PP {\\y x.'@Is'(x,y)}   # word 'a' occurs between <S> and <O>
        $Is => (S\\NP)\\PP {\\y x.'@Is'(x,y)}  # between <S> and <O> occurs word 'a'
        $Is => (S\\PP)\\NP {\\x y.'@Is'(x,y)}  # between <S> and <O> word 'a' occurs
        $Exists => S\\NP/PP {\\y x.'@Is'(x,y)}
        #$Exists => S\\NP {None}
        $Int => Adj {None} #There are no words between <S> and <O>
        $AtLeastOne => NP/N {None}
        
        $LessThan => PP/PP/N {\\x y.'@LessThan'(y,x)} #There are less than 3 words between <S> and <O>   
        $AtMost => PP/PP/N {\\x y.'@AtMost'(y,x)} #There are at most 3 words between <S> and <O>
        $AtLeast => PP/PP/N {\\x y.'@AtLeast'(y,x)} #same as above
        $MoreThan => PP/PP/N {\\x y.'@MoreThan'(y,x)} #same as above
        
        $LessThan => PP/N {\\x.'@LessThan1'(y,x)} #number of words between X and Y is less than 7.
        $AtMost => PP/N {\\x.'@AtMost1'(y,x)} 
        $AtLeast => PP/N {\\x.'@AtLeast1'(y,x)}   #same as above
        $MoreThan => PP/N {\\x.'@MoreThan1'(y,x)} #same as above

        $In => PP/NP {\\x.'@In0'(x)} #never called?
        $Separator => var\\.,var/.,var {\\x y.'@And'(x,y)} #connection between two words
        $EachOther => N {None}
        $Token => N {\\x.'@Word'(x)}
        $Word => NP/N {\\x.'@Word'(x)}
        $Word => NP/NP {\\x.'@Word'(x)}
        
        $Word => N {'tokens'} #There are no more than 3 words between <S> and <O>
        $Word => NP {'tokens'} #There are no more than 3 words between <S> and <O>
        
        $Char => N {None} #same as above
        $StartsWith => S\\NP/NP {\\y x.'@StartsWith'(x,y)}
        $EndsWith => S\\NP/NP {\\y x.'@EndsWith'(x,y)}
        $Left => PP/NP {\\x.'@Left0'(x)} # the word 'a' is before <S>
        $Left => (S\\NP)/NP {\\y x.'@Left'(y,x)}  #Precedes
        $Right => PP/NP {\\x.'@Right0'(x)}# the word 'a' ia after <S>
        $Right => (S\\NP)/NP {\\y x.'@Right'(y,x)} 
        $Within => PP/PP/N {\\x y.'@AtMost'(y,x)} #Does Within has other meaning.
        $Sentence => NP {'Sentence'}
        $Contains => S\\NP/NP {\\x y. '@In1'(y, x)} #y contains x
        $In => S\\NP/NP {\\x y. '@In1'(x, y)} # y is in x
        $Between => (S/S)/NP {\\x y.'@between'(x,y)}
        $Between => S/NP {\\x.'@between'(x)}
        $Between => PP/NP {\\x.'@between'(x)}
        $Between => (NP\\NP)/NP {\\x y.'@between'(x,y)}
        
        $PersonNER => NP {'@PER'}
        $LocationNER => NP {'@LOC'}
        $DateNER => NP {'@Date'}
        $NumberNER => NP {'@Num'}
        $OrganizationNER => NP {'@Org'}
        $NorpNER => NP {'@Norp'}
        $ArgX => NP {'ArgX'}
        $ArgY => NP {'ArgY'}
        $that => NP/N {None}
        $Apart => (S/PP)\\NP {None}
        $Direct => PP/PP {\\x.'@Direct'(x)} # the word 'a' is right before <S>   
        $Direct => (S\\NP)/PP {\\y x.'@Is'(x,'@Direct'(y))}
        $Last => Adj {None}
        $There => NP {'There'}
        $By => S\\NP\\PP/NP {\\z f x.'@By'(x,f,z)} #precedes sth by 10 chatacters       
        $By => (S\\NP)\\PP/(PP/PP) {\\F x y.'@Is'(y,F(x))} #precedes sth by no more than10 chatacters        
        $By => PP\\PP/(PP/PP) {\\F x. F(x)} #occurs before by no...
        
        $Numberof => NP/PP/NP {\\x F.'@NumberOf'(x,F)}
        
        $Of => PP/NP {\\x.'@Range0'(x)} # the word 'x' is at most 3 words of Y     
        $Of => NP/NP {\\x.x} #these two are designed to solve problems like $Is $Left $Of and $Is $Left
        $Of => N/N {\\x.x}
        $Char => NP/N {None}
        $ArgX => N {'ArgX'}
        $ArgY => N {'ArgY'}
        $Link => (S\\NP)/NP {\\x y.'@Is'(y,'@between'(x))}
        $SandWich => (S\\NP)/NP {\\x y.'@Is'(x,'@between'(y))}
        $The => N/N {\\x.x}
        $The =>NP/NP {\\x.x}
        '''

TOKEN_TERMINAL_MAP = {
    'arg': ['$Arg'],
    'argument': ['$Arg'],
    'the': ['$The'],
    'true': ['$True'],
    'correct': ['$True'],
    'false': ['$False'],
    'incorrect': ['$False'],
    'wrong': ['$False'],
    'and': ['$And'],
    'but': ['$And'],
    'or': ['$Or'],
    'nor': ['$Or'],
    'not': ['$Not'],
    'n"t': ['$Not'],
    'n\'t': ['$Not'],
    'all': ['$All'],
    'both': ['$All'],
    'any': ['$Any'],
    'a': ['$Any', '$AtLeastOne'],
    'one of': ['$Any'],
    'none': ['$None'],
    'not any': ['$None'],
    'neither': ['$None'],
    'no': ['$None', '$Int'],
    'is': ['$Is'],
    'are': ['$Is'],
    'be': ['$Is'],
    'comes': ['$Is'],
    'come': ['$Is'],
    'appears': ['$Is'],
    'appear': ['$Is'],
    'as': ['$Is'],
    'occurs': ['$Is'],
    'occur': ['$Is'],
    'is stated': ['$Is'],
    'is found': ['$Is'],
    'said': ['$Is'],
    'is identified': ['$Is'],
    'are identified': ['$Is'],
    'is used': ['$Is'],
    'is placed': ['$Is'],
    'exist': ['$Exists'],
    'exists': ['$Exists'],
    'immediately': ['$Direct'],
    'right': ['$Direct', '$Right'],
    'directly': ['$Direct'],
    'last': ['$Last'],
    'final': ['$Last'],
    'ending': ['$Last'],
    'another': ['$AtLeastOne'],
    'because': ['$Because'],
    'since': ['$Because'],
    'if': ['$Because'],
    'equal': ['$Equals'],
    'equals': ['$Equals'],
    '=': ['$Equals'],
    '==': ['$Equals'],
    'same as': ['$Equals'],
    'same': ['$Equals'],
    'identical': ['$Equals'],
    'exactly': ['$Equals'],
    'different than': ['$NotEquals'],
    'different': ['$NotEquals'],
    'less than': ['$LessThan'],
    'smaller than': ['$LessThan'],
    '<': ['$LessThan'],
    'at most': ['$AtMost'],
    'no larger than': ['$AtMost'],
    'less than or equal': ['$AtMost'],
    'within': ['$AtMost', '$Within'],
    'no more than': ['$AtMost'],
    '<=': ['$AtMost'],
    'at least': ['$AtLeast'],
    'no less than': ['$AtLeast'],
    'no smaller than': ['$AtLeast'],
    'greater than or equal': ['$AtLeast'],
    '>=': ['$AtLeast'],
    'more than': ['$MoreThan'],
    'greater than': ['$MoreThan'],
    'larger than': ['$MoreThan'],
    '>': ['$MoreThan'],
    'is in': ['$In'],
    'in': ['$In'],
    'contains': ['$Contains'],
    'contain': ['$Contains'],
    'containing': ['$Contains'],
    'include': ['$Contains'],
    'includes': ['$Contains'],
    'says': ['$Contains'],
    'states': ['$Contains'],
    'mentions': ['$Contains'],
    'mentioned': ['$Contains'],
    'referred': ['$Contains'],
    'refers': ['$Contains'],
    'is referring to': ['$Contains'],
    ',': ['$Separator', []],
    '/': ['$Separator'],
    '\'s': ['$Possessive'],
    'number': ['$Count', '$NumberNER'],
    'length': ['$Count'],
    'count': ['$Count'],
    'pair': ['$Tuple'],
    'tuple': ['$Tuple'],
    'they': ['$ArgXListAnd', '$ArgXListAnd'],
    'them': ['$ArgXListAnd', '$ArgXListAnd'],
    'entities': ['$ArgXListAnd'],
    'eachother': ['$EachOther'],
    'each other': ['$EachOther'],
    'token': ['$Token', '$Word'],
    'word': ['$Word'],
    'words': ['$Word'],
    'term': ['$Word'],
    'terms': ['$Word'],
    'tokens': ['$Word'],
    'phrase': ['$Word'],
    'phrases': ['$Word'],
    'string': ['$Word'],
    'character': ['$Char'],
    'characters': ['$Char'],
    'letter': ['$Char'],
    'letters': ['$Char'],
    'upper': ['$Upper'],
    'uppercase': ['$Upper'],
    'upper case': ['$Upper'],
    'all caps': ['$Upper'],
    'all capitalized': ['$Upper'],
    'lower': ['$Lower'],
    'lowercase': ['$Lower'],
    'lower case': ['$Lower'],
    'capital': ['$Capital'],
    'capitals': ['$Capital'],
    'capitalized': ['$Capital'],
    'starts with': ['$StartsWith'],
    'start with': ['$StartsWith'],
    'starting with': ['$StartsWith'],
    'ends with': ['$EndsWith'],
    'end with': ['$EndsWith'],
    'ending with': ['$EndsWith'],
    'to the left of': ['$Left'],
    'left': ['$Left'],
    'in front of': ['$Left'],
    'before': ['$Left'],
    'precedes': ['$Left'],
    'preceding': ['$Left'],
    'followed by': ['$Left'],
    'to the right of': ['$Right'],
    'behind': ['$Right'],
    'after': ['$Right'],
    'preceded by': ['$Right'],
    'follows': ['$Right'],
    'following': ['$Right'],
    'next': ['$Within'],
    'apart': ['$Apart'],
    'away': ['$Apart'],
    'sentence': ['$Sentence'],
    'tweet' : ['$Sentence'],
    'text': ['$Sentence'],
    'it': ['$Sentence'],
    'between': ['$Between'],
    'in between': ['$Between'],
    'sandwiched': ['$Between'],
    'enclosed': ['$Between'],
    'Between': ['$Between'],
    'admist': ['$Between'],
    'in the middle of': ['$Between'],
    'person': ['$PersonNER'],
    'people': ['$PersonNER'],
    'location': ['$LocationNER'],
    'locations': ['$LocationNER'],
    'place': ['$LocationNER'],
    'places': ['$LocationNER'],
    'date': ['$DateNER'],
    'dates': ['$DateNER'],
    'numbers': ['$NumberNER'],
    'organization': ['$OrganizationNER'],
    'organizations': ['$OrganizationNER'],
    'company': ['$OrganizationNER'],
    'companies': ['$OrganizationNER'],
    'agency': ['$OrganizationNER'],
    'agencies': ['$OrganizationNER'],
    'institution': ['$OrganizationNER'],
    'institutions': ['$OrganizationNER'],
    'political': ['$NorpNER'],
    'politician': ['$NorpNER'],
    'religious': ['$NorpNER'],
    'x': ['$ArgX'],
    '<s>': ['$ArgX'],
    'subj': ['$ArgX'],
    'subject': ['$ArgX'],
    'y': ['$ArgY'],
    '<o>': ['$ArgY'],
    'obj': ['$ArgY'],
    'object': ['$ArgY'],
    'there': ['$There'],
    'by': ['$By'],
    'which': ['$Which'],
    'the number of': ['$Numberof'],
    'of': ['$Of'],
    'links': ['$Link'],
    'link': ['$Link'],
    'connects': ['$Link'],
    'connect': ['$Link'],
    'sandwich': ['$SandWich'],
    'sandwiches': ['$SandWich'],
}

PHRASE_TERMINAL_MAP = {
    'is in': '$In',
    'one of': '$Any',
    'not any': '$None',
    'is stated': '$Is',
    'is found': '$Is',
    'is identified': '$Is',
    'are identified': '$Is',
    'is used': '$Is',
    'is placed': '$Is',
    'same as': '$Equals',
    'different than': '$NotEquals',
    'less than': '$LessThan',
    'smaller than': '$LessThan',
    'at most': '$AtMost',
    'no larger than': '$AtMost',
    'less than or equal': '$AtMost',
    'no more than': '$AtMost',
    'at least': '$AtLeast',
    'no less than': '$AtLeast',
    'no smaller than': '$AtLeast',
    'greater than or equal': '$AtLeast',
    'more than': '$MoreThan',
    'greater than': '$MoreThan',
    'larger than': '$MoreThan',
    'is referring to': '$Contains',
    'each other': '$EachOther',
    'upper case': '$Upper',
    'all caps': '$Upper',
    'all capitalized': '$Upper',
    'lower case': '$Lower',
    'starts with': '$StartsWith',
    'start with': '$StartsWith',
    'starting with': '$StartsWith',
    'ends with': '$EndsWith',
    'end with': '$EndsWith',
    'ending with': '$EndsWith',
    'to the left of': '$Left',
    'in front of': '$Left',
    'followed by': '$Left',
    'to the right of': '$Right',
    'preceded by': '$Right',
    'in between': '$Between',
    'in the middle of': '$Between',
    'the number of': '$Numberof'
}

PHRASE_ARRAY = list(PHRASE_TERMINAL_MAP.keys())
PHRASE_VALUES = set(list(PHRASE_TERMINAL_MAP.values()))

NER_TERMINAL_TO_EXECUTION_TUPLE = {
    "@LOC": ('NER','LOCATION'),
    "@Date":('NER','DATE'),
    "@Num":('NER','NUMBER'),
    "@Org":('NER','ORGANIZATION'),
    "@Norp":('NER','Norp'),
    '@PER':('NER','PERSON')
}

STRICT_MATCHING_OPS = {
    ".root"       : lambda xs: lambda c: all([x(c) for x in xs]) if type(xs) == tuple else xs(c),
    "@Word"       : lambda x: x,
    "@Is"         : lambda ws,p: lambda c: gram_f.IsFunc(ws,p,c),
    "@between"    : lambda a: lambda w,option=None: lambda c: gram_f.at_between(w,c,option,a),
    "@In0"        : lambda arg: lambda w: lambda c: gram_f.at_In0(arg,w,c),
    "@In1"        : lambda arg,w: lambda c: gram_f.at_In0(arg,w,c),
    "@And"        : lambda x,y: gram_f.merge(x,y),
    "@Num"        : lambda x,y: {'attr':y,'num':int(x)},
    "@LessThan"   : lambda funcx,nouny: lambda w: lambda c: gram_f.at_lessthan(funcx,nouny,w,c),
    "@AtMost"     : lambda funcx,nouny: lambda w: lambda c: gram_f.at_atmost(funcx,nouny,w,c),
    "@AtLeast"    : lambda funcx,nouny: lambda w: lambda c: gram_f.at_atleast(funcx,nouny,w,c),
    "@MoreThan"   : lambda funcx,nouny: lambda w: lambda c: gram_f.at_morethan(funcx,nouny,w,c),
    "@WordCount"  : lambda nounNum,nouny,F: lambda useless: lambda c: gram_f.at_WordCount(nounNum,nouny,F,c),
 
    "@NumberOf"   : lambda x,f: [x,f],
    "@LessThan1"  : lambda nounynum: lambda x: lambda c: gram_f.at_lessthan(x[1],{'attr':x[0],"num":int(nounynum)},'There',c),
    "@AtMost1"    : lambda nounynum: lambda x: lambda c: gram_f.at_atmost(x[1],{'attr':x[0],"num":int(nounynum)},'There',c),
    "@AtLeast1"   : lambda nounynum: lambda x: lambda c: gram_f.at_atleast(x[1],{'attr':x[0],"num":int(nounynum)},'There',c),
    "@MoreThan1"  : lambda nounynum: lambda x: lambda c: gram_f.at_morethan(x[1],{'attr':x[0],"num":int(nounynum)},'There',c),
 
    #By
    "@By"         : lambda x,f,z: lambda c: f(x,{'attr': z['attr'], 'range': z['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': False})(c),
    # is xx Arg 
    "@Left0"      : lambda arg: lambda w,option=None: lambda c: gram_f.at_POSI_0('Left',arg,w,c,option),
    "@Right0"     : lambda arg: lambda w,option=None: lambda c: gram_f.at_POSI_0('Right',arg,w,c,option),
    "@Range0"     : lambda arg: lambda w,option=None: lambda c: gram_f.at_POSI_0('Range',arg,w,c,option),
 
    "@Left"       : lambda arg,ws,option=None: lambda c: gram_f.at_POSI('Left',ws,arg,c,option),
    "@Right"      : lambda arg,ws,option=None: lambda c: gram_f.at_POSI('Right',ws,arg,c,option),

    "@Direct"     : lambda func: lambda w: lambda c: func(w,'Direct')(c),

    "@StartsWith" : lambda x,y: lambda c: c.with_(x[-1],'starts',y),
    "@EndsWith"   : lambda x,y: lambda c: c.with_(x[-1],'ends',y),
}

SOFT_MATCHING_OPS = {
    ".root": lambda xs: lambda label_mat,keyword_dict,mask_mat:lambda c: torch.max(torch.tensor(sum([x(label_mat,keyword_dict,mask_mat)(c) for x in xs])-len(xs)+1).to(device),torch.tensor(0.0).to(device)) if type(xs) == tuple else xs(label_mat,keyword_dict,mask_mat)(c),
    "@Word": lambda x: x,
    "@Is": lambda ws, p: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.IsFunc_soft(ws, p,label_mat,keyword_dict,mask_mat, c),
    "@between": lambda a: lambda w, option=None: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_between_soft(w, label_mat,keyword_dict,mask_mat,c, option),
    "@And": lambda x, y: soft_gram_f.merge_soft(x, y),
    "@Num": lambda x, y: {'attr': y, "num": int(x)},
    "@LessThan": lambda funcx, nouny: lambda w: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_lessthan_soft(funcx, nouny, w, label_mat,keyword_dict,mask_mat,c),
    "@AtMost": lambda funcx, nouny: lambda w: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_atmost_soft(funcx, nouny, w, label_mat,keyword_dict,mask_mat,c),
    "@AtLeast": lambda funcx, nouny: lambda w: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_atleast_soft(funcx, nouny, w, label_mat,keyword_dict,mask_mat,c),
    "@MoreThan": lambda funcx, nouny: lambda w: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_morethan_soft(funcx, nouny, w, label_mat,keyword_dict,mask_mat,c),
    "@WordCount": lambda nounNum, nouny, F:lambda useless: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_WordCount_soft(nounNum,nouny,F,label_mat,keyword_dict,mask_mat,c),

    "@NumberOf": lambda x, f: [x, f],
    "@LessThan1": lambda nounynum: lambda x: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_lessthan_soft(x[1], {'attr': x[0], "num": int(nounynum)}, 'There',label_mat,keyword_dict,mask_mat,c),
    "@AtMost1": lambda nounynum: lambda x: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_atmost_soft(x[1], {'attr': x[0], "num": int(nounynum)}, 'There', label_mat,keyword_dict,mask_mat,c),
    "@AtLeast1": lambda nounynum: lambda x: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_atleast_soft(x[1], {'attr': x[0], "num": int(nounynum)}, 'There',label_mat,keyword_dict,mask_mat,c),
    "@MoreThan1": lambda nounynum: lambda x: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_morethan_soft(x[1], {'attr': x[0], "num": int(nounynum)}, 'There',label_mat,keyword_dict,mask_mat,c),

    "@In0": lambda arg: lambda w: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_In0_soft(arg, w, label_mat,keyword_dict,mask_mat,c),
    "@In1": lambda arg,w: lambda label_mat,keyword_dict,mask_mat: lambda c: soft_gram_f.at_In0_soft(arg, w, label_mat,keyword_dict,mask_mat,c),
    "@By": lambda x, f, z: lambda label_mat,keyword_dict,mask_mat:lambda c: f(x, {'attr': z['attr'], 'range': z['num'], 'numAppear': 1, 'cmp': 'nlt','onlyCount': False})(label_mat,keyword_dict,mask_mat)(c),

    "@Left0": lambda arg: lambda w, option=None: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_POSI_0_soft('Left', arg, w, label_mat,keyword_dict,mask_mat,c, option),

    "@Right0": lambda arg: lambda w, option=None: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_POSI_0_soft('Right', arg, w, label_mat,keyword_dict,mask_mat,c, option),

    "@Range0": lambda arg: lambda w, option=None: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_POSI_0_soft('Range', arg, w, label_mat,keyword_dict,mask_mat,c, option),

    "@Left": lambda arg, ws, option=None: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_POSI_soft('Left', ws, arg, label_mat,keyword_dict,mask_mat,c, option),
    "@Right": lambda arg, ws, option=None: lambda label_mat,keyword_dict,mask_mat:lambda c: soft_gram_f.at_POSI_soft('Right', ws, arg, label_mat,keyword_dict,mask_mat,c, option),

    "@Direct": lambda func: lambda w: lambda label_mat,keyword_dict,mask_mat:lambda c: func(w, 'Direct')(label_mat,keyword_dict,mask_mat)(c)
}