"""WVS questions, demographic groups, and prompts for Experiment 2.

The build keeps a seeded one-third of countries and a seeded one-third
of items. Both draws use seed 42 and are stored with the processed
targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mode_collapse.surveys.gss import (
    ANSWER_LETTERS,
    SEED,
    build_chat_messages,
    build_question_block,
)

DATASET = "wvs"
SUBSET_FRACTION_DENOM = 3
CELL_VARS = ["country", "year", "urbrural", "age", "sex", "education"]
ORIGINAL_7_WVS_VARS = ["Q57", "Q6", "Q46", "Q199", "Q71", "Q164", "Q240"]
ITEM_QUESTIONS = {
    "Q1": (
        "For each of the following aspects, indicate how important it is in your life. "
        "Would you say it is very important, rather important, not very important or no"
        "t important at all? – Family"
    ),
    "Q2": (
        "For each of the following aspects, indicate how important it is in your life. "
        "Would you say it is very important, rather important, not very important or no"
        "t important at all? – Friends"
    ),
    "Q3": (
        "For each of the following aspects, indicate how important it is in your life. "
        "Would you say it is very important, rather important, not very important or no"
        "t important at all? – Leisure time"
    ),
    "Q4": (
        "For each of the following aspects, indicate how important it is in your life. "
        "Would you say it is very important, rather important, not very important or no"
        "t important at all? – Politics"
    ),
    "Q5": (
        "For each of the following aspects, indicate how important it is in your life. "
        "Would you say it is very important, rather important, not very important or no"
        "t important at all? – Work"
    ),
    "Q6": (
        "For each of the following aspects, indicate how important it is in your life. "
        "Would you say it is very important, rather important, not very important or no"
        "t important at all? – Religion"
    ),
    "Q27": (
        "For each of the following statements I read out, can you tell me how much you "
        "agree with each. Do you agree strongly, agree, disagree, or disagree strongly?"
        " - One of my main goals in life has been to make my parents proud"
    ),
    "Q28": (
        "For each of the following statements I read out, can you tell me how much you "
        "agree with each. Do  you agree strongly, agree, disagree, or disagree strongly"
        "? -  When a mother works for pay, the children suffer"
    ),
    "Q29": (
        "For each of the following statements I read out, can you tell me how much you "
        "agree with each. Do you agree strongly, agree, disagree, or disagree strongly?"
        " - On the whole, men make better political leaders than women do"
    ),
    "Q30": (
        "For each of the following statements I read out, can you tell me how much you "
        "agree with each. Do you agree strongly, agree, disagree, or disagree strongly?"
        " - A university education is more important for a boy than for a girl"
    ),
    "Q31": (
        "For each of the following statements I read out, can you tell me how strongly "
        "you agree or disagree with each. Do you strongly agree, agree, disagree, or st"
        "rongly disagree? - On the whole, men make better business executives than wome"
        "n do"
    ),
    "Q32": (
        "For each of the following statements I read out, can you tell me how much you "
        "agree with each. Do  you agree strongly, agree, disagree, or  disagree  strong"
        "ly? -  Being a housewife is just as fulfilling as working for pay"
    ),
    "Q33": (
        "Do you agree, disagree or neither agree nor disagree with the following statem"
        "ents? - When jobs are scarce, men should have more right to a job than women"
    ),
    "Q34": (
        "Do you agree, disagree or neither agree nor disagree with the following statem"
        "ents? - When jobs are scarce, employers should give priority to people of this"
        " country  over immigrants."
    ),
    "Q35": (
        "Do you agree, disagree or neither agree nor disagree with the following statem"
        "ents? - If a woman earns more money than her husband, it's almost certain to c"
        "ause problems"
    ),
    "Q36": (
        "Do you agree, disagree or neither agree nor disagree with the following statem"
        "ents? - Homosexual couples are as good parents as other couples"
    ),
    "Q37": (
        "Do you agree, disagree or neither agree nor disagree with the following statem"
        "ents? – It is a duty towards society to have children"
    ),
    "Q38": (
        "Do you agree, disagree or neither agree nor disagree with the following statem"
        "ents? - Adult children have the duty to provide long-term care for their paren"
        "ts"
    ),
    "Q39": (
        "Do you agree or disagree with the following statements? - People who don’t wor"
        "k turn lazy"
    ),
    "Q40": (
        "Do you agree or disagree with the following statements? – Work is a duty towar"
        "ds society"
    ),
    "Q41": (
        "Do you agree or disagree with the following statements? – Work should always c"
        "ome first, even if it means less spare time"
    ),
    "Q42": (
        "On this card are three basic kinds of attitudes concerning the society we live"
        " in. Please choose the one which best describes your own opinion."
    ),
    "Q43": (
        "I'm going to read out a list of various changes in our way of life that might "
        "take place in the near future. Please tell me for each one, if it were to happ"
        "en, whether you think it would be a good thing, a bad thing, or don't you mind"
        "? – Less importance placed on work in our lives"
    ),
    "Q44": (
        "I'm going to read out a list of various changes in our way of life that might "
        "take place in the near future. Please tell me for each one, if it were to happ"
        "en, whether you think it would be a good thing, a bad thing, or don't you mind"
        "? – More emphasis on the development of technology"
    ),
    "Q45": (
        "I'm going to read out a list of various changes in our way of life that might "
        "take place in the near future. Please tell me for each one, if it were to happ"
        "en, whether you think it would be a good thing, a bad thing, or don't you mind"
        "? – Greater respect for authority"
    ),
    "Q46": "Taking all things together, would you say you are:",
    "Q47": (
        "All in all, how would you describe your state of health these days? Would you "
        "say it is..."
    ),
    "Q48": (
        "Some people feel they have completely free choice and control over their lives"
        ", while other people feel that what they do has no real effect on what happens"
        ' to them. Please use this scale where 1 means "none at all" and 10 means "a'
        ' great deal" to indicate how much freedom of choice and control you feel you '
        "have over the way your life turns out."
    ),
    "Q49": "All things considered, how satisfied are you with your life as a whole these days?",
    "Q50": (
        "How satisfied are you with the financial situation of your household? If '1' m"
        "eans you are completely dissatisfied on this scale, and '10' means you are com"
        "pletely satisfied, where would you put your satisfaction with your household's"
        " financial situation?"
    ),
    "Q56": (
        "Comparing your standard of living with your parents’ standard of living when t"
        "hey were about your age, would you say that you are better off, worse off or a"
        "bout the same?"
    ),
    "Q57": (
        "Generally speaking, would you say that most people can be trusted or that you "
        "need to be very careful in dealing with people?"
    ),
    "Q58": (
        "I ‘d like to ask you how much you trust people from various groups. Could you "
        "tell me for each whether you trust people from this group completely, somewhat"
        ", not very much or not at all? Your family"
    ),
    "Q59": (
        "I ‘d like to ask you how much you trust people from various groups. Could you "
        "tell me for each whether you trust people from this group completely, somewhat"
        ", not very much or not at all? Your neighborhood"
    ),
    "Q60": (
        "I ‘d like to ask you how much you trust people from various groups. Could you "
        "tell me for each whether you trust people from this group completely, somewhat"
        ", not very much or not at all? People you know personally"
    ),
    "Q61": (
        "I ‘d like to ask you how much you trust people from various groups. Could you "
        "tell me for each whether you trust people from this group completely, somewhat"
        ", not very much or not at all? People you meet for the first time"
    ),
    "Q62": (
        "I ‘d like to ask you how much you trust people from various groups. Could you "
        "tell me for each whether you trust people from this group completely, somewhat"
        ", not very much or not at all? People of another religion"
    ),
    "Q63": (
        "I ‘d like to ask you how much you trust people from various groups. Could you "
        "tell me for each whether you trust people from this group completely, somewhat"
        ", not very much or not at all? People of another nationality"
    ),
    "Q64": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? The Churc"
        "hes (mosque, temple etc.)"
    ),
    "Q65": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? The armed"
        " forces"
    ),
    "Q66": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? The press"
    ),
    "Q67": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? Televisio"
        "n"
    ),
    "Q68": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? Labour Un"
        "ions"
    ),
    "Q69": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? The polic"
        "e"
    ),
    "Q70": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? The court"
        "s"
    ),
    "Q71": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? The gover"
        "nment (in your nation’s capital)"
    ),
    "Q72": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? Political"
        " Parties"
    ),
    "Q73": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? Parliamen"
        "t"
    ),
    "Q74": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? The Civil"
        " Services"
    ),
    "Q75": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? Universit"
        "ies"
    ),
    "Q76": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? Elections"
    ),
    "Q77": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? Major Com"
        "panies"
    ),
    "Q78": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? Banks"
    ),
    "Q79": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? Environme"
        "ntal organizations"
    ),
    "Q80": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? Women´s o"
        "rganizations"
    ),
    "Q81": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not  very much  confidence or  none at  all? Chari"
        "table or  humanitarian organizations"
    ),
    "Q82": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? <major re"
        "gional organization>"
    ),
    "Q83": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? The Unite"
        "d Nations"
    ),
    "Q84": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? The Inter"
        "national Monetary Fund (IMF)"
    ),
    "Q85": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or  none at  all? Interna"
        "tional Criminal Court (ICC)"
    ),
    "Q86": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence,  not  very  much  confidence or  none  at  all?  T"
        "he  North  Atlantic  Treaty Organization (NATO)"
    ),
    "Q87": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? The World"
        " Bank"
    ),
    "Q88": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? The World"
        " Health Organization (WHO)"
    ),
    "Q89": (
        "I am going to name a number of organizations. For each one, could you tell me "
        "how much confidence  you have in  them: is  it  a  great deal of  confidence, "
        "quite a  lot of confidence, not very much confidence or none at all? The World"
        " Trade Organization (WTO)"
    ),
    "Q90": (
        "Nowadays there’s a lot of talk about international organizations. People somet"
        "imes say that international organizations should prioritize improving people’s"
        " lives, even if this may  imply that  decisions are  not  made  democratically"
        ". What  do  you  think should international organizations prioritize, being ef"
        "fective or being democratic? If your views are somewhat mixed, choose the appr"
        "opriate number in between."
    ),
    "Q106": (
        "Now I'd like you to tell me your views on various issues. How would you place "
        "your views on this scale? 1 means you agree completely with the statement on t"
        "he left; 10 means you agree completely with the statement on the  right; and i"
        "f  your views fall somewhere in between, you can choose any number in between."
    ),
    "Q107": (
        "Now I'd like you to tell me your views on various issues. How would you place "
        "your views on this scale? 1 means you agree completely with the statement on t"
        "he left; 10 means you agree completely with the statement on the  right; and i"
        "f  your views fall somewhere in between, you can choose any number in between."
    ),
    "Q108": (
        "Now I'd like you to tell me your views on various issues. How would you place "
        "your views on this scale? 1 means you agree completely with the statement on t"
        "he left; 10 means you agree completely with the statement on the  right; and i"
        "f  your views fall somewhere in between, you can choose any number in between."
    ),
    "Q109": (
        "Now I'd like you to tell me your views on various issues. How would you place "
        "your views on this scale? 1 means you agree completely with the statement on t"
        "he left; 10 means you agree completely with the statement on the  right; and i"
        "f  your views fall somewhere in between, you can choose any number in between."
    ),
    "Q110": (
        "Now I'd like you to tell me your views on various issues. How would you place "
        "your views on this scale? 1 means you agree completely with the statement on t"
        "he left; 10 means you agree completely with the statement on the  right; and i"
        "f  your views fall somewhere in between, you can chose any number in between."
    ),
    "Q111": (
        "Here are two statements people sometimes make when discussing the environment "
        "and economic growth. Which of them comes closer to your own point of view? A. "
        "Protecting the environment should be given priority, even if it causes slower "
        "economic growth and some loss of jobs B. Economic growth and creating jobs sho"
        "uld be the top priority, even if the environment suffers to some extent"
    ),
    "Q112": (
        "Now I'd like you to tell me your views on corruption – when people pay a bribe"
        ", give a gift or do a favor to other people in order to get the things they ne"
        "ed done or the services they need. How would you place your views on corruptio"
        "n in your country on a 10 -point scale where “1” means “there is no corruption"
        " in this country” and “10” means “there is abundant corruption in this country"
        "”. If your views are somewhat mixed, choose the appropriate number in between."
    ),
    "Q113": (
        "Among  the  following groups of  people, how  many  do  you  believe are  invo"
        "lved in corruption? Tell me for each group if you believe it is none of them, "
        "few of them, most of them or all of them? State authorities"
    ),
    "Q114": (
        "Among  the  following groups of  people, how  many  do  you  believe are  invo"
        "lved in corruption? Tell me for each group if you believe it is none of them, "
        "few of them, most of them or all of them? Business executives"
    ),
    "Q115": (
        "Among  the  following groups of  people, how  many  do  you  believe are  invo"
        "lved in corruption? Tell me for each group if you believe it is none of them, "
        "few of them, most of them or all of them? Local authorities"
    ),
    "Q116": (
        "Among  the  following groups of  people, how  many  do  you  believe are  invo"
        "lved in corruption? Tell me for each group if you believe it is none of them, "
        "few of them, most of them or all of them? Civil service providers (police, jud"
        "iciary, civil servants, doctors, teachers)"
    ),
    "Q117": (
        "Among  the  following groups of  people, how  many  do  you  believe are  invo"
        "lved in corruption? Tell me for each group if you believe it is none of them, "
        "few of them, most of them or all of them? Journalists and media"
    ),
    "Q118": (
        "officials/service providers in order to get services We want to know about you"
        "r experience with local officials and service providers, like police officers,"
        " lawyers, doctors, teachers and civil servants in your community. How often do"
        " you think ordinary people like yourself or people from your neighbourhood hav"
        "e to pay a bribe, give a gift or do a favor to these people in order to get th"
        "e services you need? Does it happen never, rarely, frequently or always?"
    ),
    "Q119": (
        "Can you tell me how strongly you agree or disagree with the following statemen"
        "t: On the whole, women are less corrupt than men"
    ),
    "Q120": (
        "How high is the risk in this country to be held accountable for giving or rece"
        "iving a bribe, gift or favour in return for public service? To indicate your o"
        "pinion, use a 10 - point scale where “1” means “no risk at all” and “10” means"
        " “very high risk”."
    ),
    "Q130": (
        "From your point of view, what have been the effects of immigrants on the devel"
        "opment of this country? - Increase the risks of terrorism"
    ),
    "Q141": (
        "How frequently do the following things occur in your neighborhood? Drug sale i"
        "n streets"
    ),
    "Q158": (
        "comfortable Now, I would like to read some statements and ask how much you agr"
        "ee or disagree with each of these statements. For these questions, a 1 means t"
        "hat you “completely disagree” and a 10 means that you “completely agree.” Scie"
        "nce and technology are making our lives healthier, easier, and more comfortabl"
        "e"
    ),
    "Q159": (
        "generation Now, I would like to read some statements and ask how much you agre"
        "e or disagree with each of these statements. For these questions, a 1 means th"
        "at you “completely disagree” and a 10 means that you “completely agree.” Becau"
        "se of science and technology, there will be more opportunities for the next ge"
        "neration"
    ),
    "Q160": (
        "Now, I would like to read some statements and ask how much you agree or disagr"
        "ee with each of these statements. For these questions, a 1 means that you “com"
        "pletely disagree” and a 10 means that you “completely agree.” We depend too mu"
        "ch on science and not enough on faith"
    ),
    "Q161": (
        "wrong Now, I would like to read some statements and ask how much you agree or "
        "disagree with each of these statements. For these questions, a 1 means that yo"
        "u “completely disagree” and a 10 means that you “completely agree”: One of the"
        " bad effects of science is that it breaks down people’s ideas of right and wro"
        "ng"
    ),
    "Q162": (
        "Now, I would like to read some statements and ask how much you agree or disagr"
        "ee with each of these statements. For these questions, a 1 means that you “com"
        "pletely disagree” and a 10 means that you “completely agree”: It is not import"
        "ant for me to know about science in my daily life"
    ),
    "Q163": (
        "All things considered, would you say that the world is better off, or worse of"
        "f, because of science and technology? Please tell me which comes closest to yo"
        "ur view on this scale: 1 means that “the world is a lot worse off,” and 10 mea"
        "ns that “the world is a lot better off.” The world is better off, or worse off"
        ", because of science and technology"
    ),
    "Q164": (
        "How important is God in your life?. Please use this scale to indicate. 10 mean"
        "s “very important” and 1 means “not at all important”."
    ),
    "Q165": "In which of the following things do you believe, if you believe in any? - God",
    "Q166": (
        "In which of the following things do you believe, if you believe in any? - Life"
        " after death"
    ),
    "Q167": "In which of the following things do you believe, if you believe in any? - Hell",
    "Q168": "In which of the following things do you believe, if you believe in any? - Heaven",
    "Q169": (
        "Please tell us if you strongly agree, agree, disagree, or strongly disagree wi"
        "th the following statements: Whenever science and religion conflict, religion "
        "is always right"
    ),
    "Q170": (
        "Please tell us if you strongly agree, agree, disagree, or strongly disagree wi"
        "th the following statements: The only acceptable religion is my religion"
    ),
    "Q173": "Independently of whether you go to church or not, would you say you are…",
    "Q174": (
        "other people With which one of the following statements do you agree most? The"
        " meaning of religion is…"
    ),
    "Q175": (
        "this world And with which of the following statements do you agree most? The m"
        "eaning of religion is…"
    ),
    "Q176": (
        "are the right ones to follow How much do you agree or disagree with the statem"
        "ent that nowadays one often has trouble deciding which moral rules are the rig"
        "ht ones to follow?"
    ),
    "Q177": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be  justified, or  something in  between, using  thi"
        "s  card.  -  Claiming government benefits to which you are not entitled"
    ),
    "Q178": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd.  - Avoiding a fare on public transport"
    ),
    "Q179": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Stealing property"
    ),
    "Q180": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Cheating on taxes if you have a chance"
    ),
    "Q181": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Someone accepting a bribe in the course of their duties"
    ),
    "Q182": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Homosexuality"
    ),
    "Q183": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Prostitution"
    ),
    "Q184": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Abortion"
    ),
    "Q185": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Divorce"
    ),
    "Q186": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Sex before marriage"
    ),
    "Q187": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Suicide"
    ),
    "Q188": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Euthanasia"
    ),
    "Q189": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. For a man to beat his wife"
    ),
    "Q190": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Parents beating children"
    ),
    "Q191": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Violence against other people"
    ),
    "Q192": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Terrorism as a political, ideological or religious mean"
    ),
    "Q193": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Having casual sex"
    ),
    "Q194": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Political violence"
    ),
    "Q195": (
        "Please tell me for each of the following statements whether you think it can a"
        "lways be justified, never be justified, or something in between, using this ca"
        "rd. Death penalty"
    ),
    "Q196": (
        "Do you think this country’s government should or should not have the right to "
        "do the following: Keep people under video surveillance in public areas"
    ),
    "Q197": (
        "exchanged on the Internet Do you think this country’s government should or sho"
        "uld not have the right to do the following: Monitor all emails and any other i"
        "nformation exchanged on the Internet"
    ),
    "Q198": (
        "country without their knowledge Do you think that this country’s government sh"
        "ould or should not have the right to do the following: Collect information abo"
        "ut anyone living in  this country without their knowledge"
    ),
    "Q199": "How interested would you say you are in politics?",
    "Q224": (
        "In your view, how often do the following things occur in this country’s electi"
        "ons? Votes are counted fairly"
    ),
    "Q225": (
        "running In  your view,  how  often do  the following  things occur in  this co"
        "untry’s  elections? Opposition candidates are prevented from running"
    ),
    "Q226": (
        "In your view, how often do the following things occur in this country’s electi"
        "ons? TV news favors the governing party"
    ),
    "Q227": (
        "In your view, how often do the following things occur in this country’s electi"
        "ons? Voters are bribed"
    ),
    "Q228": (
        "In  your view,  how  often do  the following  things occur in  this country’s "
        " elections? Journalists provide fair coverage of elections"
    ),
    "Q229": (
        "In  your view,  how  often do  the following  things occur in  this country’s "
        " elections? Election officials are fair"
    ),
    "Q230": (
        "In your view, how often do the following things occur in this country’s electi"
        "ons? Rich people buy elections"
    ),
    "Q231": (
        "In your view, how often do the following things occur in this country’s electi"
        "ons? Voters are threatened with violence at the polls"
    ),
    "Q232": (
        "elections In your view, how often do the following things occur in this countr"
        "y’s elections? Voters are offered a genuine choice in the elections"
    ),
    "Q233": (
        "In  your view,  how  often do  the following things occur in  this country’s e"
        "lections?: Women have equal opportunities to run the office"
    ),
    "Q234": (
        "lives;  other people think that it doesn’t matter much Do you think that hones"
        "t elections play an important role in deciding whether you and your family are"
        " able to make a good living?. How important would you say this is —very import"
        "ant, fairly important, not very important or not at all important?"
    ),
    "Q234A": (
        "How much would you say the political system in your country allows people like"
        " you to have a say in what the government does?"
    ),
    "Q235": (
        "parliament and elections I'm going to describe various types of political syst"
        "ems and ask what you think about each as a way of governing this country. For "
        "each one, would you say it is a very good, fairly good, fairly bad or very bad"
        " way of governing this country? Having a strong leader who does not have to bo"
        "ther with parliament and elections"
    ),
    "Q236": (
        "what they think is best for the country I'm going to describe various types of"
        " political systems and ask what you think about each as a way of governing thi"
        "s country. For each one, would you say it is a very good, fairly good, fairly "
        "bad or very bad way of governing this country? Having experts, not government,"
        " make decisions according to what they think is best for the country"
    ),
    "Q237": (
        "I'm going to describe various types of political systems and ask what you thin"
        "k about each as a way of governing this country. For each one, would you say i"
        "t is a very good, fairly good, fairly bad or very bad way of governing this co"
        "untry? Having the army rule"
    ),
    "Q238": (
        "I'm going to describe various types of political systems and ask what you thin"
        "k about each as a way of governing this country. For each one, would you say i"
        "t is a very good, fairly good, fairly bad or very bad way of governing this co"
        "untry? Having a democratic political system"
    ),
    "Q239": (
        "political parties or elections I'm going to describe various types of politica"
        "l systems and ask what you think about each as a way of governing this country"
        ". For each one, would you say it is a very good, fairly good, fairly bad or ve"
        "ry bad way of governing this country? Having a system governed by religious la"
        "w in which there are no political parties or elections"
    ),
    "Q240": (
        'In political matters, people talk of "the left" and "the right." How would'
        " you place your views on this scale, generally speaking?"
    ),
    "Q241": (
        "Please tell  me  for  each  of  the  following things how  essential you  thin"
        "k it  is  as  a characteristic of democracy. Use  this scale where 1  means “n"
        "ot at  all  an  essential characteristic of democracy” and 10 means it definit"
        "ely is “an essential characteristic of democracy. - Governments tax the rich a"
        "nd subsidize the poor"
    ),
    "Q242": (
        "Please tell  me  for  each  of  the  following things how  essential you  thin"
        "k it  is  as  a characteristic of democracy. Use  this scale where 1  means “n"
        "ot at  all  an essential characteristic of democracy” and 10 means it definite"
        "ly is “an essential characteristic of democracy. - Religious authorities inter"
        "pret the laws"
    ),
    "Q243": (
        "Please tell  me  for  each  of  the  following things how  essential you  thin"
        "k it  is  as  a characteristic of democracy. Use  this scale where 1  means “n"
        "ot at  all  an essential characteristic of democracy” and 10 means it definite"
        "ly is “an essential characteristic of democracy. - People choose their leaders"
        " in free elections"
    ),
    "Q244": (
        "Please tell  me  for  each  of  the  following things how  essential you  thin"
        "k it  is  as  a characteristic of democracy. Use  this scale where 1  means “n"
        "ot at  all  an essential characteristic of democracy” and 10 means it definite"
        "ly is “an essential characteristic of democracy. - People receive state aid fo"
        "r unemployment"
    ),
    "Q245": (
        "Please tell  me  for  each  of  the  following things how  essential you  thin"
        "k it  is  as  a characteristic of democracy. Use  this scale where 1  means “n"
        "ot at  all  an essential characteristic of democracy” and 10 means it definite"
        "ly is “an essential characteristic of democracy. - The army takes over when go"
        "vernment is incompetent"
    ),
    "Q246": (
        "Please tell  me  for  each  of  the  following things how  essential you  thin"
        "k it  is  as  a characteristic of democracy. Use  this scale where 1  means “n"
        "ot at  all  an essential characteristic of democracy” and 10 means it definite"
        "ly is “an essential characteristic of democracy. - Civil rights protect people"
        "’s liberty from state oppression"
    ),
    "Q247": (
        "Please tell  me  for  each  of  the  following things how  essential you  thin"
        "k it  is  as  a characteristic of democracy. Use  this scale where 1  means “n"
        "ot at  all  an essential characteristic of democracy” and 10 means it definite"
        "ly is “an essential characteristic of democracy. - The state makes people´s in"
        "comes equal"
    ),
    "Q248": (
        "Please tell  me  for  each  of  the  following things how  essential you  thin"
        "k it  is  as  a characteristic of democracy. Use  this scale where 1  means “n"
        "ot at  all  an essential characteristic of democracy” and 10 means it definite"
        "ly is “an essential characteristic of democracy. - People obey their rulers"
    ),
    "Q249": (
        "Please tell  me  for  each  of  the  following things how  essential you  thin"
        "k it  is  as  a characteristic of democracy. Use  this scale where 1  means “n"
        "ot at  all  an essential characteristic of democracy” and 10 means it definite"
        "ly is “an essential characteristic of democracy. - Women have the same rights "
        "as men"
    ),
    "Q250": (
        "How important is it for you to live in a country that is governed democratical"
        "ly? On this scale where 1 means it is “not at all important” and 10 means “abs"
        "olutely important” what position would you choose?"
    ),
    "Q251": (
        "And how democratically is this country being governed today?. Again using a sc"
        "ale from 1  to  10,  where 1  means that it  is  “not at  all  democratic” and"
        "  10  means that it  is “completely democratic,” what position would you choos"
        "e?"
    ),
    "Q252": (
        "How satisfied are you with how the political system is functioning in your cou"
        "ntry these days? On the scale from 1 to 10 below, 1 means not satisfied at all"
        " and 10 means completely satisfied."
    ),
    "Q253": "How much respect is there for individual human rights nowadays in this country?",
    "Q254": "How proud are you to be of nationality of this country?",
    "Q255": (
        "People have different views about themselves and how they relate to the world."
        " Using this card, would you tell me how close do you feel to…? Your village, t"
        "own or city"
    ),
    "Q256": (
        "People have different views about themselves and how they relate to the world."
        " Using this card, would you tell me how strongly you agree or disagree with ea"
        "ch of the following statements about how you see yourself? Your county, region"
        ", district"
    ),
    "Q257": (
        "People have different views about themselves and how they relate to the world."
        " Using this card, would you  tell me  how  strongly you agree or  disagree wit"
        "h  each of  the following statements about how you see yourself? Country"
    ),
    "Q258": (
        "People have different views about themselves and how they relate to the world."
        " Using this card, would you tell me how close do you feel to…? Continent; e.g."
        " Europe, Asia etc."
    ),
    "Q259": (
        "People have different views about themselves and how they relate to the world."
        " Using this card, would you  tell me  how  strongly you agree or  disagree wit"
        "h  each of  the following statements about how you see yourself? World"
    ),
}
WVS_ITEM_CODES = {
    "Q1": (
        "Q1",
        {
            1: "Very important",
            2: "Rather important",
            3: "Not very important",
            4: "Not at all important",
        },
    ),
    "Q2": (
        "Q2",
        {
            1: "Very important",
            2: "Rather important",
            3: "Not very important",
            4: "Not at all important",
        },
    ),
    "Q3": (
        "Q3",
        {
            1: "Very important",
            2: "Rather important",
            3: "Not very important",
            4: "Not at all important",
        },
    ),
    "Q4": (
        "Q4",
        {
            1: "Very important",
            2: "Rather important",
            3: "Not very important",
            4: "Not at all important",
        },
    ),
    "Q5": (
        "Q5",
        {
            1: "Very important",
            2: "Rather important",
            3: "Not very important",
            4: "Not at all important",
        },
    ),
    "Q6": (
        "Q6",
        {
            1: "Very important",
            2: "Rather important",
            3: "Not very important",
            4: "Not at all important",
        },
    ),
    "Q27": (
        "Q27",
        {1: "Agree strongly", 2: "Agree", 3: "Disagree", 4: "Strongly disagree"},
    ),
    "Q28": (
        "Q28",
        {1: "Agree strongly", 2: "Agree", 3: "Disagree", 4: "Strongly disagree"},
    ),
    "Q29": (
        "Q29",
        {1: "Agree strongly", 2: "Agree", 3: "Disagree", 4: "Strongly disagree"},
    ),
    "Q30": (
        "Q30",
        {1: "Agree strongly", 2: "Agree", 3: "Disagree", 4: "Strongly disagree"},
    ),
    "Q31": (
        "Q31",
        {1: "Agree strongly", 2: "Agree", 3: "Disagree", 4: "Strongly disagree"},
    ),
    "Q32": (
        "Q32",
        {1: "Agree strongly", 2: "Agree", 3: "Disagree", 4: "Strongly disagree"},
    ),
    "Q33": (
        "Q33",
        {
            1: "Agree",
            2: "Disagree",
            3: "Neither agree nor disagree",
            4: "Disagree",
            5: "Disagree strongly",
        },
    ),
    "Q34": (
        "Q34",
        {
            1: "Agree",
            2: "Disagree",
            3: "Neither agree nor disagree",
            4: "Disagree",
            5: "Disagree strongly",
        },
    ),
    "Q35": (
        "Q35",
        {
            1: "Agree",
            2: "Disagree",
            3: "Neither agree nor disagree",
            4: "Disagree",
            5: "Disagree strongly",
        },
    ),
    "Q36": (
        "Q36",
        {
            1: "Agree strongly",
            2: "Agree",
            3: "Neither agree nor disagree",
            4: "Disagree",
            5: "Disagree strongly",
        },
    ),
    "Q37": (
        "Q37",
        {
            1: "Agree strongly",
            2: "Agree",
            3: "Neither agree nor disagree",
            4: "Disagree",
            5: "Disagree strongly",
        },
    ),
    "Q38": (
        "Q38",
        {
            1: "Agree strongly",
            2: "Agree",
            3: "Neither agree nor disagree",
            4: "Disagree",
            5: "Disagree strongly",
        },
    ),
    "Q39": (
        "Q39",
        {
            1: "Strongly agree",
            2: "Agree",
            3: "Neither agree or disagree",
            4: "Disagree",
            5: "Strongly disagree",
        },
    ),
    "Q40": (
        "Q40",
        {
            1: "Strongly agree",
            2: "Agree",
            3: "Neither agree or disagree",
            4: "Disagree",
            5: "Strongly disagree",
        },
    ),
    "Q41": (
        "Q41",
        {
            1: "Agree strongly",
            2: "Agree",
            3: "Neither agree nor disagree",
            4: "Disagree",
            5: "Disagree strongly",
        },
    ),
    "Q42": (
        "Q42",
        {
            1: (
                "The entire way our society is organized must be radically changed by r"
                "evolutionary"
            ),
            2: "Our society must be gradually improved by reforms",
            3: (
                "Our present society must be valiantly defended against all subversive "
                "forces"
            ),
        },
    ),
    "Q43": ("Q43", {1: "Good thing", 2: "Don´t mind", 3: "Bad thing"}),
    "Q44": ("Q44", {1: "Good thing", 2: "Don´t mind", 3: "Bad thing"}),
    "Q45": ("Q45", {1: "Good thing", 2: "Don´t mind", 3: "Bad thing"}),
    "Q46": (
        "Q46",
        {1: "Very happy", 2: "Quite happy", 3: "Not very happy", 4: "Not at all happy"},
    ),
    "Q47": ("Q47", {1: "Very good", 2: "Good", 3: "Fair", 4: "Poor", 5: "Very poor"}),
    "Q48": (
        "Q48",
        {
            1: "None at all",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "A great deal",
        },
    ),
    "Q49": (
        "Q49",
        {
            1: "Completely dissatisfied",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Completely satisfied",
        },
    ),
    "Q50": (
        "Q50",
        {
            1: "Dissatisfied",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Satisfied",
        },
    ),
    "Q56": ("Q56", {1: "Better off", 2: "Worse off", 3: "Or about the same"}),
    "Q57": ("Q57", {1: "Most people can be trusted", 2: "Need to be very careful"}),
    "Q58": (
        "Q58",
        {
            1: "Trust completely",
            2: "Trust somewhat",
            3: "Do not trust very much",
            4: "Do not trust at all",
        },
    ),
    "Q59": (
        "Q59",
        {
            1: "Trust completely",
            2: "Trust somewhat",
            3: "Do not trust very much",
            4: "Do not trust at all",
        },
    ),
    "Q60": (
        "Q60",
        {
            1: "Trust completely",
            2: "Trust somewhat",
            3: "Do not trust very much",
            4: "Do not trust at all",
        },
    ),
    "Q61": (
        "Q61",
        {
            1: "Trust completely",
            2: "Trust somewhat",
            3: "Do not trust very much",
            4: "Do not trust at all",
        },
    ),
    "Q62": (
        "Q62",
        {
            1: "Trust completely",
            2: "Trust somewhat",
            3: "Do not trust very much",
            4: "Do not trust at all",
        },
    ),
    "Q63": (
        "Q63",
        {
            1: "Trust completely",
            2: "Trust somewhat",
            3: "Do not trust very much",
            4: "Do not trust at all",
        },
    ),
    "Q64": (
        "Q64",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q65": (
        "Q65",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q66": (
        "Q66",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q67": (
        "Q67",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q68": (
        "Q68",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q69": (
        "Q69",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q70": (
        "Q70",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q71": (
        "Q71",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q72": (
        "Q72",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q73": (
        "Q73",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q74": (
        "Q74",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q75": (
        "Q75",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q76": (
        "Q76",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q77": (
        "Q77",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q78": (
        "Q78",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q79": (
        "Q79",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q80": (
        "Q80",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q81": (
        "Q81",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q82": (
        "Q82",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q83": (
        "Q83",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q84": (
        "Q84",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q85": (
        "Q85",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q86": (
        "Q86",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q87": (
        "Q87",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q88": (
        "Q88",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q89": (
        "Q89",
        {1: "A great deal", 2: "Quite a lot", 3: "Not very much", 4: "None at all"},
    ),
    "Q90": (
        "Q90",
        {
            1: "Being effective",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Being democratic",
        },
    ),
    "Q106": (
        "Q106",
        {
            1: "Incomes more equal",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Larger income differences",
        },
    ),
    "Q107": (
        "Q107",
        {
            1: "Private ownership of business should be increased",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Government ownership of business should be increased",
        },
    ),
    "Q108": (
        "Q108",
        {
            1: "The government should take more responsibility to ensure that everyone is",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "People should take more responsibility to provide for themselves",
        },
    ),
    "Q109": (
        "Q109",
        {
            1: "Competition is good",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Competition is harmful",
        },
    ),
    "Q110": (
        "Q110",
        {
            1: "In the long run, hard work usually brings a better life",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Hard work doesn´t generally bring success - it´s more a matter of luck and",
        },
    ),
    "Q111": (
        "Q111",
        {1: "A: Protect environment", 2: "B: Economic growth", 3: "Other answer"},
    ),
    "Q112": (
        "Q112",
        {
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "10",
        },
    ),
    "Q113": (
        "Q113",
        {1: "None of them", 2: "Few of them", 3: "Most of them", 4: "All of them"},
    ),
    "Q114": (
        "Q114",
        {1: "None of them", 2: "Few of them", 3: "Most of them", 4: "All of them"},
    ),
    "Q115": (
        "Q115",
        {1: "None of them", 2: "Few of them", 3: "Most of them", 4: "All of them"},
    ),
    "Q116": (
        "Q116",
        {1: "None of them", 2: "Few of them", 3: "Most of them", 4: "All of them"},
    ),
    "Q117": (
        "Q117",
        {1: "None of them", 2: "Few of them", 3: "Most of them", 4: "All of them"},
    ),
    "Q118": ("Q118", {1: "Never", 2: "Rarely", 3: "Frequently", 4: "Always"}),
    "Q119": (
        "Q119",
        {1: "Strongly agree", 2: "Agree", 3: "Disagree", 4: "Strongly disagree"},
    ),
    "Q120": (
        "Q120",
        {
            1: "No risk at all",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Very high risk",
        },
    ),
    "Q130": (
        "Q130",
        {
            1: "Let anyone come who wants to",
            2: "Let people come as long as there are jobs available",
            3: "Place strict limits on the number of foreigners who can come here",
            4: "Prohibit people coming here from other countries",
        },
    ),
    "Q141": ("Q141", {1: "Yes", 2: "No"}),
    "Q158": (
        "Q158",
        {
            1: "Completely disagree",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Completely agree",
        },
    ),
    "Q159": (
        "Q159",
        {
            1: "Completely disagree",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Completely agree",
        },
    ),
    "Q160": (
        "Q160",
        {
            1: "Completely disagree",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Completely agree",
        },
    ),
    "Q161": (
        "Q161",
        {
            1: "Completely disagree",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Completely agree",
        },
    ),
    "Q162": (
        "Q162",
        {
            1: "Completely disagree",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Completely agree",
        },
    ),
    "Q163": (
        "Q163",
        {
            1: "A lot worse off",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "A lot better off",
        },
    ),
    "Q164": (
        "Q164",
        {
            1: "Not at all important",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Very important",
        },
    ),
    "Q165": ("Q165", {1: "Yes", 2: "No"}),
    "Q166": ("Q166", {1: "Yes", 2: "No"}),
    "Q167": ("Q167", {1: "Yes", 2: "No"}),
    "Q168": ("Q168", {1: "Yes", 2: "No"}),
    "Q169": (
        "Q169",
        {1: "Strongly agree", 2: "Agree", 3: "Disagree", 4: "Strongly disagree"},
    ),
    "Q170": (
        "Q170",
        {1: "Strongly agree", 2: "Agree", 3: "Disagree", 4: "Strongly disagree"},
    ),
    "Q173": (
        "Q173",
        {1: "A religious person", 2: "Not a religious person", 3: "An atheist"},
    ),
    "Q174": (
        "Q174",
        {1: "Follow religious norms and ceremonies", 2: "Do good to other people"},
    ),
    "Q175": (
        "Q175",
        {1: "Make sense of life after death", 2: "Make sense of life in this world"},
    ),
    "Q176": (
        "Q176",
        {
            1: "Completely agree",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Completely disagree",
        },
    ),
    "Q177": (
        "Q177",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q178": (
        "Q178",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q179": (
        "Q179",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q180": (
        "Q180",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q181": (
        "Q181",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q182": (
        "Q182",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q183": (
        "Q183",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q184": (
        "Q184",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q185": (
        "Q185",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q186": (
        "Q186",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q187": (
        "Q187",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q188": (
        "Q188",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q189": (
        "Q189",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q190": (
        "Q190",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q191": (
        "Q191",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q192": (
        "Q192",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q193": (
        "Q193",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q194": (
        "Q194",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q195": (
        "Q195",
        {
            1: "Never justifiable",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Always justifiable",
        },
    ),
    "Q196": (
        "Q196",
        {
            1: "Definitely should have the right",
            2: "Probably should have the right",
            3: "Probably should not have the right",
            4: "Definitely should not have the right",
        },
    ),
    "Q197": (
        "Q197",
        {
            1: "Definitely should have the right",
            2: "Probably should have the right",
            3: "Probably should not have the right",
            4: "Definitely should not have the right",
        },
    ),
    "Q198": (
        "Q198",
        {
            1: "Definitely should have the right",
            2: "Probably should have the right",
            3: "Probably should not have the right",
            4: "Definitely should not have the right",
        },
    ),
    "Q199": (
        "Q199",
        {
            1: "Very interested",
            2: "Somewhat interested",
            3: "Not very interested",
            4: "Not at all interested",
        },
    ),
    "Q224": (
        "Q224",
        {1: "Very often", 2: "Fairly often", 3: "Not often", 4: "Not at all often"},
    ),
    "Q225": (
        "Q225",
        {1: "Very often", 2: "Fairly often", 3: "Not often", 4: "Not at all often"},
    ),
    "Q226": (
        "Q226",
        {1: "Very often", 2: "Fairly often", 3: "Not often", 4: "Not at all often"},
    ),
    "Q227": (
        "Q227",
        {1: "Very often", 2: "Fairly often", 3: "Not often", 4: "Not at all often"},
    ),
    "Q228": (
        "Q228",
        {1: "Very often", 2: "Fairly often", 3: "Not often", 4: "Not at all often"},
    ),
    "Q229": (
        "Q229",
        {1: "Very often", 2: "Fairly often", 3: "Not often", 4: "Not at all often"},
    ),
    "Q230": (
        "Q230",
        {1: "Very often", 2: "Fairly often", 3: "Not often", 4: "Not at all often"},
    ),
    "Q231": (
        "Q231",
        {1: "Very often", 2: "Fairly often", 3: "Not often", 4: "Not at all often"},
    ),
    "Q232": (
        "Q232",
        {1: "Very often", 2: "Fairly often", 3: "Not often", 4: "Not at all often"},
    ),
    "Q233": (
        "Q233",
        {1: "Very often", 2: "Fairly often", 3: "Not often", 4: "Not at all often"},
    ),
    "Q234": (
        "Q234",
        {
            1: "Very important",
            2: "Rather important",
            3: "Not very important",
            4: "Not at all important",
        },
    ),
    "Q234A": (
        "Q234A",
        {1: "A great deal", 2: "A lot", 3: "Some", 4: "Very little", 5: "Not at all"},
    ),
    "Q235": (
        "Q235",
        {1: "Very good", 2: "Fairly good", 3: "Fairly bad", 4: "Very bad"},
    ),
    "Q236": (
        "Q236",
        {1: "Very good", 2: "Fairly good", 3: "Fairly bad", 4: "Very bad"},
    ),
    "Q237": (
        "Q237",
        {1: "Very good", 2: "Fairly good", 3: "Fairly bad", 4: "Very bad"},
    ),
    "Q238": (
        "Q238",
        {1: "Very good", 2: "Fairly good", 3: "Fairly bad", 4: "Very bad"},
    ),
    "Q239": ("Q239", {1: "Very good", 2: "Fairly good", 3: "Bad", 4: "Very bad"}),
    "Q240": (
        "Q240",
        {
            1: "Left",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Right",
        },
    ),
    "Q241": (
        "Q241",
        {
            1: "Not an essential characteristic of democracy",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "An essential characteristic of democracy",
        },
    ),
    "Q242": (
        "Q242",
        {
            1: "Not an essential characteristic of democracy",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "An essential characteristic of democracy",
        },
    ),
    "Q243": (
        "Q243",
        {
            1: "Not an essential characteristic of democracy",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "An essential characteristic of democracy",
        },
    ),
    "Q244": (
        "Q244",
        {
            1: "Not an essential characteristic of democracy",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "An essential characteristic of democracy",
        },
    ),
    "Q245": (
        "Q245",
        {
            1: "Not an essential characteristic of democracy",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "An essential characteristic of democracy",
        },
    ),
    "Q246": (
        "Q246",
        {
            1: "Not an essential characteristic of democracy",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "An essential characteristic of democracy",
        },
    ),
    "Q247": (
        "Q247",
        {
            1: "Not an essential characteristic of democracy",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "An essential characteristic of democracy",
        },
    ),
    "Q248": (
        "Q248",
        {
            1: "Not an essential characteristic of democracy",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "An essential characteristic of democracy",
        },
    ),
    "Q249": (
        "Q249",
        {
            1: "Not an essential characteristic of democracy",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "An essential characteristic of democracy",
        },
    ),
    "Q250": (
        "Q250",
        {
            1: "Not at all important",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Absolutely important",
        },
    ),
    "Q251": (
        "Q251",
        {
            1: "Not at all democratic",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Completely democratic",
        },
    ),
    "Q252": (
        "Q252",
        {
            1: "Not satisfied at all",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "Completely satisfied",
        },
    ),
    "Q253": (
        "Q253",
        {
            1: "A great deal of respect",
            2: "Fairly much  respect",
            3: "Not much respect",
            4: "No respect at all",
        },
    ),
    "Q254": (
        "Q254",
        {
            1: "Very proud",
            2: "Quite proud",
            3: "Not very proud",
            4: "Not at all proud",
            5: "I am not [nationality]",
        },
    ),
    "Q255": (
        "Q255",
        {1: "Very close", 2: "Close", 3: "Not very close", 4: "Not close at all"},
    ),
    "Q256": (
        "Q256",
        {1: "Very close", 2: "Close", 3: "Not very close", 4: "Not close at all"},
    ),
    "Q257": (
        "Q257",
        {1: "Very close", 2: "Close", 3: "Not very close", 4: "Not close at all"},
    ),
    "Q258": (
        "Q258",
        {1: "Very close", 2: "Close", 3: "Not very close", 4: "Not close at all"},
    ),
    "Q259": (
        "Q259",
        {
            1: "No",
            2: "Yes, own parent(s)",
            3: "Yes, parent(s) in law",
            4: "Yes, both own parent(s) and parent(s) in law",
        },
    ),
}
ITEM_VARS = list(ITEM_QUESTIONS.keys())
SEX_VAR = "Q260"
AGE_VAR = "Q262"
EDU_VAR = "Q275R"
COUNTRY_VAR = "B_COUNTRY_ALPHA"
YEAR_VAR = "A_YEAR"
URBRURAL_VAR = "H_URBRURAL"
SEX_LABELS = {1: "man", 2: "woman"}
EDU_LABELS = {
    1: "lower (up to lower-secondary school)",
    2: "middle (upper-secondary or post-secondary)",
    3: "higher (university-level)",
}
URBRURAL_LABELS = {1: "an urban area", 2: "a rural area"}
AGE_BUCKET_EDGES = [16, 30, 45, 60, 200]
AGE_BUCKET_LABELS = ["16-29", "30-44", "45-59", "60+"]
COUNTRY_NAMES = {
    "AND": "Andorra",
    "ARG": "Argentina",
    "ARM": "Armenia",
    "AUS": "Australia",
    "BGD": "Bangladesh",
    "BOL": "Bolivia",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CHL": "Chile",
    "CHN": "China",
    "COL": "Colombia",
    "CYP": "Cyprus",
    "CZE": "Czechia",
    "DEU": "Germany",
    "ECU": "Ecuador",
    "EGY": "Egypt",
    "ETH": "Ethiopia",
    "GBR": "Great Britain",
    "GRC": "Greece",
    "GTM": "Guatemala",
    "HKG": "Hong Kong",
    "IDN": "Indonesia",
    "IND": "India",
    "IRN": "Iran",
    "IRQ": "Iraq",
    "JOR": "Jordan",
    "JPN": "Japan",
    "KAZ": "Kazakhstan",
    "KEN": "Kenya",
    "KGZ": "Kyrgyzstan",
    "KOR": "South Korea",
    "LBN": "Lebanon",
    "LBY": "Libya",
    "MAC": "Macau",
    "MAR": "Morocco",
    "MDV": "the Maldives",
    "MEX": "Mexico",
    "MMR": "Myanmar",
    "MNG": "Mongolia",
    "MYS": "Malaysia",
    "NGA": "Nigeria",
    "NIC": "Nicaragua",
    "NIR": "Northern Ireland",
    "NLD": "the Netherlands",
    "NZL": "New Zealand",
    "PAK": "Pakistan",
    "PER": "Peru",
    "PHL": "the Philippines",
    "PRI": "Puerto Rico",
    "ROU": "Romania",
    "RUS": "Russia",
    "SGP": "Singapore",
    "SRB": "Serbia",
    "SVK": "Slovakia",
    "THA": "Thailand",
    "TJK": "Tajikistan",
    "TUN": "Tunisia",
    "TUR": "Turkey",
    "TWN": "Taiwan",
    "UKR": "Ukraine",
    "URY": "Uruguay",
    "USA": "the United States",
    "UZB": "Uzbekistan",
    "VEN": "Venezuela",
    "VNM": "Vietnam",
    "ZWE": "Zimbabwe",
}


def age_bucket_of(age: float) -> str:
    """Returns the age-bucket label for a respondent age.

    Args:
        age: Respondent age in years.

    Returns:
        Age-bucket label from AGE_BUCKET_LABELS. Ages outside the edges fall
        through to the last label.
    """
    for lo, hi, label in zip(
        AGE_BUCKET_EDGES[:-1],
        AGE_BUCKET_EDGES[1:],
        AGE_BUCKET_LABELS,
        strict=False,
    ):
        if lo <= age < hi:
            return label
    return AGE_BUCKET_LABELS[-1]


def cell_id_of(row: pd.Series) -> str:
    """Builds the demographic-group identifier for a respondent row.

    Args:
        row: Respondent row whose CELL_VARS columns are already labels.

    Returns:
        Pipe-joined group identifier.
    """
    return "|".join(str(row[var]) for var in CELL_VARS)


def build_persona(row: pd.Series) -> str:
    """Builds the persona sentence for a respondent's demographic group.

    Args:
        row: Respondent row whose CELL_VARS columns are already labels.

    Returns:
        Persona text placed before the question.
    """
    return (
        f"It is {row['year']}. You live in {row['urbrural']} of "
        f"{row['country']}. You are a "
        f"{row['age']} year-old {row['sex']}. Your highest level of education "
        f"is {row['education']}."
    )


def select_wvs_subset(
    all_countries: list[str], all_items: list[str], seed: int = SEED
) -> dict:
    """Selects a reproducible one-third of countries and one-third of items.

    Countries are drawn from a sorted list and items from the fixed item order.
    Kept items are returned in their original order. The two draws are the
    first and second draws from ``numpy`` Generator ``seed``.

    Args:
        all_countries: Country labels present after group-variable cleaning.
        all_items: Ordered measurable item codes.
        seed: RNG seed. The survey build uses 42.

    Returns:
        Seed, kept counts, sorted country labels, and item codes in original
        order.
    """
    rng = np.random.default_rng(seed)
    sorted_countries = sorted(all_countries)
    n_country_keep = len(sorted_countries) // SUBSET_FRACTION_DENOM
    n_item_keep = len(all_items) // SUBSET_FRACTION_DENOM
    assert n_country_keep > 0 and n_item_keep > 0, (
        "subset fraction collapses to an empty set"
    )

    chosen_countries = sorted(
        rng.choice(sorted_countries, size=n_country_keep, replace=False).tolist()
    )
    chosen_item_set = set(
        rng.choice(all_items, size=n_item_keep, replace=False).tolist()
    )
    chosen_items = [item for item in all_items if item in chosen_item_set]

    return {
        "seed": seed,
        "fraction_denominator": SUBSET_FRACTION_DENOM,
        "n_countries_full": len(sorted_countries),
        "n_countries_kept": len(chosen_countries),
        "countries": chosen_countries,
        "n_items_full": len(all_items),
        "n_items_kept": len(chosen_items),
        "items": chosen_items,
    }


__all__ = [
    "ANSWER_LETTERS",
    "SEED",
    "build_chat_messages",
    "build_question_block",
    "DATASET",
    "SUBSET_FRACTION_DENOM",
    "CELL_VARS",
    "ORIGINAL_7_WVS_VARS",
    "ITEM_QUESTIONS",
    "WVS_ITEM_CODES",
    "ITEM_VARS",
    "select_wvs_subset",
    "age_bucket_of",
    "cell_id_of",
    "build_persona",
]
